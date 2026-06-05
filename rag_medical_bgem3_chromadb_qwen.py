#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
檔名：rag_medical_bgem3_chromadb_qwen.py

用途：
這是一支完整的 RAG 醫療問答程式。
它會讀取台灣 e 院醫療問答 CSV，建立向量資料庫，並用 Qwen Instruct 產生回答。

資料格式：
CSV 必須包含以下欄位：
page, 問題編號, question_title, question_text, answer_text

本程式流程：
1. 使用 BAAI/bge-m3 建立 embedding
2. 將向量存入 ChromaDB
3. 使用相似度搜尋取得 Top-K 問答
4. 使用 Qwen Instruct LLM 產生回答
5. 使用 Recall@K、MRR、ROUGE-L 評估

安裝套件：
pip install pandas tqdm torch sentence-transformers chromadb transformers accelerate

若要用 4-bit 省 GPU 記憶體，可再安裝：
pip install bitsandbytes

範例執行：

一、建立 ChromaDB 向量資料庫
python rag_medical_bgem3_chromadb_qwen.py \
  --mode build \
  --csv "merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line(2).csv" \
  --persist_dir chroma_db \
  --rebuild

二、單題 RAG 問答
python rag_medical_bgem3_chromadb_qwen.py \
  --mode chat \
  --question "排卵期前出血需要看醫生嗎？" \
  --persist_dir chroma_db

三、評估檢索效果
python rag_medical_bgem3_chromadb_qwen.py \
  --mode eval_retrieval \
  --csv "merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line(2).csv" \
  --persist_dir chroma_db \
  --eval_sample_size 300

四、評估生成效果
python rag_medical_bgem3_chromadb_qwen.py \
  --mode eval_generation \
  --csv "merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line(2).csv" \
  --persist_dir chroma_db \
  --eval_sample_size 20

五、完整流程：建庫 + 檢索評估 + 生成評估
python rag_medical_bgem3_chromadb_qwen.py \
  --mode all \
  --csv "merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line(2).csv" \
  --persist_dir chroma_db \
  --rebuild
"""

import argparse
import os
import random
import re
import shutil
from typing import Dict, List, Tuple

import chromadb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 0. 必要欄位設定
# ============================================================

REQUIRED_COLUMNS = [
    "page",
    "問題編號",
    "question_title",
    "question_text",
    "answer_text",
]


# ============================================================
# 1. 文字與 CSV 前處理
# ============================================================

def clean_text(text) -> str:
    """
    清理文字。
    目標是讓資料比較乾淨，避免奇怪符號影響 embedding。
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # 移除台灣 e 院資料中常見的特殊符號
    text = text.replace("\uf0a4", " ")

    # 把換行、Tab、多個空白整理成一個空白
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_medical_csv(csv_path: str) -> pd.DataFrame:
    """
    讀取醫療問答 CSV。
    同時檢查必要欄位是否存在。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 檔案：{csv_path}")

    df = pd.read_csv(csv_path)

    # 檢查欄位
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV 缺少欄位：{missing_columns}")

    # 只保留本專案需要的欄位
    df = df[REQUIRED_COLUMNS].copy()

    # 清理三個主要文字欄位
    df["question_title"] = df["question_title"].apply(clean_text)
    df["question_text"] = df["question_text"].apply(clean_text)
    df["answer_text"] = df["answer_text"].apply(clean_text)

    # 移除沒有問題或沒有答案的資料
    df = df[(df["question_text"] != "") & (df["answer_text"] != "")].copy()

    # 重設 index
    df = df.reset_index(drop=True)

    # 建立固定文件 ID
    # 之後評估 Recall@K 和 MRR 會用到
    df["doc_id"] = [f"qa_{i:06d}" for i in range(len(df))]

    return df


def build_document(row: pd.Series) -> str:
    """
    把一筆問答資料合成一段文件。
    這段文件會被轉成 embedding，放入 ChromaDB。
    """
    document = (
        f"問題標題：{row['question_title']}\n"
        f"病患提問：{row['question_text']}\n"
        f"醫師回答：{row['answer_text']}"
    )
    return document


def build_query(row: pd.Series) -> str:
    """
    建立查詢文字。
    評估時會用問題標題 + 病患提問去檢索。
    """
    query = (
        f"問題標題：{row['question_title']}\n"
        f"病患提問：{row['question_text']}"
    )
    return query


# ============================================================
# 2. 載入 BAAI/bge-m3 embedding 模型
# ============================================================

def load_embedding_model(model_name: str, device: str):
    """
    載入 embedding 模型。
    預設使用 BAAI/bge-m3。
    bge-m3 適合中文與多語檢索。
    """
    print("\n[Embedding] 載入模型")
    print(f"模型名稱：{model_name}")
    print(f"使用裝置：{device}")

    model = SentenceTransformer(model_name, device=device)

    return model


def encode_texts(
    model,
    texts: List[str],
    batch_size: int = 16,
) -> List[List[float]]:
    """
    將多段文字轉成 embedding 向量。
    normalize_embeddings=True 代表把向量正規化。
    這樣比較適合 cosine similarity。
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


# ============================================================
# 3. ChromaDB 建立與查詢
# ============================================================

def get_collection(
    persist_dir: str,
    collection_name: str,
):
    """
    取得 ChromaDB collection。
    collection 可以想成一張向量資料表。
    """
    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return client, collection


def rebuild_collection(
    persist_dir: str,
    collection_name: str,
):
    """
    重新建立 ChromaDB collection。
    如果舊資料存在，就先刪除。
    """
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(collection_name)
        print(f"[ChromaDB] 已刪除舊 collection：{collection_name}")
    except Exception:
        print(f"[ChromaDB] 沒有舊 collection 可刪除：{collection_name}")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return client, collection


def build_chroma_index(
    csv_path: str,
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    batch_size: int,
    device: str,
    rebuild: bool,
):
    """
    建立向量資料庫。
    這是 RAG 的資料準備階段。
    """
    print("\n========== 建立 ChromaDB 向量資料庫 ==========")

    # 讀取 CSV
    df = load_medical_csv(csv_path)
    print(f"[資料] 有效問答筆數：{len(df)}")

    # 建立或重建 collection
    if rebuild:
        _, collection = rebuild_collection(persist_dir, collection_name)
    else:
        _, collection = get_collection(persist_dir, collection_name)

    # 如果 collection 已經有資料，而且沒有要求重建，就不重複寫入
    if collection.count() > 0 and not rebuild:
        print(f"[ChromaDB] 已有 {collection.count()} 筆資料")
        print("[提醒] 若要重新建立，請加上 --rebuild")
        return

    # 載入 embedding 模型
    embedding_model = load_embedding_model(embedding_model_name, device)

    # 將每筆資料整理成 document
    documents = [build_document(row) for _, row in df.iterrows()]

    # 建立 ChromaDB 的 ID
    ids = df["doc_id"].tolist()

    # metadata 是附加資訊，方便之後顯示來源
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append(
            {
                "doc_id": row["doc_id"],
                "page": int(row["page"]),
                "question_id": str(row["問題編號"]),
                "question_title": row["question_title"],
            }
        )

    # 建立 embedding
    print("[Embedding] 開始將問答轉成向量")
    embeddings = encode_texts(
        model=embedding_model,
        texts=documents,
        batch_size=batch_size,
    )

    # 寫入 ChromaDB
    print("[ChromaDB] 開始寫入資料")
    for start in tqdm(range(0, len(documents), batch_size), desc="寫入 ChromaDB"):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"[完成] 已建立 {collection.count()} 筆向量資料")
    print(f"[位置] {persist_dir}")


def search_top_k(
    question: str,
    persist_dir: str,
    collection_name: str,
    embedding_model,
    top_k: int,
) -> List[Dict]:
    """
    使用相似度搜尋取得 Top-K 問答。
    Top-K 代表最相似的前 K 筆資料。
    """
    _, collection = get_collection(persist_dir, collection_name)

    # 把使用者問題轉成 embedding
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    # 到 ChromaDB 查詢最相似的資料
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        # cosine distance 越小越相似
        # similarity = 1 - distance，越大越相似
        similarity = 1.0 - float(distance)

        output.append(
            {
                "document": doc,
                "metadata": meta,
                "distance": float(distance),
                "similarity": similarity,
            }
        )

    return output


# ============================================================
# 4. 載入 Qwen Instruct LLM 並生成回答
# ============================================================

def load_qwen_model(llm_model_name: str, load_in_4bit: bool):
    """
    載入 Qwen Instruct 模型。
    RTX6000 通常可使用 Qwen2.5-7B-Instruct。
    若 GPU 記憶體不足，可以使用 --load_in_4bit 或改用 3B 模型。
    """
    print("\n[LLM] 載入 Qwen Instruct 模型")
    print(f"模型名稱：{llm_model_name}")
    print(f"4-bit 量化：{load_in_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        trust_remote_code=True,
    )

    if load_in_4bit:
        # 4-bit 可以降低 GPU 記憶體需求
        # 需要先安裝 bitsandbytes
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()

    return tokenizer, model


def build_rag_prompt(question: str, retrieved_docs: List[Dict]) -> str:
    """
    建立給 LLM 的 prompt。
    我們要求模型只能根據參考資料回答。
    這樣可以降低幻覺，也就是模型亂編答案的問題。
    """
    context_blocks = []

    for i, item in enumerate(retrieved_docs, start=1):
        meta = item["metadata"]

        context_block = (
            f"【參考資料 {i}】\n"
            f"問題編號：{meta.get('question_id', '')}\n"
            f"相似度：{item['similarity']:.4f}\n"
            f"{item['document']}"
        )

        context_blocks.append(context_block)

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
你是一個繁體中文醫療問答輔助系統。
請根據「參考資料」回答使用者問題。

回答規則：
1. 只能根據參考資料回答，不要自己編造。
2. 如果資料不足，請說「根據目前資料無法完全判斷」。
3. 回答要簡潔、清楚、保守。
4. 請提醒使用者：此回答不能取代醫師診斷。
5. 如果需要，請建議使用者就醫或掛適合科別。

使用者問題：
{question}

參考資料：
{context_text}

請用繁體中文回答：
""".strip()

    return prompt


def generate_answer(
    question: str,
    retrieved_docs: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int,
) -> str:
    """
    使用 Qwen Instruct 產生回答。
    """
    prompt = build_rag_prompt(question, retrieved_docs)

    # Qwen Instruct 適合使用 chat template
    messages = [
        {
            "role": "system",
            "content": "你是謹慎、保守的繁體中文醫療問答輔助系統。",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取模型新生成的部分
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


def run_chat(
    question: str,
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    llm_model_name: str,
    top_k: int,
    batch_size: int,
    device: str,
    max_new_tokens: int,
    load_in_4bit: bool,
):
    """
    單題 RAG 問答流程。
    """
    if question.strip() == "":
        raise ValueError("請使用 --question 輸入問題")

    print("\n========== RAG 單題問答 ==========")

    # 載入 embedding 模型
    embedding_model = load_embedding_model(embedding_model_name, device)

    # 查詢 Top-K 相似問答
    retrieved_docs = search_top_k(
        question=question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    # 顯示 Top-K 來源
    print("\n========== Top-K 相似問答 ==========")
    for i, item in enumerate(retrieved_docs, start=1):
        meta = item["metadata"]
        print(f"\n[{i}] 問題編號：{meta.get('question_id', '')}")
        print(f"標題：{meta.get('question_title', '')}")
        print(f"相似度：{item['similarity']:.4f}")

    # 載入 Qwen
    tokenizer, model = load_qwen_model(llm_model_name, load_in_4bit)

    # 生成回答
    answer = generate_answer(
        question=question,
        retrieved_docs=retrieved_docs,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )

    print("\n========== RAG 生成回答 ==========")
    print(answer)


# ============================================================
# 5. Retrieval 評估：Recall@K 與 MRR
# ============================================================

def calculate_recall_mrr(
    ranked_ids: List[str],
    correct_id: str,
    k_values: List[int],
) -> Tuple[Dict[int, float], float]:
    """
    計算一題的 Recall@K 與 MRR。

    Recall@K：
    正確答案有出現在前 K 筆，就是 1。
    沒有出現，就是 0。

    MRR：
    正確答案排名越前面，分數越高。
    第 1 名是 1/1。
    第 2 名是 1/2。
    第 5 名是 1/5。
    """
    recall_scores = {}

    for k in k_values:
        top_k_ids = ranked_ids[:k]
        recall_scores[k] = 1.0 if correct_id in top_k_ids else 0.0

    if correct_id in ranked_ids:
        rank = ranked_ids.index(correct_id) + 1
        mrr = 1.0 / rank
    else:
        mrr = 0.0

    return recall_scores, mrr


def evaluate_retrieval(
    csv_path: str,
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    eval_sample_size: int,
    top_k: int,
    batch_size: int,
    device: str,
    output_csv: str,
):
    """
    評估檢索效果。
    這裡會測試：
    系統能不能用原問題找回原本那筆醫療問答。
    """
    print("\n========== 檢索評估：Recall@K / MRR ==========")

    df = load_medical_csv(csv_path)

    # 抽樣，避免評估太久
    if eval_sample_size > 0 and eval_sample_size < len(df):
        df_eval = df.sample(n=eval_sample_size, random_state=42).reset_index(drop=True)
    else:
        df_eval = df.copy()

    print(f"[評估資料] {len(df_eval)} 筆")

    embedding_model = load_embedding_model(embedding_model_name, device)
    _, collection = get_collection(persist_dir, collection_name)

    k_values = [1, 3, 5, 10]
    top_k_for_eval = max(top_k, 10)

    total_recall = {k: 0.0 for k in k_values}
    total_mrr = 0.0
    records = []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="檢索評估"):
        query = build_query(row)
        correct_id = row["doc_id"]

        query_embedding = embedding_model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k_for_eval,
            include=["metadatas", "distances"],
        )

        ranked_ids = [meta["doc_id"] for meta in results["metadatas"][0]]

        recall_scores, mrr = calculate_recall_mrr(
            ranked_ids=ranked_ids,
            correct_id=correct_id,
            k_values=k_values,
        )

        for k in k_values:
            total_recall[k] += recall_scores[k]

        total_mrr += mrr

        records.append(
            {
                "doc_id": correct_id,
                "question_id": row["問題編號"],
                "question_title": row["question_title"],
                "ranked_ids": " | ".join(ranked_ids),
                "mrr": mrr,
                **{f"recall@{k}": recall_scores[k] for k in k_values},
            }
        )

    n = len(df_eval)

    print("\n========== 檢索評估結果 ==========")

    for k in k_values:
        score = total_recall[k] / n
        print(f"Recall@{k}: {score:.4f}")

    mrr_score = total_mrr / n
    print(f"MRR: {mrr_score:.4f}")

    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[輸出] 詳細結果已存成：{output_csv}")


# ============================================================
# 6. Generation 評估：ROUGE-L
# ============================================================

def lcs_length(a: List[str], b: List[str]) -> int:
    """
    計算 LCS，也就是最長共同子序列。
    ROUGE-L 就是用 LCS 概念計算。
    """
    m = len(a)
    n = len(b)

    # dp[j] 用來記錄目前比對到的位置
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0

        for j in range(1, n + 1):
            temp = dp[j]

            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])

            prev = temp

    return dp[n]


def rouge_l_char_level(prediction: str, reference: str) -> Dict[str, float]:
    """
    使用中文字元層級計算 ROUGE-L。
    中文不像英文有空白分詞，所以用「字」當單位比較簡單。
    """
    pred_chars = list(clean_text(prediction))
    ref_chars = list(clean_text(reference))

    if len(pred_chars) == 0 or len(ref_chars) == 0:
        return {
            "rouge_l_precision": 0.0,
            "rouge_l_recall": 0.0,
            "rouge_l_f1": 0.0,
        }

    lcs = lcs_length(pred_chars, ref_chars)

    precision = lcs / len(pred_chars)
    recall = lcs / len(ref_chars)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_f1": f1,
    }


def evaluate_generation(
    csv_path: str,
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    llm_model_name: str,
    eval_sample_size: int,
    top_k: int,
    batch_size: int,
    device: str,
    max_new_tokens: int,
    output_csv: str,
    load_in_4bit: bool,
):
    """
    評估生成效果。
    使用 ROUGE-L 比較：
    RAG 生成回答 vs 原始醫師回答。
    """
    print("\n========== 生成評估：ROUGE-L ==========")

    df = load_medical_csv(csv_path)

    # 生成評估很慢，所以通常抽 20 到 50 筆即可
    if eval_sample_size > 0 and eval_sample_size < len(df):
        df_eval = df.sample(n=eval_sample_size, random_state=42).reset_index(drop=True)
    else:
        df_eval = df.copy()

    print(f"[評估資料] {len(df_eval)} 筆")
    print("[提醒] 生成評估會載入 Qwen，可能需要較久時間")

    embedding_model = load_embedding_model(embedding_model_name, device)
    tokenizer, model = load_qwen_model(llm_model_name, load_in_4bit)

    records = []
    total_f1 = 0.0

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="生成評估"):
        question = build_query(row)
        reference_answer = row["answer_text"]

        # 先找 Top-K 參考資料
        retrieved_docs = search_top_k(
            question=question,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            top_k=top_k,
        )

        # 再用 Qwen 生成回答
        generated_answer = generate_answer(
            question=question,
            retrieved_docs=retrieved_docs,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
        )

        # 計算 ROUGE-L
        rouge_scores = rouge_l_char_level(
            prediction=generated_answer,
            reference=reference_answer,
        )

        total_f1 += rouge_scores["rouge_l_f1"]

        records.append(
            {
                "doc_id": row["doc_id"],
                "question_id": row["問題編號"],
                "question_title": row["question_title"],
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                **rouge_scores,
            }
        )

    avg_f1 = total_f1 / len(df_eval)

    print("\n========== 生成評估結果 ==========")
    print(f"Average ROUGE-L F1: {avg_f1:.4f}")

    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[輸出] 詳細結果已存成：{output_csv}")


# ============================================================
# 7. 命令列參數
# ============================================================

def parse_args():
    """
    設定可以在終端機輸入的參數。
    """
    parser = argparse.ArgumentParser(
        description="RAG 醫療問答：BAAI/bge-m3 + ChromaDB + Qwen Instruct",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        choices=["build", "chat", "eval_retrieval", "eval_generation", "all"],
        help="執行模式",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line(2).csv",
        help="CSV 檔案路徑",
    )

    parser.add_argument(
        "--persist_dir",
        type=str,
        default="chroma_db",
        help="ChromaDB 儲存資料夾",
    )

    parser.add_argument(
        "--collection_name",
        type=str,
        default="taiwan_ehospital_medical_qa",
        help="ChromaDB collection 名稱",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding 模型名稱",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Qwen Instruct 模型名稱",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="mode=chat 時輸入的問題",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="相似度搜尋取前 K 筆",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批次大小",
    )

    parser.add_argument(
        "--eval_sample_size",
        type=int,
        default=100,
        help="評估抽樣筆數，0 代表全部資料",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="LLM 最多生成 token 數",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="embedding 使用裝置，通常是 cuda 或 cpu",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="是否刪除舊 ChromaDB 並重新建立",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="是否用 4-bit 載入 Qwen，較省 GPU 記憶體",
    )

    parser.add_argument(
        "--retrieval_output_csv",
        type=str,
        default="retrieval_eval_result.csv",
        help="檢索評估輸出檔",
    )

    parser.add_argument(
        "--generation_output_csv",
        type=str,
        default="generation_eval_result.csv",
        help="生成評估輸出檔",
    )

    return parser.parse_args()


# ============================================================
# 8. 主程式
# ============================================================

def main():
    """
    主程式入口。
    會依照 --mode 執行不同功能。
    """
    args = parse_args()

    # 固定隨機種子，讓抽樣結果比較穩定
    random.seed(42)
    torch.manual_seed(42)

    print("\n========== 系統設定 ==========")
    print(f"mode: {args.mode}")
    print(f"csv: {args.csv}")
    print(f"persist_dir: {args.persist_dir}")
    print(f"collection_name: {args.collection_name}")
    print(f"embedding_model: {args.embedding_model}")
    print(f"llm_model: {args.llm_model}")
    print(f"device: {args.device}")
    print(f"top_k: {args.top_k}")

    if args.mode == "build":
        build_chroma_index(
            csv_path=args.csv,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            rebuild=args.rebuild,
        )

    elif args.mode == "chat":
        run_chat(
            question=args.question,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            llm_model_name=args.llm_model,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
        )

    elif args.mode == "eval_retrieval":
        evaluate_retrieval(
            csv_path=args.csv,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            eval_sample_size=args.eval_sample_size,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            output_csv=args.retrieval_output_csv,
        )

    elif args.mode == "eval_generation":
        evaluate_generation(
            csv_path=args.csv,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            llm_model_name=args.llm_model,
            eval_sample_size=args.eval_sample_size,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            output_csv=args.generation_output_csv,
            load_in_4bit=args.load_in_4bit,
        )

    elif args.mode == "all":
        # 先建立向量資料庫
        build_chroma_index(
            csv_path=args.csv,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            rebuild=args.rebuild,
        )

        # 再評估檢索
        evaluate_retrieval(
            csv_path=args.csv,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            eval_sample_size=args.eval_sample_size,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            output_csv=args.retrieval_output_csv,
        )

        # 最後評估生成
        # 為避免太久，all 模式最多先跑 20 筆生成評估
        generation_sample_size = args.eval_sample_size
        if generation_sample_size == 0 or generation_sample_size > 20:
            generation_sample_size = 20

        evaluate_generation(
            csv_path=args.csv,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            llm_model_name=args.llm_model,
            eval_sample_size=generation_sample_size,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            output_csv=args.generation_output_csv,
            load_in_4bit=args.load_in_4bit,
        )


if __name__ == "__main__":
    main()
