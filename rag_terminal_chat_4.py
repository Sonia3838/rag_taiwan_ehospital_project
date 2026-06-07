#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
檔名：rag_terminla_chat_4.py

用途：
這支程式可以讓你在 terminal / 虛擬機中連續輸入醫療問題，
系統會從 ChromaDB 找出最相似的台灣 e 院問答資料，
再使用 Qwen Instruct 模型產生 RAG 回答。

本版重點：
1. 每次醫療回答都會提醒使用者：此回答不能取代醫師診斷。
2. 新增 eval_generation 模式。
3. eval_generation 模式使用較短 prompt：
   「請根據參考資料回答，盡量使用與參考資料相近的用詞。」

安裝套件：
pip install pandas tqdm torch sentence-transformers chromadb transformers accelerate

如果 GPU 記憶體不足，想用 4-bit 載入 Qwen：
pip install bitsandbytes

常用執行方式：

一、互動式醫療問答
python rag_terminla_chat_4.py \
  --mode chat \
  --persist_dir chroma_db \
  --collection_name taiwan_ehospital_medical_qa \
  --top_k 5 \
  --load_in_4bit

二、如果 ChromaDB 是空的，先用 CSV 自動建立
python rag_terminla_chat_4.py \
  --mode chat \
  --csv "merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line.csv" \
  --persist_dir chroma_db \
  --collection_name taiwan_ehospital_medical_qa \
  --build_if_empty \
  --load_in_4bit

三、真實使用者問題檢索評估
python rag_terminla_chat_4.py \
  --mode eval_user_retrieval \
  --eval_query_csv "./data/eval_user_questions.csv" \
  --persist_dir chroma_db \
  --collection_name taiwan_ehospital_medical_qa \
  --top_k 5

四、生成回答評估
python rag_terminla_chat_4.py \
  --mode eval_generation \
  --csv "merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line.csv" \
  --persist_dir chroma_db \
  --collection_name taiwan_ehospital_medical_qa \
  --eval_sample_size 20 \
  --top_k 5 \
  --load_in_4bit

離開方式：
在問題輸入處輸入 exit、quit、q 或直接按 Ctrl+C。
"""

import argparse
import os
import random
import re
from typing import Dict, List, Optional

import chromadb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_COLUMNS = [
    "page",
    "問題編號",
    "question_title",
    "question_text",
    "answer_text",
]

MEDICAL_DISCLAIMER = "提醒：此回答不能取代醫師診斷。"


def clean_text(text) -> str:
    """清理文字，避免換行、Tab、特殊符號影響檢索與生成。"""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\uf0a4", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_medical_csv(csv_path: str) -> pd.DataFrame:
    """讀取醫療問答 CSV，並檢查必要欄位。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 檔案：{csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV 缺少欄位：{missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["question_title"] = df["question_title"].apply(clean_text)
    df["question_text"] = df["question_text"].apply(clean_text)
    df["answer_text"] = df["answer_text"].apply(clean_text)

    df = df[(df["question_text"] != "") & (df["answer_text"] != "")].copy()
    df = df.reset_index(drop=True)
    df["doc_id"] = [f"qa_{i:06d}" for i in range(len(df))]

    return df


def build_document(row: pd.Series) -> str:
    """把一筆資料整理成要放進 ChromaDB 的文件。"""
    return (
        f"問題標題：{row['question_title']}\n"
        f"病患提問：{row['question_text']}\n"
        f"醫師回答：{row['answer_text']}"
    )


def build_query(row: pd.Series) -> str:
    """建立 eval_generation 時使用的查詢文字。"""
    title = clean_text(row.get("question_title", ""))
    question = clean_text(row.get("question_text", ""))

    if title:
        return f"問題標題：{title}\n病患提問：{question}"

    return question


def get_collection(persist_dir: str, collection_name: str):
    """取得 ChromaDB collection。"""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def load_embedding_model(model_name: str, device: str):
    """載入 embedding 模型。"""
    print("\n[Embedding] 載入模型")
    print(f"模型名稱：{model_name}")
    print(f"使用裝置：{device}")
    return SentenceTransformer(model_name, device=device)


def build_chroma_if_empty(
    csv_path: str,
    persist_dir: str,
    collection_name: str,
    embedding_model,
    batch_size: int,
):
    """如果 ChromaDB 是空的，就用 CSV 建立向量資料庫。"""
    if not csv_path:
        raise ValueError("ChromaDB 目前是空的，請提供 --csv 或先用原本程式建立 chroma_db。")

    df = load_medical_csv(csv_path)
    _, collection = get_collection(persist_dir, collection_name)

    if collection.count() > 0:
        print(f"[ChromaDB] 已有 {collection.count()} 筆資料，不需要重新建立。")
        return

    print("\n========== 建立 ChromaDB 向量資料庫 ==========")
    print(f"[資料] 有效問答筆數：{len(df)}")

    documents = [build_document(row) for _, row in df.iterrows()]
    ids = df["doc_id"].tolist()

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

    print("[Embedding] 開始建立向量")
    embeddings = embedding_model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

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


def prepare_collection(args, embedding_model):
    """確認 ChromaDB 是否可用；若為空且允許，就自動建立。"""
    _, collection = get_collection(args.persist_dir, args.collection_name)

    if collection.count() == 0:
        if args.build_if_empty:
            build_chroma_if_empty(
                csv_path=args.csv,
                persist_dir=args.persist_dir,
                collection_name=args.collection_name,
                embedding_model=embedding_model,
                batch_size=args.batch_size,
            )
            _, collection = get_collection(args.persist_dir, args.collection_name)
        else:
            raise ValueError(
                "目前 ChromaDB 是空的。請先建立向量資料庫，"
                "或執行本程式時加入 --csv 與 --build_if_empty。"
            )

    print(f"\n[ChromaDB] 已載入 {collection.count()} 筆向量資料")
    return collection


def search_top_k(
    question: str,
    persist_dir: str,
    collection_name: str,
    embedding_model,
    top_k: int,
) -> List[Dict]:
    """用使用者問題查詢 Top-K 相似問答。"""
    _, collection = get_collection(persist_dir, collection_name)

    if collection.count() == 0:
        raise ValueError("ChromaDB collection 是空的，請先建立向量資料庫。")

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
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


def load_qwen_model(llm_model_name: str, load_in_4bit: bool):
    """載入 Qwen Instruct 模型。"""
    print("\n[LLM] 載入 Qwen Instruct 模型")
    print(f"模型名稱：{llm_model_name}")
    print(f"4-bit 量化：{load_in_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        trust_remote_code=True,
    )

    if load_in_4bit:
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


def ensure_medical_disclaimer(answer: str) -> str:
    """確保回答一定包含醫療提醒。"""
    answer = clean_text(answer)

    if "不能取代醫師診斷" not in answer:
        answer = f"{answer}\n\n{MEDICAL_DISCLAIMER}".strip()

    return answer


def build_rag_prompt(question: str, retrieved_docs: List[Dict]) -> str:
    """建立互動式 chat 使用的 RAG prompt。"""
    context_blocks = []

    for i, item in enumerate(retrieved_docs, start=1):
        meta = item["metadata"]
        context_blocks.append(
            f"【參考資料 {i}】\n"
            f"問題編號：{meta.get('question_id', '')}\n"
            f"相似度：{item['similarity']:.4f}\n"
            f"{item['document']}"
        )

    context_text = "\n\n".join(context_blocks)

    return f"""
你是一個繁體中文醫療問答輔助系統。
請根據「參考資料」回答使用者問題。

回答規則：
1. 只能根據參考資料回答，不要自行編造不存在的資訊。
2. 如果資料不足，請說「根據目前資料無法完全判斷」。
3. 回答要簡潔、清楚、保守。
4. 請提醒使用者：此回答不能取代醫師診斷。
5. 如果症狀明顯、嚴重或持續，請建議就醫。
6. 回答最後請加上一句「建議科別：XXX科」，若無法判斷則寫「建議科別：請先至家醫科或一般內科評估」。

使用者問題：
{question}

參考資料：
{context_text}

請用繁體中文回答：
""".strip()


def build_eval_generation_prompt(question: str, retrieved_docs: List[Dict]) -> str:
    """
    建立 eval_generation 使用的較短 prompt。

    依照需求，這裡刻意使用短指令：
    「請根據參考資料回答，盡量使用與參考資料相近的用詞。」
    """
    context_blocks = []

    for i, item in enumerate(retrieved_docs, start=1):
        context_blocks.append(
            f"【參考資料 {i}】\n{item['document']}"
        )

    context_text = "\n\n".join(context_blocks)

    return f"""
請根據參考資料回答，盡量使用與參考資料相近的用詞。
請提醒使用者：此回答不能取代醫師診斷。

問題：
{question}

參考資料：
{context_text}

回答：
""".strip()


def generate_answer_by_prompt(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int,
    system_message: str = "你是謹慎、保守的繁體中文醫療問答輔助系統。",
) -> str:
    """使用指定 prompt 產生回答。"""
    messages = [
        {
            "role": "system",
            "content": system_message,
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

    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return ensure_medical_disclaimer(answer)


def generate_answer(
    question: str,
    retrieved_docs: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int,
) -> str:
    """互動式 chat 使用的回答生成。"""
    prompt = build_rag_prompt(question, retrieved_docs)
    return generate_answer_by_prompt(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )


def print_sources(retrieved_docs: List[Dict], show_context: bool):
    """顯示檢索到的 Top-K 來源。"""
    print("\n========== Top-K 相似問答來源 ==========")

    for i, item in enumerate(retrieved_docs, start=1):
        meta = item["metadata"]
        print(f"\n[{i}] 問題編號：{meta.get('question_id', '')}")
        print(f"標題：{meta.get('question_title', '')}")
        print(f"相似度：{item['similarity']:.4f}")

        if show_context:
            print("--- 參考內容 ---")
            print(item["document"][:800])
            if len(item["document"]) > 800:
                print("...（內容過長已省略）")


def lcs_length(a: List[str], b: List[str]) -> int:
    """計算 LCS 長度，供 ROUGE-L 使用。"""
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)

    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current

    return previous[-1]


def rouge_l_score(prediction: str, reference: str) -> Dict[str, float]:
    """
    計算簡易版 ROUGE-L。

    中文文字沒有空白分詞時，這裡用「字元」作為 token，
    優點是不用額外安裝 jieba 或 rouge 套件。
    """
    pred_tokens = list(clean_text(prediction))
    ref_tokens = list(clean_text(reference))

    if not pred_tokens or not ref_tokens:
        return {
            "rouge_l_precision": 0.0,
            "rouge_l_recall": 0.0,
            "rouge_l_f1": 0.0,
        }

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_f1": f1,
    }


def evaluate_user_queries_retrieval(
    eval_query_csv: str,
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
    batch_size: int,
    device: str,
    output_csv: str,
):
    """
    使用真實使用者問題評估 Retrieval 效果。

    eval_query_csv 需要包含：
    user_question, gold_question_id
    """
    print("\n========== 真實使用者問題檢索評估：Recall@K / MRR ==========")

    if not os.path.exists(eval_query_csv):
        raise FileNotFoundError(f"找不到使用者問題評估檔：{eval_query_csv}")

    df_eval = pd.read_csv(eval_query_csv)

    required_cols = ["user_question", "gold_question_id"]
    missing_cols = [col for col in required_cols if col not in df_eval.columns]

    if missing_cols:
        raise ValueError(f"評估檔缺少欄位：{missing_cols}")

    df_eval["user_question"] = df_eval["user_question"].apply(clean_text)
    df_eval["gold_question_id"] = df_eval["gold_question_id"].astype(str)

    df_eval = df_eval[df_eval["user_question"] != ""].reset_index(drop=True)

    print(f"[評估資料] {len(df_eval)} 筆")

    embedding_model = load_embedding_model(embedding_model_name, device)

    _, collection = get_collection(persist_dir, collection_name)

    if collection.count() == 0:
        raise ValueError("ChromaDB collection 是空的，請先建立向量資料庫。")

    k_values = [1, 3, 5, 10]
    top_k_for_eval = max(top_k, 10)

    total_recall = {k: 0.0 for k in k_values}
    total_mrr = 0.0
    records = []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="真實使用者檢索評估"):
        user_question = row["user_question"]
        gold_question_id = str(row["gold_question_id"])

        query_embedding = embedding_model.encode(
            [user_question],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k_for_eval,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_question_ids = [
            str(meta.get("question_id", ""))
            for meta in results["metadatas"][0]
        ]

        retrieved_titles = [
            str(meta.get("question_title", ""))
            for meta in results["metadatas"][0]
        ]

        distances = results["distances"][0]
        similarities = [1.0 - float(d) for d in distances]

        recall_scores = {}

        for k in k_values:
            top_k_ids = retrieved_question_ids[:k]
            recall_scores[k] = 1.0 if gold_question_id in top_k_ids else 0.0
            total_recall[k] += recall_scores[k]

        if gold_question_id in retrieved_question_ids:
            rank = retrieved_question_ids.index(gold_question_id) + 1
            mrr = 1.0 / rank
        else:
            rank = 0
            mrr = 0.0

        total_mrr += mrr

        records.append(
            {
                "user_question": user_question,
                "gold_question_id": gold_question_id,
                "retrieved_question_ids": " | ".join(retrieved_question_ids),
                "retrieved_titles": " | ".join(retrieved_titles),
                "similarities": " | ".join([f"{s:.4f}" for s in similarities]),
                "rank": rank,
                "mrr": mrr,
                **{f"recall@{k}": recall_scores[k] for k in k_values},
            }
        )

    n = len(df_eval)

    print("\n========== 真實使用者問題檢索評估結果 ==========")

    for k in k_values:
        score = total_recall[k] / n
        print(f"Recall@{k}: {score:.4f}")

    mrr_score = total_mrr / n
    print(f"MRR: {mrr_score:.4f}")

    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[輸出] 詳細結果已存成：{output_csv}")


def evaluate_generation(args):
    """
    評估 RAG 生成回答。

    流程：
    1. 從 CSV 取出題目。
    2. 用題目查詢 Top-K 參考資料。
    3. 使用較短 prompt 讓 LLM 產生回答。
    4. 以原本 answer_text 當 reference，計算 ROUGE-L。
    5. 輸出 generation_eval_result.csv。
    """
    print("\n========== RAG 生成評估：ROUGE-L ==========")

    if not args.csv:
        raise ValueError("eval_generation 模式需要提供 --csv，才能取得 question_text 與 answer_text。")

    df = load_medical_csv(args.csv)

    if args.eval_sample_size > 0 and args.eval_sample_size < len(df):
        df_eval = df.sample(
            n=args.eval_sample_size,
            random_state=args.random_seed,
        ).reset_index(drop=True)
    else:
        df_eval = df.reset_index(drop=True)

    print(f"[評估資料] {len(df_eval)} 筆")

    embedding_model = load_embedding_model(args.embedding_model, args.device)
    prepare_collection(args, embedding_model)

    tokenizer, model = load_qwen_model(args.llm_model, args.load_in_4bit)

    records = []
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="生成評估"):
        query = build_query(row)

        retrieved_docs = search_top_k(
            question=query,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model=embedding_model,
            top_k=args.top_k,
        )

        prompt = build_eval_generation_prompt(query, retrieved_docs)

        generated_answer = generate_answer_by_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
        )

        reference_answer = clean_text(row["answer_text"])
        rouge = rouge_l_score(generated_answer, reference_answer)

        total_precision += rouge["rouge_l_precision"]
        total_recall += rouge["rouge_l_recall"]
        total_f1 += rouge["rouge_l_f1"]

        retrieved_question_ids = [
            str(item["metadata"].get("question_id", ""))
            for item in retrieved_docs
        ]

        records.append(
            {
                "question_id": str(row["問題編號"]),
                "question_title": row["question_title"],
                "question_text": row["question_text"],
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "retrieved_question_ids": " | ".join(retrieved_question_ids),
                "rouge_l_precision": rouge["rouge_l_precision"],
                "rouge_l_recall": rouge["rouge_l_recall"],
                "rouge_l_f1": rouge["rouge_l_f1"],
            }
        )

    n = len(df_eval)
    avg_precision = total_precision / n if n else 0.0
    avg_recall = total_recall / n if n else 0.0
    avg_f1 = total_f1 / n if n else 0.0

    print("\n========== 生成評估結果 ==========")
    print(f"ROUGE-L Precision: {avg_precision:.4f}")
    print(f"ROUGE-L Recall: {avg_recall:.4f}")
    print(f"ROUGE-L F1: {avg_f1:.4f}")

    pd.DataFrame(records).to_csv(
        args.generation_output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n[輸出] 詳細結果已存成：{args.generation_output_csv}")


def interactive_chat(args):
    """terminal 互動式 RAG 問答。"""
    print("\n========== RAG Terminal 醫療問答系統 ==========")
    print("輸入 exit、quit、q 可離開。")
    print(MEDICAL_DISCLAIMER)

    embedding_model = load_embedding_model(args.embedding_model, args.device)
    prepare_collection(args, embedding_model)

    tokenizer, model = load_qwen_model(args.llm_model, args.load_in_4bit)

    while True:
        try:
            question = input("\n請輸入醫療問題 > ").strip()
        except KeyboardInterrupt:
            print("\n已離開 RAG 問答系統。")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("已離開 RAG 問答系統。")
            break

        if not question:
            print("請輸入問題，不要留空。")
            continue

        try:
            retrieved_docs = search_top_k(
                question=question,
                persist_dir=args.persist_dir,
                collection_name=args.collection_name,
                embedding_model=embedding_model,
                top_k=args.top_k,
            )

            print_sources(retrieved_docs, args.show_context)

            answer = generate_answer(
                question=question,
                retrieved_docs=retrieved_docs,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
            )

            print("\n========== RAG 生成回答 ==========")
            print(answer)

        except Exception as e:
            print(f"\n[錯誤] 這次問題處理失敗：{e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Terminal 互動式 RAG 醫療問答：BAAI/bge-m3 + ChromaDB + Qwen Instruct",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        choices=["chat", "eval_user_retrieval", "eval_generation"],
        help="執行模式：chat、eval_user_retrieval、eval_generation。",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="醫療問答 CSV 路徑。build_if_empty 與 eval_generation 會用到。",
    )

    parser.add_argument(
        "--persist_dir",
        type=str,
        default="chroma_db",
        help="ChromaDB 儲存資料夾，例如 chroma_db。",
    )

    parser.add_argument(
        "--collection_name",
        type=str,
        default="taiwan_ehospital_medical_qa",
        help="ChromaDB collection 名稱。需與建庫時相同。",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding 模型名稱。",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="LLM 模型名稱。",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="每次檢索最相似的前 K 筆資料。",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="建立 embedding 時的 batch size。",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="LLM 最多生成 token 數。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Embedding 使用裝置，通常是 cuda 或 cpu。",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="使用 4-bit 載入 Qwen，較省 GPU 記憶體。",
    )

    parser.add_argument(
        "--build_if_empty",
        action="store_true",
        help="如果 ChromaDB 是空的，就用 --csv 自動建立向量資料庫。",
    )

    parser.add_argument(
        "--show_context",
        action="store_true",
        help="顯示 Top-K 參考資料的部分原文內容。",
    )

    parser.add_argument(
        "--eval_query_csv",
        type=str,
        default="./data/eval_user_questions.csv",
        help="真實使用者問題評估檔，需包含 user_question, gold_question_id。",
    )

    parser.add_argument(
        "--user_retrieval_output_csv",
        type=str,
        default="user_retrieval_eval_result.csv",
        help="真實使用者問題檢索評估輸出檔。",
    )

    parser.add_argument(
        "--eval_sample_size",
        type=int,
        default=20,
        help="eval_generation 抽樣評估筆數；設為 0 或大於資料量代表全部評估。",
    )

    parser.add_argument(
        "--generation_output_csv",
        type=str,
        default="generation_eval_result.csv",
        help="eval_generation 輸出 CSV 檔名。",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="eval_generation 抽樣用 random seed。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.random_seed)

    print("\n========== 系統設定 ==========")
    print(f"mode: {args.mode}")
    print(f"csv: {args.csv if args.csv else '未指定'}")
    print(f"persist_dir: {args.persist_dir}")
    print(f"collection_name: {args.collection_name}")
    print(f"embedding_model: {args.embedding_model}")
    print(f"llm_model: {args.llm_model}")
    print(f"device: {args.device}")
    print(f"top_k: {args.top_k}")
    print(MEDICAL_DISCLAIMER)

    if args.mode == "chat":
        interactive_chat(args)

    elif args.mode == "eval_user_retrieval":
        evaluate_user_queries_retrieval(
            eval_query_csv=args.eval_query_csv,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            output_csv=args.user_retrieval_output_csv,
        )

    elif args.mode == "eval_generation":
        evaluate_generation(args)

    else:
        raise ValueError(f"不支援的 mode：{args.mode}")


if __name__ == "__main__":
    main()
