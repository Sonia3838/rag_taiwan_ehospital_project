#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
檔名：rag_terminal_chat.py

用途：
這支程式可以讓你在 terminal / 虛擬機中連續輸入醫療問題，
系統會從 ChromaDB 找出最相似的台灣 e 院問答資料，
再使用 Qwen Instruct 模型產生 RAG 回答。

適合情境：
1. 你已經用原本的 rag_medical_bgem3_chromadb_qwen.py 建好 chroma_db。
2. 想要像聊天機器人一樣，在 terminal 直接輸入問題。
3. 每次回答都會顯示 Top-K 參考來源與 RAG 生成結果。

安裝套件：
pip install torch sentence-transformers chromadb transformers accelerate

如果 GPU 記憶體不足，想用 4-bit 載入 Qwen：
pip install bitsandbytes

常用執行方式：
python rag_terminal_chat.py \
  --persist_dir chroma_db \
  --collection_name taiwan_ehospital_medical_qa \
  --top_k 5 \
  --load_in_4bit

如果你還沒有建立 ChromaDB，也可以用這支程式先建立：
python rag_terminal_chat.py \
  --csv "merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line.csv" \
  --persist_dir chroma_db \
  --build_if_empty \
  --load_in_4bit

離開方式：
在問題輸入處輸入 exit、quit、q 或 直接按 Ctrl+C。
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

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


def clean_text(text) -> str:
    """清理文字，避免換行、Tab、特殊符號影響檢索。"""
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
    """
    如果 ChromaDB 是空的，就用 CSV 建立向量資料庫。
    這樣使用者不用另外跑 build 模式。
    """
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


def build_rag_prompt(question: str, retrieved_docs: List[Dict]) -> str:
    """建立 RAG prompt。"""
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
5. 如果症狀明顯或持續，請建議就醫。
6. 回答最後請加上一句「建議科別：XXX科」，若無法判斷則寫「建議科別：請先至家醫科或一般內科評估」。

使用者問題：
{question}

參考資料：
{context_text}

請用繁體中文回答：
""".strip()


def generate_answer(
    question: str,
    retrieved_docs: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int,
) -> str:
    """使用 Qwen 產生回答。"""
    prompt = build_rag_prompt(question, retrieved_docs)

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
    return answer.strip()


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


def interactive_chat(args):
    """terminal 互動式 RAG 問答。"""
    print("\n========== RAG Terminal 醫療問答系統 ==========")
    print("輸入 exit、quit、q 可離開。")

    embedding_model = load_embedding_model(args.embedding_model, args.device)

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
        else:
            raise ValueError(
                "目前 ChromaDB 是空的。請先用原本程式 build，"
                "或執行本程式時加入 --csv 與 --build_if_empty。"
            )
    else:
        print(f"\n[ChromaDB] 已載入 {collection.count()} 筆向量資料")

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
        "--csv",
        type=str,
        default="",
        help="醫療問答 CSV 路徑。只有在 --build_if_empty 時需要。",
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

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n========== 系統設定 ==========")
    print(f"csv: {args.csv if args.csv else '未指定'}")
    print(f"persist_dir: {args.persist_dir}")
    print(f"collection_name: {args.collection_name}")
    print(f"embedding_model: {args.embedding_model}")
    print(f"llm_model: {args.llm_model}")
    print(f"device: {args.device}")
    print(f"top_k: {args.top_k}")

    interactive_chat(args)


if __name__ == "__main__":
    main()
