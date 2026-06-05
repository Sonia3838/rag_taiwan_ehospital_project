import argparse
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def make_document(row: pd.Series) -> str:
    title = str(row.get("question_title", "")).strip()
    question = str(row.get("question_text", "")).strip()
    answer = str(row.get("answer_text", "")).strip()
    return f"標題：{title}\n\n病患提問：\n{question}\n\n醫師回覆：\n{answer}"


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/merged_medical_qa.csv")
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--collection", default="taiwan_ehospital_qa")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.reset_index(drop=True)

    docs = [make_document(row) for _, row in df.iterrows()]
    ids = [str(row["問題編號"]) for _, row in df.iterrows()]
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "問題編號": str(row.get("問題編號", "")),
            "page": str(row.get("page", "")),
            "question_title": str(row.get("question_title", "")),
        })

    print("載入 embedding model：", args.embedding_model)
    model = SentenceTransformer(args.embedding_model, device="cuda")

    client = chromadb.PersistentClient(path=args.persist_dir)
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    collection = client.create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    for doc_batch, id_batch, meta_batch in tqdm(
        zip(batched(docs, args.batch_size), batched(ids, args.batch_size), batched(metadatas, args.batch_size)),
        total=(len(docs) + args.batch_size - 1) // args.batch_size,
        desc="建立向量資料庫"
    ):
        embeddings = model.encode(
            doc_batch,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=id_batch,
            documents=doc_batch,
            metadatas=meta_batch,
            embeddings=embeddings,
        )

    print(f"完成：共建立 {collection.count()} 筆向量資料")
    print(f"ChromaDB 位置：{Path(args.persist_dir).resolve()}")


if __name__ == "__main__":
    main()
