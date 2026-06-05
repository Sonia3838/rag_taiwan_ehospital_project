import argparse
import random

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def recall_at_k(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks):
    vals = [1 / r for r in ranks if r is not None]
    return sum(vals) / len(ranks) if ranks else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/merged_medical_qa.csv")
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--collection", default="taiwan_ehospital_qa")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).reset_index(drop=True)
    random.seed(args.seed)
    if args.sample_size > 0 and args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=args.seed).reset_index(drop=True)

    embedder = SentenceTransformer(args.embedding_model, device="cuda")
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(args.collection)

    ranks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="評估檢索"):
        query = f"{row['question_title']}\n{row['question_text']}"
        target_id = str(row["問題編號"])
        query_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        result = collection.query(
            query_embeddings=[query_emb],
            n_results=args.top_k,
            include=["metadatas"],
        )
        ids = [str(m.get("問題編號")) for m in result["metadatas"][0]]
        rank = ids.index(target_id) + 1 if target_id in ids else None
        ranks.append(rank)

    print("\n===== Retrieval Evaluation =====")
    print(f"樣本數：{len(df)}")
    print(f"Recall@1：{recall_at_k(ranks, 1):.4f}")
    print(f"Recall@3：{recall_at_k(ranks, 3):.4f}")
    print(f"Recall@5：{recall_at_k(ranks, 5):.4f}")
    print(f"Recall@10：{recall_at_k(ranks, 10):.4f}")
    print(f"MRR@{args.top_k}：{mrr(ranks):.4f}")


if __name__ == "__main__":
    main()
