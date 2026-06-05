import argparse

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

from chat_cli import build_context, generate_answer, load_llm
from retriever import MedicalQARetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/merged_medical_qa.csv")
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--collection", default="taiwan_ehospital_qa")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", default="generation_eval_result.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sample(args.sample_size, random_state=42).reset_index(drop=True)
    retriever = MedicalQARetriever(args.persist_dir, args.collection, args.embedding_model)
    tokenizer, model = load_llm(args.llm_model)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="評估生成"):
        question = f"{row['question_title']}\n{row['question_text']}"
        reference = str(row["answer_text"])
        hits = retriever.search(question, top_k=args.top_k)
        context = build_context(hits)
        prediction = generate_answer(tokenizer, model, question, context)
        rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
        rows.append({
            "問題編號": row["問題編號"],
            "question_title": row["question_title"],
            "reference_answer": reference,
            "rag_answer": prediction,
            "rougeL_f1": rouge_l,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"輸出：{args.output}")
    print(f"平均 ROUGE-L F1：{out['rougeL_f1'].mean():.4f}")


if __name__ == "__main__":
    main()
