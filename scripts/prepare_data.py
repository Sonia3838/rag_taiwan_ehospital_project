import argparse
import re
from pathlib import Path

import pandas as pd


def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\uf0a4", "")
    x = re.sub(r"\r\n|\r", "\n", x)
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 CSV 路徑")
    parser.add_argument("--output", default="data/merged_medical_qa.csv", help="輸出清理後 CSV 路徑")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = ["問題編號", "question_title", "question_text", "answer_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少欄位：{missing}")

    for col in ["question_title", "question_text", "answer_text"]:
        df[col] = df[col].apply(clean_text)

    df = df.dropna(subset=["question_text", "answer_text"])
    df = df[(df["question_text"].str.len() > 0) & (df["answer_text"].str.len() > 0)]
    df = df.drop_duplicates(subset=["問題編號"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"清理完成：{args.output}")
    print(f"資料筆數：{len(df)}")
    print(f"欄位：{list(df.columns)}")


if __name__ == "__main__":
    main()
