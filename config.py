from pathlib import Path

# ===== 可依環境調整 =====
DATA_PATH = Path("data/merged_medical_qa_0604_utf8_整理結果_removed_answer_first_line.csv")
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "taiwan_ehospital_qa"

# 
EMBEDDING_MODEL = "BAAI/bge-m3"

# RTX6000 / RTX4500 Ada 通常可跑 4-bit 7B。
# 記憶體不足時可改成："Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TOP_K = 5
MAX_NEW_TOKENS = 512
