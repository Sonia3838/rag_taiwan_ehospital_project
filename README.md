# 台灣 e 院醫療問答 RAG 期末專題

## 1. 專題題目
使用 RAG 建立台灣 e 院醫療問答輔助系統。

## 2. 系統流程
1. 載入台灣 e 院 Q&A CSV。
2. 清理文字，保留 question_title、question_text、answer_text。
3. 使用 BAAI/bge-m3 建立 embedding。
4. 將向量存入 ChromaDB。
5. 使用相似度搜尋取得 Top-K 問答。
6. 使用 Qwen Instruct LLM 產生回答。
7. 使用 Recall@K、MRR、ROUGE-L 評估。

## 3. 安裝
```bash
conda create -n rag_medical python=3.10 -y
conda activate rag_medical
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 4. 資料準備
把原始 CSV 放到 data/ 底下，然後執行：
```bash
python scripts/prepare_data.py \
  --input data/merged_medical_qa_0416_UTF8_整理結果_1250_removed_answer_first_line.csv \
  --output data/merged_medical_qa.csv
```

## 5. 建立向量資料庫
```bash
python scripts/build_index.py \
  --csv data/merged_medical_qa.csv \
  --persist_dir chroma_db
```

## 6. 問答測試
```bash
python scripts/chat_cli.py \
  --question "排卵期前出血需要看醫生嗎？" \
  --persist_dir chroma_db
```

## 7. 評估檢索
```bash
python scripts/evaluate_retrieval.py \
  --csv data/merged_medical_qa.csv \
  --persist_dir chroma_db \
  --sample_size 200
```

## 8. 評估生成
```bash
python scripts/evaluate_generation_rouge.py \
  --csv data/merged_medical_qa.csv \
  --persist_dir chroma_db \
  --sample_size 30
```

## 9. 啟動網頁介面
```bash
python scripts/app_gradio.py --persist_dir chroma_db --server_port 7860
```
# rag_taiwan_ehospital_project
