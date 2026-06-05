import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from retriever import MedicalQARetriever


SYSTEM_PROMPT = """
你是一個醫療問答 RAG 助手。
你只能根據提供的相似問答資料回答。
回答目的為學術研究與就醫科別建議，不可取代醫師診斷。
若資料不足，請明確說「根據目前檢索資料不足以判斷」。
回答請包含：
1. 可能情況
2. 建議處理方式
3. 建議就醫科別
4. 參考到的問題編號
""".strip()


def build_context(hits):
    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        blocks.append(
            f"[參考資料 {i}]\n"
            f"問題編號：{meta.get('問題編號')}\n"
            f"標題：{meta.get('question_title')}\n"
            f"相似度：{hit['score']:.4f}\n"
            f"內容：\n{hit['document']}"
        )
    return "\n\n".join(blocks)


def load_llm(model_name):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return tokenizer, model


def generate_answer(tokenizer, model, question, context, max_new_tokens=512):
    user_prompt = f"""
使用者問題：
{question}

以下是向量資料庫檢索到的相似台灣 e 院問答：
{context}

請根據上述資料，用繁體中文回答。
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--collection", default="taiwan_ehospital_qa")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = MedicalQARetriever(args.persist_dir, args.collection, args.embedding_model)
    hits = retriever.search(args.question, top_k=args.top_k)
    context = build_context(hits)

    print("\n===== 檢索結果 =====")
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        print(f"{i}. 問題編號={meta.get('問題編號')}｜標題={meta.get('question_title')}｜相似度={hit['score']:.4f}")

    print("\n===== 載入 LLM =====")
    tokenizer, model = load_llm(args.llm_model)
    answer = generate_answer(tokenizer, model, args.question, context)

    print("\n===== RAG 回答 =====")
    print(answer)


if __name__ == "__main__":
    main()
