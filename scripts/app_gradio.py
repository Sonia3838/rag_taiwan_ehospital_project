import argparse

import gradio as gr

from chat_cli import build_context, generate_answer, load_llm
from retriever import MedicalQARetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--collection", default="taiwan_ehospital_qa")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    retriever = MedicalQARetriever(args.persist_dir, args.collection, args.embedding_model)
    tokenizer, model = load_llm(args.llm_model)

    def answer(question):
        hits = retriever.search(question, top_k=args.top_k)
        context = build_context(hits)
        rag_answer = generate_answer(tokenizer, model, question, context)
        sources = "\n".join([
            f"{i}. 問題編號={h['metadata'].get('問題編號')}｜標題={h['metadata'].get('question_title')}｜相似度={h['score']:.4f}"
            for i, h in enumerate(hits, 1)
        ])
        return rag_answer, sources

    demo = gr.Interface(
        fn=answer,
        inputs=gr.Textbox(lines=6, label="輸入醫療問題"),
        outputs=[
            gr.Textbox(lines=12, label="RAG 回答"),
            gr.Textbox(lines=8, label="檢索來源"),
        ],
        title="台灣 e 院醫療問答 RAG 系統",
        description="此系統僅供期末報告與學術展示，不可取代醫師診斷。",
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
