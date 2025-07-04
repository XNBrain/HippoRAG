import os
from typing import List
from dotenv import load_dotenv
from docx import Document

from src.hipporag import HippoRAG

# 读取 doc 文件（.doc 需另存为 .docx 或用 python-docx 兼容库处理）
def load_docx_text(doc_path: str) -> List[str]:
    document = Document(doc_path)
    return [para.text.strip() for para in document.paragraphs if para.text.strip()]

def main():
    # 1. 读取 .env 配置
    load_dotenv()
    llm_base_url = os.getenv("CUSTOM_LLM_API_URL")
    embedding_base_url = os.getenv("CUSTOM_EMBEDDING_API_URL")
    
    # 2. Word 文件路径（请确保另存为 .docx 格式）
    docx_path = r"D:\item\HippoRAG\data\application.docx"
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"找不到 Word 文件: {docx_path}")
    
    # 3. 加载文档文本
    docs = load_docx_text(docx_path)
    
    # 4. 初始化 HippoRAG（使用你的代理 API）
    hipporag = HippoRAG(
        save_dir='outputs/project_doc',
        llm_model_name='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        llm_base_url=llm_base_url,
        embedding_model_name='text-embedding-3-small',
        embedding_base_url=embedding_base_url,
    )

    # 5. 文档向量索引
    hipporag.index(docs=docs)

    # 6. 提问
    queries = [
        "申请书中，申请人提到了哪些政治理论学习内容?",
        "申请人通过哪些具体事件或事迹，表达了对中国共产党领导能力和宗旨的认同？",
    ]

    answers = hipporag.rag_qa(queries=queries)
    for i, q in enumerate(queries):
        print(f"\n❓ {q}\n➡️ {answers[0][i]}")

if __name__ == "__main__":
    main()
