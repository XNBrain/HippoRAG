import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

from src.hipporag import HippoRAG

def main():
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    # 从环境变量读取你在 .env 中配置的 LLM 和 embedding 服务地址
    llm_base_url = os.getenv("CUSTOM_LLM_API_URL")
    embedding_base_url = os.getenv("CUSTOM_EMBEDDING_API_URL")

    # 模型名称（deepseek 为例）
    llm_model_name = 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'
    embedding_model_name = 'text-embedding-3-small'

    save_dir = 'outputs/openai'

    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model_name,
        embedding_base_url=embedding_base_url,
    )

    hipporag.index(docs=docs)

    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        [
            "Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince."
        ],
        [
            "Erik Hort's birthplace is Montebello.",
            "Montebello is a part of Rockland County."
        ]
    ]

    print(hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers))


if __name__ == "__main__":
    main()
