import os
import asyncio
from GOSU.GOSU import GOSU, QueryParam
from GOSU.llm import openai_complete_if_cache, openai_embedding
from GOSU.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
load_dotenv()
WORKING_DIR = "Your file path"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Selected Generation Model Name",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_APIKEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        **kwargs,
    )
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="Selected Embedding Model Name",
        api_key=os.getenv("EM_APIKEY"),
        base_url=os.getenv("EM_BASE_URL"),
    )
async def main():
    try:
        rag = GOSU(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func,
            ),
        )
        with open("Your file path") as f:
            await rag.ainsert(f.read())
        print(
            await rag.aquery(
                "Your question",
                param=QueryParam(mode="ERS"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    asyncio.run(main())