import os
import json
import time
import numpy as np
from GOSU.GOSU import GOSU
from GOSU.utils import EmbeddingFunc
from GOSU.llm import openai_complete_if_cache, openai_embedding
from dotenv import load_dotenv
load_dotenv()

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
def insert_text(rag, file_path):
    with open(file_path, mode="r", encoding='utf-8') as f:
        unique_contexts = json.load(f)
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}.")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries.")
cls = ""
WORKING_DIR = f"Your file path"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
rag = GOSU(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embedding_func
    ),
)
insert_text(rag, f"Your file path")
