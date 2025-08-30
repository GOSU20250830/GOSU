import os
import re
import json
import asyncio
from GOSU.GOSU import GOSU, QueryParam
from tqdm import tqdm
from GOSU.llm import openai_complete_if_cache, openai_embedding
from GOSU.utils import EmbeddingFunc
import numpy as np
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
def extract_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    data = data.replace("**", "")
    queries = re.findall(r"- Question \d+: (.+)", data)
    return queries
async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()
    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True
        for query_text in tqdm(queries, desc="Processing queries", unit="query"):
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )
            print("query: "+query_text)
            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")
        result_file.write("\n]")
        result_file.flush()
if __name__ == "__main__":
    cls = ""
    mode = "ERS"
    WORKING_DIR = f"Your file path"
    rag = GOSU(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embedding_func
        ),
    )
    query_param = QueryParam(mode=mode)
    questions_dir = "Your file path"
    answers_dir = f"Your file path"
    queries = extract_queries(f"Your file path")
    run_queries_and_save_to_json(
        queries, rag, query_param, f"Your file path", f"Your file path"
    )