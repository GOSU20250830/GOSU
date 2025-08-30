import json
from openai import OpenAI
from transformers import GPT2Tokenizer
from dotenv import load_dotenv
from GOSU.llm import openai_complete_if_cache
import os
import asyncio
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


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def get_summary(context, tot_tokens=2000):
    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000 : 1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens) : 1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary


clses = [""]
for cls in clses:
    with open(f"Your file path", mode="r", encoding='utf-8') as f:
        unique_contexts = json.load(f)

    summaries = [get_summary(context) for context in unique_contexts]

    total_description = "\n\n".join(summaries)

    prompt = f"""
    Given the following description of a dataset:

    {total_description}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """


    result = asyncio.run(llm_model_func(prompt=prompt))


    file_path = f"Your file path"
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(result)

    print(f"{cls}_questions written to {file_path}.")
