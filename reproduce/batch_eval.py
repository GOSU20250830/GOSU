import re
import json
import jsonlines
import asyncio
from dotenv import load_dotenv
import os
from GOSU.llm import openai_complete_if_cache
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
async def batch_eval(query_file, result1_file, result2_file, output_file_path):
    with open(query_file, "r", encoding='utf-8') as f:
        data = f.read()
    queries = re.findall(r"- Question \d+: (.+)", data)
    with open(result1_file, "r", encoding='utf-8') as f:
        answers1 = json.load(f)
    answers1 = [i["result"] for i in answers1]
    with open(result2_file, "r", encoding='utf-8') as f:
        answers2 = json.load(f)
    answers2 = [i["result"] for i in answers2]
    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        print(f"{i} / {len(queries)}")
        sys_prompt = """
                ---Role---
                You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
                """
        prompt = f"""
                You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

                - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
                - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
                - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

                For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

                Here is the question:
                {query}

                Here are the two answers:

                **Answer 1:**
                {answer1}

                **Answer 2:**
                {answer2}

                Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

                Output your evaluation in the following JSON format:

                {{
                    "Comprehensiveness": {{
                        "Winner": "[Answer 1 or Answer 2 or neither]",
                        "Explanation": "[Provide explanation here]"
                    }},
                    "Empowerment": {{
                        "Winner": "[Answer 1 or Answer 2 or neither]",
                        "Explanation": "[Provide explanation here]"
                    }},
                    "Overall Winner": {{
                        "Winner": "[Answer 1 or Answer 2 or neither]",
                        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
                    }}
                }}
                """
        evaluation_result = await llm_model_func(
            prompt,
            system_prompt=sys_prompt,
        )
        requests.append(
            {
                "custom_id": f"request-{i+1}",
                "evaluation": evaluation_result,
            }
        )
    with jsonlines.open(output_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)
    print(f"Batch evaluation results written to {output_file_path}.")
if __name__ == "__main__":
    cls = "Your file path"
    method1 = "ERS"
    method2 = ""
    asyncio.run(batch_eval(f"Your file path",
                           f"Your file path",
                           f"Your file path",
                           f"Your file path"))