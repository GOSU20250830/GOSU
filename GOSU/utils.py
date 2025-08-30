import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

ENCODER = None

logger = logging.getLogger("GOSU")


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_combine_contexts(hl, ll):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result

def process_combine_contexts_multi(
    *csv_section_strs: str | None,
    dedupe: bool = True,
    keep_order: str = "first",   # "first"=保留首次出现；"last"=用最后版本覆盖
    fill_missing: bool = True,
) -> str:
    """
    合并多个 CSV 文本片段（每段格式类似你现有输出），并重新编号。

    参数
    ----
    *csv_section_strs:
        任意数量的 csv 字符串（不含围栏；若含围栏也会尝试剥掉）。
        可以是 None 或空串，将被忽略。

    dedupe:
        是否去重（按除 id 列外的剩余列组合）。默认 True。

    keep_order:
        当 dedupe=True 时，冲突处理方式：
        - "first": 保留首次出现的行（后来的丢弃） —— 保持你原函数行为
        - "last" : 后来者覆盖先前（更新字段）

    fill_missing:
        对列数不足的行是否补空列以对齐 header。默认 True。

    返回
    ----
    合并后的 CSV 字符串（含 header）。若全部为空，返回 ""。
    """

    # ---------- 1. 预清理 & 过滤空 ----------
    cleaned: list[str] = []
    for s in csv_section_strs:
        if not s:
            continue
        txt = s.strip()
        if not txt:
            continue
        # 粗略去掉围栏残留（比如 ```csv ... ```），以防误传整段
        if txt.startswith("```"):
            # 尝试只取 fenced code block 内部
            # NOTE: 简单拆解；若格式严格可以 regex
            parts = txt.split("```")
            if len(parts) >= 3:
                # 结构: ['', 'csv', 'body', ...] or ['```csv', 'body', ...]
                txt = parts[2] if parts[1].strip().startswith("csv") else parts[1]
            txt = txt.strip()
        cleaned.append(txt)

    if not cleaned:
        return ""

    # ---------- 2. 解析为 list-of-list ----------
    parsed = []
    for s in cleaned:
        tbl = csv_string_to_list(s)
        if tbl:
            parsed.append(tbl)

    if not parsed:
        return ""

    # ---------- 3. 基准 header ----------
    base_header = parsed[0][0]  # e.g., ["id","entity","type","description","rank"]
    num_cols = len(base_header)

    # ---------- 4. 收集所有行 ----------
    rows_no_id: list[list[str]] = []
    # 用于去重
    seen: dict[tuple, int] = {}  # 记录首次 index

    for tbl in parsed:
        if not tbl:
            continue
        header = tbl[0]

        # 逐数据行
        for row in tbl[1:]:
            if not row:
                continue
            cells = row[1:]  # drop id

            # 对齐列数
            if fill_missing:
                if len(cells) < num_cols - 1:
                    cells = cells + [""] * (num_cols - 1 - len(cells))
                elif len(cells) > num_cols - 1:
                    cells = cells[: num_cols - 1]

            key = tuple(cells)
            if not dedupe:
                rows_no_id.append(cells)
            else:
                if keep_order == "first":
                    if key in seen:
                        continue
                    seen[key] = len(rows_no_id)
                    rows_no_id.append(cells)
                else:  # keep_order == "last"
                    if key in seen:
                        # 覆盖旧行
                        rows_no_id[seen[key]] = cells
                    else:
                        seen[key] = len(rows_no_id)
                        rows_no_id.append(cells)

    # ---------- 5. 重排 id & 输出 ----------
    out_lines = [",\t".join(base_header)]
    for i, cells in enumerate(rows_no_id, start=1):
        out_lines.append(f"{i},\t" + ",\t".join(cells))

    return "\n".join(out_lines)

