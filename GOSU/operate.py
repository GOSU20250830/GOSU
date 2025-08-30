import asyncio
import json
import re
import hashlib
from typing import Union, Any
from collections import Counter, defaultdict
import warnings
from itertools import combinations
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    process_combine_contexts_multi,
    locate_json_string_body_from_string,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
BATCH_LIMIT = 64
QUOTE_RE = re.compile(r'^[\'"]+|[\'"]+$')
BIG_ORDER = 10**9

def chunking_by_token_size(
    content: str,
    overlap_token_size=128,
    max_token_size=1024,
    tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size],
            model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    already_sem_units = []
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        already_sem_units.extend(
            split_string_by_multi_markers(already_node.get("source_semunit_id", ""), [GRAPH_FIELD_SEP])
        )
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    new_sem_units = [dp.get("source_semunit_id", "") for dp in nodes_data]
    source_semunit_id = GRAPH_FIELD_SEP.join(sorted(set(new_sem_units + already_sem_units)))
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        source_semunit_id=source_semunit_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data

async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_sem_units = []
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )
        already_sem_units.extend(
            split_string_by_multi_markers(already_edge.get("source_semunit_id", ""), [GRAPH_FIELD_SEP])
        )
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    new_sem_units = [dp.get("source_semunit_id", "") for dp in edges_data]
    source_semunit_id = GRAPH_FIELD_SEP.join(sorted(set(new_sem_units + already_sem_units)))
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            source_semunit_id=source_semunit_id,
        ),
    )
    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_semunit_id=source_semunit_id,
    )
    return edge_data

async def extract_entities_directly(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
    already_processed = 0
    already_entities = 0
    already_relations = 0
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, "
            f"{already_entities} entities(duplicated), {already_relations} relations(duplicated).\r",
            end="", flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    results = await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[_merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
          for k, v in maybe_nodes.items()]
    )
    all_relationships_data = await asyncio.gather(
        *[_merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
          for k, v in maybe_edges.items()]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working.")
        return None
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships, maybe your LLM is not working.")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"] + dp["src_id"] + dp["tgt_id"] + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)
    return knowledge_graph_inst

async def _handle_single_semantic_unit_extraction(
    record_attributes: list[str],
    chunk_key: str,
    full_doc_id: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"semantic_unit"':
        return None
    try:
        summary = clean_str(record_attributes[1])
        if not summary.strip():
            return None
        content = clean_str(record_attributes[2])
        unit_id = compute_mdhash_id(summary, prefix="semantic_unit-")
        return dict(
            semantic_unit_id=unit_id,
            unit_summary=summary,
            unit_content=content,
            source_chunk_id=chunk_key,
            full_doc_id=full_doc_id,
        )
    except Exception as e:
        logger.warning(f"[ParseError] in semantic_unit: {e}.")
        return None

async def extract_semantic_units(
    chunks:  dict[str, TextChunkSchema],
    semantic_unit_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[None, dict[str, dict]]:
    use_llm_func: callable = global_config["llm_model_func"]
    semantic_unit_extract_max_gleaning = global_config["semantic_unit_max_gleaning"]
    ordered_chunks = list(chunks.items())
    semantic_unit_extract_prompt = PROMPTS["semantic_unit_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )
    continue_prompt = PROMPTS["semantic_unit_continue_extraction"]
    if_loop_prompt = PROMPTS["semantic_unit_if_loop_extraction"]
    already_processed = 0
    already_semantic_units = 0
    all_semantic_units = {}
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_semantic_units
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        full_doc_id = chunk_dp["full_doc_id"]
        hint_prompt = semantic_unit_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(semantic_unit_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == semantic_unit_extract_max_gleaning - 1:
                break
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        maybe_semantic_units = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_semantic_units = await _handle_single_semantic_unit_extraction(
                record_attributes, chunk_key, full_doc_id
            )
            if if_semantic_units is not None:
                maybe_semantic_units[if_semantic_units["unit_summary"]].append(if_semantic_units)
                continue
        already_processed += 1
        already_semantic_units += len(maybe_semantic_units)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_semantic_units} semantic_units(duplicated).\r",
            end="",
            flush=True,
        )
        return dict(maybe_semantic_units)
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()
    semantic_unit_docs = {}
    for unit_dict in results:
        for unit_list in unit_dict.values():
            for dp in unit_list:
                unit_id = dp["semantic_unit_id"]
                semantic_unit_docs[unit_id] = {
                    "unit_summary": dp["unit_summary"],
                    "content": dp["unit_summary"] + dp["unit_content"],
                    "source_chunk_id": dp["source_chunk_id"],
                    "full_doc_id": dp["full_doc_id"],
                }
    if not semantic_unit_docs:
        logger.warning("No semantic units extracted.")
        return None
    logger.info(f"[Semantic Unit Extraction] extracted {len(semantic_unit_docs)} semantic units.")
    return semantic_unit_docs

async def extract_entities_by_semantic_units(
    semantic_unit_id: str,
    summary_txt: str,
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS["sem_unit_centered_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
    already_processed = 0
    already_entities = 0
    already_relations = 0
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, semantic_unit=summary_txt, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key)
            if if_entities is not None:
                if_entities["source_semunit_id"] = semantic_unit_id
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                if_relation["source_semunit_id"] = semantic_unit_id
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, "
            f"{already_entities} entities(duplicated), {already_relations} relations(duplicated).\r",
            end="", flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    results = await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[_merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
          for k, v in maybe_nodes.items()]
    )
    all_relationships_data = await asyncio.gather(
        *[_merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
          for k, v in maybe_edges.items()]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working.")
        return None
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships, maybe your LLM is not working.")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": summary_txt + dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
                "source_semunit_id": dp["source_semunit_id"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": summary_txt + dp["keywords"] + dp["src_id"] + dp["tgt_id"] + dp["description"],
                "source_semunit_id": dp["source_semunit_id"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)
    return knowledge_graph_inst

async def semantic_unit_pairs_judge_by_similarity(
    embedding_func,
    unit_docs: Dict[str, dict],
    threshold: float = 0.70,
) -> List[Tuple[str, str, float, float]]:
    unit_ids = list(unit_docs.keys())
    summaries = [unit_docs[u].get("unit_summary", "") for u in unit_ids]
    contents  = [unit_docs[u].get("content",        "") for u in unit_ids]
    if len(unit_ids) < 2:
        return []
    async def _embed(texts: List[str]) -> np.ndarray:
        batches = [texts[i : i + BATCH_LIMIT] for i in range(0, len(texts), BATCH_LIMIT)]
        embs = await asyncio.gather(*(embedding_func(b) for b in batches))
        return np.asarray([v for batch in embs for v in batch], dtype=np.float32)
    embeds_sum  = await _embed(summaries)
    embeds_cont = await _embed(contents)
    embeds_sum  /= np.linalg.norm(embeds_sum,  axis=1, keepdims=True) + 1e-12
    embeds_cont /= np.linalg.norm(embeds_cont, axis=1, keepdims=True) + 1e-12
    sim_sum  = embeds_sum  @ embeds_sum.T
    sim_cont = embeds_cont @ embeds_cont.T
    has_sum  = [bool(txt.strip()) for txt in summaries]
    has_cont = [bool(txt.strip()) for txt in contents]
    pairs: List[Tuple[str, str, float, float]] = []
    N = len(unit_ids)
    for i, j in combinations(range(N), 2):
        need_sum  = has_sum[i]  and has_sum[j]
        need_cont = has_cont[i] and has_cont[j]
        ok_sum  = (not need_sum)  or (sim_sum[i, j]  > threshold)
        ok_cont = (not need_cont) or (sim_cont[i, j] > threshold)
        if ok_sum and ok_cont:
            pairs.append(
                (unit_ids[i], unit_ids[j], float(sim_sum[i, j]), float(sim_cont[i, j]))
            )
    pairs.sort(key=lambda t: (t[3], t[2]), reverse=True)
    logger.info(f"[similarity_candidates_semantic_unit] qualified pairs: {len(pairs)}.")
    return pairs

async def semantic_unit_pairs_judge_by_LLM(
    llm_call,
    unit1: Dict,
    unit2: Dict,
) -> bool:
    similarity_prompt_temp = PROMPTS["judge_sim_semantic_unit"]
    similarity_prompt = similarity_prompt_temp.format(su1=unit1["unit_summary"] + " —— " + unit1["content"], su2=unit2["unit_summary"] + " —— " + unit2["content"])
    result = await llm_call(similarity_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        similarity_data = json.loads(json_text)
        return bool(similarity_data.get("result"))
    except Exception:
        return False

async def similar_semunit_unit(
    self,
    semantic_units: Dict[str, Dict],
    coarse_threshold: float = 0.70,
) -> List[Tuple[str, str]]:
    coarse_pairs = await semantic_unit_pairs_judge_by_similarity(
        self.embedding_func,
        semantic_units,
        threshold=coarse_threshold,
    )
    final_pairs: List[Tuple[str, str]] = []
    for uid1, uid2, *_ in coarse_pairs:
        u1 = semantic_units[uid1]
        u2 = semantic_units[uid2]
        try:
            same_flag = await semantic_unit_pairs_judge_by_LLM(
                self.llm_model_func,
                u1,
                u2,
            )
            if same_flag:
                final_pairs.append((uid1, uid2))
        except Exception as e:
            logger.error(f"LLM judge error for ({uid1}, {uid2}): {e}.")
    logger.info(f"[similarity_semunit_final] qualified pairs: {len(final_pairs)}.")
    return final_pairs

def merge_similar_semantic_units(
    semunit_dict: Dict[str, Dict],
    sim_pairs: List[Tuple[str, str]],
) -> Dict[str, Dict]:
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    for uid in semunit_dict:
        parent[uid] = uid
    for u1, u2 in sim_pairs:
        if u1 in semunit_dict and u2 in semunit_dict:
            union(u1, u2)
    groups = {}
    for uid in semunit_dict:
        root = find(uid)
        groups.setdefault(root, []).append(uid)
    for root, members in groups.items():
        if len(members) > 1:
            print(f"Merge group root={root}: {members}")
    for group in groups.values():
        if len(group) == 1:
            continue
        main_uid = group[0]
        semunit_dict[main_uid]["_merged"] = True
        summaries = set()
        contents = set()
        chunk_ids = set()
        doc_ids = set()
        for uid in group:
            su = semunit_dict[uid]
            summaries.add(su["unit_summary"])
            contents.add(su["content"])
            chunk_ids.add(su["source_chunk_id"])
            doc_ids.add(su["full_doc_id"])
            if uid != main_uid:
                semunit_dict.pop(uid)
        semunit_dict[main_uid]["unit_summary"] = " /// ".join(summaries)
        semunit_dict[main_uid]["content"] = " /// ".join(contents)
        semunit_dict[main_uid]["source_chunk_id"] = ";;;".join(chunk_ids)
        semunit_dict[main_uid]["full_doc_id"] = ";;;".join(doc_ids)
    before_cnt = len(parent)
    after_cnt = len(semunit_dict)
    removed_cnt = before_cnt - after_cnt
    logger.info(
        f"[merge_similar_semantic_units] extracted {before_cnt}, merged {removed_cnt} units → remaining {after_cnt} units."
    )
    return semunit_dict

async def get_chunks_for_semunit(
    semunit_dict: Dict[str, Dict],
    semunit_id: str,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
) -> List[str]:
    if semunit_id not in semunit_dict:
        raise ValueError(f"{semunit_id} not found in semunit_dict")
    cid_field = semunit_dict[semunit_id].get("source_chunk_id", "")
    if not cid_field:
        return []
    chunk_ids = [cid.strip() for cid in cid_field.split(";;;") if cid.strip()]
    chunk_vals = await text_chunks_db.get_by_ids(chunk_ids)
    chunks = [cv["content"] for cv in chunk_vals if cv]
    missing = set(chunk_ids) - {
        cid for cid, cv in zip(chunk_ids, chunk_vals) if cv
    }
    for cid in missing:
        logger.warning(f"Chunk ID {cid} not found in text_chunks storage")
    return chunks

async def hybrid_retriever_context_chunk(
    query_text: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedding_func,
    top_k: int = 5,
    threshold: float = 0.70,
    expand_ratio: int = 3,
) -> List[Tuple[str, float, str]]:
    expand_k = expand_ratio * top_k
    rough_results = await chunks_vdb.query(query_text, top_k=expand_k)
    if not rough_results:
        return []
    chunk_ids   = [r["id"] for r in rough_results]
    chunk_vals  = await text_chunks_db.get_by_ids(chunk_ids)
    chunk_texts = [cv["content"] for cv in chunk_vals if cv]
    q_vec = await embedding_func([query_text])
    q_vec = q_vec[0] / (np.linalg.norm(q_vec[0]) + 1e-12)
    async def _embed_batches(texts):
        batches = [
            texts[i:i + BATCH_LIMIT]
            for i in range(0, len(texts), BATCH_LIMIT)
        ]
        embs = await asyncio.gather(*(embedding_func(b) for b in batches))
        return np.vstack(embs)
    chunk_vecs = await _embed_batches(chunk_texts)
    chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-12
    sims = chunk_vecs @ q_vec
    qualified = [
        (txt, float(sim), cid)
        for txt, sim, cid in zip(chunk_texts, sims, chunk_ids)
        if sim > threshold
    ]
    qualified.sort(key=lambda t: t[1], reverse=True)
    return qualified[:top_k]

async def get_target_kg_single_semunit(
    self,
    semunit_dict: Dict[str, Dict],
    semunit_id: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    top_k: int = 5,
    threshold=0.65,
) -> Tuple[Dict, str]:
    base_chunks = await get_chunks_for_semunit(
        semunit_dict, semunit_id, text_chunks_db
    )
    query_txt  = semunit_dict[semunit_id]["unit_summary"]
    extra_ctx = await hybrid_retriever_context_chunk(query_txt, chunks_vdb, text_chunks_db, self.embedding_func, top_k=top_k, threshold=threshold, expand_ratio=3)
    logger.debug(
        f"[Enrich_sem_unit] {semunit_id} sims: {[round(t[1], 3) for t in extra_ctx]}"
    )
    extra_chunks = [t[0] for t in extra_ctx]
    extra_cids   = [t[2] for t in extra_ctx]
    extra_vals = await text_chunks_db.get_by_ids(extra_cids)
    extra_fids = [
        cv["full_doc_id"] for cv in extra_vals if cv and "full_doc_id" in cv
    ]
    old_cids = [cid.strip() for cid in semunit_dict[semunit_id]
                .get("source_chunk_id", "").split(";;;") if cid.strip()]
    all_cids = sorted(set(old_cids) | set(extra_cids))
    semunit_dict[semunit_id]["source_chunk_id"] = ";;;".join(all_cids)
    old_fids = [fid.strip() for fid in semunit_dict[semunit_id]
                .get("full_doc_id", "").split(";;;") if fid.strip()]
    all_fids = sorted(set(old_fids) | set(extra_fids))
    semunit_dict[semunit_id]["full_doc_id"] = ";;;".join(all_fids)
    unique_chunks = list(set(base_chunks + extra_chunks))
    context_text  = " ".join(unique_chunks)
    merged_flag = semunit_dict[semunit_id].get("_merged", False)
    if merged_flag or len(all_cids) > 1:
        raw_unit = semunit_dict[semunit_id]["unit_summary"]
        prompt = PROMPTS["enhance_sem_unit"].format(
            RU=raw_unit,
            CT=context_text,
        )
        result = await self.llm_model_func(prompt)
        json_text = locate_json_string_body_from_string(result)
        try:
            enhanced = json.loads(json_text)
            summary_val = enhanced.get("unit_summary", [])
            if isinstance(summary_val, list):
                semunit_dict[semunit_id]["unit_summary"] = ", ".join(summary_val)
            else:
                semunit_dict[semunit_id]["unit_summary"] = str(summary_val)
            content_val = enhanced.get("unit_content", [])
            if isinstance(content_val, list):
                semunit_dict[semunit_id]["content"] = " ".join(content_val)
            else:
                semunit_dict[semunit_id]["content"] = str(content_val)
        except Exception as e:
            logger.error(f"[Enhance_sem_unit] LLM or JSON error: {e}.")
    logger.info(f"[Enrich_sem_unit] {semunit_id}: +{len(extra_cids)} chunks, source_chunk_id={len(all_cids)}, full_doc_id={len(all_fids)}.")
    return semunit_dict[semunit_id], context_text

async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1
            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]
    if not all_text_units:
        logger.warning("No valid text units found.")
        return []
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()
    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data

async def _find_most_related_semantic_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    semantic_units_kv: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
) -> list[dict]:
    if not node_datas:
        return []
    per_seed_sem_lists: list[list[str]] = []
    for nd in node_datas:
        src_field = nd.get("source_semunit_id", "")
        if not src_field:
            node = await knowledge_graph_inst.get_node(nd["entity_name"])
            if node:
                src_field = node.get("source_semunit_id", "")
        sem_ids = (
            [s.strip() for s in src_field.split(GRAPH_FIELD_SEP) if s.strip()]
            if src_field else []
        )
        per_seed_sem_lists.append(sem_ids)
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(nd["entity_name"]) for nd in node_datas]
    )
    all_one_hop_nodes = set()
    for nd, this_edges in zip(node_datas, edges):
        if not this_edges:
            continue
        e_name = nd["entity_name"]
        for src, tgt in this_edges:
            other = tgt if src == e_name else src
            all_one_hop_nodes.add(other)
    all_one_hop_nodes = list(all_one_hop_nodes)
    hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(nm) for nm in all_one_hop_nodes]
    )
    neighbor_sem_lookup: dict[str, set[str]] = {}
    for nm, nd in zip(all_one_hop_nodes, hop_nodes_data):
        if not nd:
            continue
        src_field = nd.get("source_semunit_id", "")
        if not src_field:
            continue
        neighbor_sem_lookup[nm] = {
            s.strip() for s in src_field.split(GRAPH_FIELD_SEP) if s.strip()
        }
    semunit_meta: dict[str, dict] = {}
    for seed_idx, (seed_nd, sem_list, this_edges) in enumerate(
        zip(node_datas, per_seed_sem_lists, edges)
    ):
        neighs = set()
        if this_edges:
            e_name = seed_nd["entity_name"]
            for src, tgt in this_edges:
                neighs.add(tgt if src == e_name else src)
        for uid in sem_list:
            meta = semunit_meta.setdefault(uid, {"order": seed_idx, "relation_counts": 0})
            rc_add = 0
            for nb in neighs:
                nb_set = neighbor_sem_lookup.get(nb)
                if nb_set and uid in nb_set:
                    rc_add += 1
            meta["relation_counts"] += rc_add
    if not semunit_meta:
        return []
    uid_list = list(semunit_meta.keys())
    kv_vals = await semantic_units_kv.get_by_ids(uid_list)
    rows: list[dict] = []
    for uid, kv in zip(uid_list, kv_vals):
        if not kv:
            continue
        rows.append(
            {
                "unit_id": uid,
                "unit_summary": kv.get("unit_summary", "").strip(),
                "unit_content": kv.get("unit_content", kv.get("content", "")).strip(),
                **semunit_meta[uid],
            }
        )
    if not rows:
        return []
    rows.sort(key=lambda r: (r["order"], -r["relation_counts"]))
    rows = truncate_list_by_token_size(
        rows,
        key=lambda r: r["unit_content"],
        max_token_size=query_param.max_token_for_semantic_unit,
    )
    return rows

def normalize_for_match(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    if (t.startswith('"""') and t.endswith('"""')) or (t.startswith("'''") and t.endswith("'''")):
        t = t[3:-3].strip()
    t = QUOTE_RE.sub("", t)
    t = " ".join(t.split())
    return t

def strip_leading_summary(raw: str, summary: str) -> str:
    if not raw or not summary:
        return raw
    norm_summary = normalize_for_match(summary)
    if not norm_summary:
        return raw
    window_len = int(len(norm_summary) * 1.4) + 8
    candidate_prefix = raw[:window_len]
    norm_candidate = normalize_for_match(candidate_prefix)
    if not norm_candidate.lower().startswith(norm_summary.lower()):
        return raw
    pattern = re.escape(norm_summary)
    pattern = re.sub(r'\\\s+', r'\\s+', pattern)
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(candidate_prefix)
    if not match:
        cut_pos = len(candidate_prefix)
    else:
        cut_pos = match.end()
    trimmed = raw[cut_pos:].lstrip()
    if len(trimmed) < 0.2 * len(raw) and not re.search(r'[.,;!?。！？\s]', trimmed):
        return raw
    while trimmed.startswith('"') or trimmed.startswith("'"):
        trimmed = trimmed[1:].lstrip()
    return trimmed or raw

async def _build_entity_level_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    semantic_units_kv: BaseKVStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam
):
    results = await entities_vdb.query(keywords, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    use_semantic_units = await _find_most_related_semantic_unit_from_entities(
        node_datas, query_param, semantic_units_kv, knowledge_graph_inst
    )
    logger.info(
        f"Entity level query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_semantic_units)} semantic units, {len(use_text_units)} text units."
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    semu_section_list = [["id", "summary", "content", "order", "relation_counts"]]
    for i, su in enumerate(use_semantic_units):
        summary = su.get("unit_summary", "") or ""
        raw = su.get("unit_content", "") or ""
        cleaned = strip_leading_summary(raw, summary)
        cleaned = " ".join(cleaned.split())
        semu_section_list.append(
            [
                i,
                summary,
                cleaned,
                su.get("order", -1),
                su.get("relation_counts", 0),
            ]
        )
    semantic_units_context = list_of_list_to_csv(semu_section_list)
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    context_md = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Semantic-Units-----
```csv
{semantic_units_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    return context_md, use_semantic_units, node_datas, use_relations, use_text_units

async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()
    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    return node_datas

async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [  # type: ignore
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None  # type: ignore
    ]  # type: ignore
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]  # type: ignore
    return all_text_units

async def _find_related_sem_units_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    semantic_units_kv: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
) -> list[dict]:
    su_lookup: dict[str, dict] = {}
    for edge_idx, ed in enumerate(edge_datas):
        src_field = ed.get("source_semunit_id", "")
        if not src_field:
            continue
        for uid in src_field.split(GRAPH_FIELD_SEP):
            uid = uid.strip()
            if not uid:
                continue
            srec = su_lookup.get(uid)
            if srec is None:
                su_lookup[uid] = {"order": edge_idx, "relation_counts": 1}
            else:
                srec["relation_counts"] += 1
    if not su_lookup:
        logger.warning("No semantic units referenced by provided relationships.")
        return []
    uid_list = list(su_lookup.keys())
    kv_vals = await semantic_units_kv.get_by_ids(uid_list)
    out_rows: list[dict] = []
    for uid, kv in zip(uid_list, kv_vals):
        if not kv:
            continue
        unit_content = kv.get("unit_content", kv.get("content", ""))
        out_rows.append(
            {
                "unit_id": uid,
                "unit_summary": kv.get("unit_summary", "").strip(),
                "unit_content": unit_content,
                "order": su_lookup[uid]["order"],
                "relation_counts": su_lookup[uid]["relation_counts"],
            }
        )
    if not out_rows:
        return []
    out_rows.sort(key=lambda r: (-r["relation_counts"], r["order"]))
    out_rows = truncate_list_by_token_size(
        out_rows,
        key=lambda x: x["unit_content"],
        max_token_size=query_param.max_token_for_semantic_unit,
    )
    return out_rows

async def _build_relationship_level_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    semantic_units_kv: BaseKVStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
    if not len(results):
        return None
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )
    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged.")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "src_tgt": (k["src_id"], k["tgt_id"]),
            "rank": d,
            **v,
        }
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_semantic_units = await _find_related_sem_units_from_relationships(
        edge_datas, query_param, semantic_units_kv, knowledge_graph_inst
    )
    logger.info(
        f"Relationship level query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_semantic_units)} semantic units, {len(use_text_units)} text units."
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    semu_section_list = [["id", "summary", "content", "order", "relation_counts"]]
    for i, su in enumerate(use_semantic_units):
        summary = su.get("unit_summary", "") or ""
        raw = su.get("unit_content", "") or ""
        cleaned = strip_leading_summary(raw, summary)
        cleaned = " ".join(cleaned.split())
        semu_section_list.append(
            [
                i,
                summary,
                cleaned,
                su.get("order", -1),
                su.get("relation_counts", 0),
            ]
        )
    semantic_units_context = list_of_list_to_csv(semu_section_list)
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    context_md = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Semantic-Units-----
```csv
{semantic_units_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    return context_md, use_semantic_units, use_entities, edge_datas, use_text_units

async def _find_most_related_entities_from_semantic_units(
    sem_units: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    skip_names: set[str] | None = None,
) -> list[dict]:
    skip_names = skip_names or set()
    unit_id_set = {u["unit_id"] for u in sem_units if u.get("unit_id")}
    if not unit_id_set:
        return []
    try:
        nx_graph = knowledge_graph_inst.get_nx_graph()  # type: ignore
    except Exception:
        raise RuntimeError(
            "knowledge_graph_inst.get_nx_graph() unavailable, please try again."
        )
    rows: list[dict] = []
    for node_id, data in nx_graph.nodes(data=True):
        if node_id in skip_names:
            continue
        src_field = data.get("source_semunit_id", "")
        if not src_field:
            continue
        overlap = unit_id_set.intersection(
            uid.strip() for uid in src_field.split(GRAPH_FIELD_SEP) if uid.strip()
        )
        if not overlap:
            continue
        deg = nx_graph.degree(node_id)
        rows.append(
            {
                "entity_name": node_id,
                "entity_type": data.get("entity_type", "UNKNOWN"),
                "description": data.get("description", ""),
                "rank": deg,
                "hit_count": len(overlap),
            }
        )
    if not rows:
        return []
    rows.sort(key=lambda r: (r["hit_count"], r["rank"]), reverse=True)
    max_tok = getattr(query_param, "max_token_for_semantic_context", None)
    if not max_tok:
        max_tok = query_param.max_token_for_local_context
    rows = truncate_list_by_token_size(
        rows,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    return rows

async def _find_most_related_edges_from_semantic_units(
    sem_units: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    skip_pairs: set[tuple[str, str]] | None = None,
) -> list[dict]:
    skip_pairs = skip_pairs or set()
    unit_id_set: set[str] = {u["unit_id"] for u in sem_units if u.get("unit_id")}
    if not unit_id_set:
        return []
    try:
        nx_graph = knowledge_graph_inst.get_nx_graph() # type: ignore
    except Exception as e:
        raise RuntimeError("Graph storage does not expose get_nx_graph().") from e
    rows: list[dict] = []
    for u, v, data in nx_graph.edges(data=True):
        if (u, v) in skip_pairs or (v, u) in skip_pairs:
            continue
        src_field = data.get("source_semunit_id", "")
        if not src_field:
            continue
        overlap = unit_id_set.intersection(
            uid.strip() for uid in src_field.split(GRAPH_FIELD_SEP) if uid.strip()
        )
        if not overlap:
            continue
        try:
            deg = await knowledge_graph_inst.edge_degree(u, v)
        except Exception:
            deg = nx_graph.degree(u) + nx_graph.degree(v)
        rows.append(
            {
                "src_id": u,
                "tgt_id": v,
                "src_tgt": (u, v),
                "description": data.get("description", ""),
                "keywords": data.get("keywords", ""),
                "weight": data.get("weight", 1.0),
                "source_semunit_id": src_field,
                "hit_sem_count": len(overlap),
                "rank": deg,
            }
        )
    if not rows:
        return []
    rows.sort(
        key=lambda r: (
            r.get("hit_sem_count", 0),
            r.get("rank", 0),
            r.get("weight", 0.0),
        ),
        reverse=True,
    )
    max_tok = getattr(query_param, "max_token_for_semantic_context", None)
    if not max_tok:
        max_tok = getattr(query_param, "max_token_for_global_context", 2048)
    rows = truncate_list_by_token_size(
        rows,
        key=lambda x: x.get("description", ""),
        max_token_size=query_param.max_token_for_global_context,
    )
    return rows

async def _find_most_related_text_unit_from_semantic_units(
    sem_units: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    skip_chunk_ids: set[str] | None = None,
) -> list[TextChunkSchema]:
    skip_chunk_ids = skip_chunk_ids or set()
    unit_id_set: set[str] = {u["unit_id"] for u in sem_units if u.get("unit_id")}
    if not unit_id_set:
        logger.warning("[_find_most_related_text_unit_from_semantic_units] empty sem_units.")
        return []
    semu_chunk_lists: list[list[str]] = []
    for su in sem_units:
        if "src_chunk_ids" in su and isinstance(su["src_chunk_ids"], (list, tuple)):
            cid_list = [c for c in su["src_chunk_ids"] if c]
        else:
            raw = su.get("source_chunk_id", "")
            cid_list = [c.strip() for c in raw.split(";;;") if c.strip()]
        semu_chunk_lists.append(cid_list)
    nx_graph = knowledge_graph_inst.get_nx_graph()  # type: ignore
    hit_ent_chunks: dict[str, set[str]] = {}
    for node_id, data in nx_graph.nodes(data=True):
        src_field = data.get("source_semunit_id", "")
        if not src_field:
            continue
        if any(uid in unit_id_set for uid in src_field.split(GRAPH_FIELD_SEP)):
            ent_chunk_ids = split_string_by_multi_markers(
                data.get("source_id", ""), [GRAPH_FIELD_SEP]
            )
            hit_ent_chunks[node_id] = set(ent_chunk_ids)
    all_chunks_lookup: dict[str, dict] = {}
    for su_idx, cid_list in enumerate(semu_chunk_lists):
        for cid in cid_list:
            if not cid:
                continue
            entry = all_chunks_lookup.get(cid)
            if entry is None:
                all_chunks_lookup[cid] = {
                    "order": su_idx,
                    "relation_counts": 1,
                    "data": None,
                }
            else:
                entry["relation_counts"] += 1
    for ent, cid_set in hit_ent_chunks.items():
        for cid in cid_set:
            if not cid:
                continue
            entry = all_chunks_lookup.get(cid)
            if entry is None:
                all_chunks_lookup[cid] = {
                    "order": len(sem_units),
                    "relation_counts": 1,
                    "data": None,
                }
            else:
                entry["relation_counts"] += 1
    if not all_chunks_lookup:
        logger.warning("[_find_most_related_text_unit_from_semantic_units] no chunks collected.")
        return []
    cid_list = list(all_chunks_lookup.keys())
    chunk_vals = await text_chunks_db.get_by_ids(cid_list)
    for cid, cd in zip(cid_list, chunk_vals):
        if cd is not None and "content" in cd:
            all_chunks_lookup[cid]["data"] = cd
        else:
            all_chunks_lookup[cid]["data"] = None
    items = []
    for cid, meta in all_chunks_lookup.items():
        if cid in skip_chunk_ids:
            continue
        if meta["data"] is None:
            continue
        items.append(
            {
                "id": cid,
                "data": meta["data"],
                "order": meta["order"],
                "relation_counts": meta["relation_counts"],
            }
        )
    if not items:
        logger.warning("[_find_most_related_text_unit_from_semantic_units] all chunk lookups failed.")
        return []
    items.sort(key=lambda x: (x["order"], -x["relation_counts"]))
    items = truncate_list_by_token_size(
        items,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    out_chunks: list[TextChunkSchema] = [it["data"] for it in items]
    return out_chunks

def _norm_order(o):
    return o if (o is not None and o >= 0) else 10**9

async def _build_semantic_unit_level_query_context(
    keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    semantic_unit_vdb: BaseVectorStorage,
    semantic_units_kv: BaseKVStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    presel_sem_units: list[dict] | None = None,
    skip_entity_names: set[str] | None = None,
    skip_edge_pairs: set[tuple[str, str]] | None = None,
    skip_chunk_ids:    set[str] | None = None,
):
    su_cand: dict[str, dict] = {}
    if presel_sem_units:
        for su in presel_sem_units:
            uid = su.get("unit_id")
            if not uid:
                continue
            su_cand[uid] = {
                "unit_id": uid,
                "unit_summary": su.get("unit_summary", "").strip(),
                "unit_content": su.get("unit_content", ""),
                "sim0": su.get("sim0", 1.0),
                "order": su.get("order", -1),
                "relation_counts": su.get("relation_counts", 0),
                "src_chunk_ids": su.get("src_chunk_ids", []),
            }
    remain = max(0, query_param.top_k - len(su_cand))
    if remain > 0:
        vec_hits = await semantic_unit_vdb.query(keywords, top_k=remain)
        for h in vec_hits:
            uid = h["id"]
            if uid in su_cand:
                continue
            if len(su_cand) >= query_param.top_k:
                break
            su_cand[uid] = {
                "unit_id": uid,
                "unit_summary": h.get("unit_summary", "").strip(),
                "unit_content": "",
                "sim0": 1.0 - h.get("distance", 0.0) / 2,
                "order": -1,
                "relation_counts": 0,
                "src_chunk_ids": [
                    cid for cid in h.get("source_chunk_id", "").split(";;;") if cid
                ],
            }
    if not su_cand:
        return None
    uid_list = list(su_cand.keys())
    kv_vals = await semantic_units_kv.get_by_ids(uid_list)
    for uid, kv in zip(uid_list, kv_vals):
        if not kv:
            continue
        rec = su_cand[uid]
        if not rec["unit_content"]:
            rec["unit_content"] = kv.get("unit_content", kv.get("content", "")) or ""
        if kv.get("source_chunk_id"):
            rec["src_chunk_ids"] = [
                cid for cid in kv["source_chunk_id"].split(";;;") if cid
            ]
    def _norm_order(o: int | None) -> int:
        if o is None or o < 0:
            return 10 ** 9
        return o
    for su in su_cand.values():
        su.setdefault("order", -1)
        su.setdefault("relation_counts", 0)
        su.setdefault("sim0", 0.0)
    sem_units = sorted(
        su_cand.values(),
        key=lambda x: (
            _norm_order(x.get("order")),
            -x.get("relation_counts", 0),
            -x.get("sim0", 0.0),
            x.get("unit_id", "")
        )
    )
    sem_units = truncate_list_by_token_size(
        sem_units,
        key=lambda x: x["unit_content"],
        max_token_size=query_param.max_token_for_semantic_context,
    )
    if not sem_units:
        return None
    use_entities = await _find_most_related_entities_from_semantic_units(sem_units, query_param, knowledge_graph_inst, skip_entity_names)
    use_relations = await _find_most_related_edges_from_semantic_units(sem_units, query_param, knowledge_graph_inst, skip_edge_pairs)
    use_text_units = await _find_most_related_text_unit_from_semantic_units(sem_units, query_param, text_chunks_db, knowledge_graph_inst, skip_chunk_ids)
    ent_section = [["id", "entity", "type", "description", "rank"]]
    for i, ent in enumerate(use_entities):
        ent_section.append(
            [
                i,
                ent["entity_name"],
                ent.get("entity_type", "UNKNOWN"),
                ent.get("description", "UNKNOWN"),
                ent["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(ent_section)
    su_section = [["id", "summary", "content", "order", "relation_counts"]]
    for i, su in enumerate(sem_units):
        summary = su.get("unit_summary", "") or ""
        raw = su.get("unit_content", "") or ""
        cleaned = strip_leading_summary(raw, summary)
        cleaned = " ".join(cleaned.split())
        su_section.append([
            i,
            summary,
            cleaned,
            su.get("order", -1),
            su.get("relation_counts", 0),
        ])
    semantic_units_context = list_of_list_to_csv(su_section)
    rel_section = [["id", "source", "target", "description", "keywords", "weight", "rank"]]
    for i, e in enumerate(use_relations):
        rel_section.append(
            [
                i,
                e.get("src_id") or e.get("src"),
                e.get("tgt_id") or e.get("tgt"),
                e.get("desc", e.get("description", "")),
                e.get("keywords", ""),
                e.get("weight", 1.0),
                e.get("rank", 0),
            ]
        )
    relations_context = list_of_list_to_csv(rel_section)
    src_section = [["id", "content"]]
    for i, ck in enumerate(use_text_units):
        src_section.append([i, ck["content"]])
    text_units_context = list_of_list_to_csv(src_section)
    logger.info(
        "Semantic unit level query uses %d semantic units, %d entities, %d relations, %d chunks.",
        len(sem_units), len(use_entities), len(use_relations), len(use_text_units)
    )
    context_md = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Semantic-Units-----
```csv
{semantic_units_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    return context_md, sem_units, use_entities, use_relations, use_text_units

def merge_semantic_units(*lists):
    merged: dict[str, dict] = {}
    for lst in lists:
        if not lst: continue
        for su in lst:
            uid = su.get("unit_id")
            if not uid: continue
            su.setdefault("order", -1)
            su.setdefault("relation_counts", 0)
            su.setdefault("sim0", 0.0)
            su.setdefault("unit_summary", "")
            su.setdefault("unit_content", "")
            old = merged.get(uid)
            if not old:
                merged[uid] = su
            else:
                if (old.get("order", -1) < 0) and (su.get("order", -1) >= 0):
                    old["order"] = su["order"]
                if su.get("relation_counts", 0) > old.get("relation_counts", 0):
                    old["relation_counts"] = su["relation_counts"]
                if len(su.get("unit_content","")) > len(old.get("unit_content","")):
                    old["unit_content"] = su.get("unit_content","")
                if len(su.get("unit_summary","")) > len(old.get("unit_summary","")):
                    old["unit_summary"] = su.get("unit_summary","")
                if su.get("sim0",0.0) > old.get("sim0",0.0):
                    old["sim0"] = su.get("sim0",0.0)
    merged_list = list(merged.values())
    merged_list.sort(
        key=lambda x: (
            _norm_order(x.get("order")),
            -x.get("relation_counts", 0),
            -x.get("sim0", 0.0),
            x.get("unit_id","")
        )
    )
    return merged_list

def merge_entities(*lists):
    merged: dict[str, dict] = {}
    for lst in lists:
        if not lst: continue
        for ent in lst:
            name = ent.get("entity_name")
            if not name: continue
            ent.setdefault("rank", 0)
            ent.setdefault("hit_count", 0)
            ent.setdefault("direct_hit", False)
            old = merged.get(name)
            if not old:
                merged[name] = ent
            else:
                if (not old.get("direct_hit")) and ent.get("direct_hit"):
                    merged[name] = ent
                    old = ent
                if ent.get("hit_count",0) > old.get("hit_count",0):
                    old["hit_count"] = ent["hit_count"]
                if ent.get("rank",0) > old.get("rank",0):
                    old["rank"] = ent["rank"]
                if len(ent.get("description","")) > len(old.get("description","")):
                    old["description"] = ent.get("description","")
    merged_list = list(merged.values())
    merged_list.sort(
        key=lambda x: (
            -int(x.get("direct_hit", False)),
            -x.get("hit_count", 0),
            -x.get("rank", 0),
            x.get("entity_name","")
        )
    )
    return merged_list

def norm_edge_pair(r: dict):
    if "src_tgt" in r and r["src_tgt"]:
        a, b = r["src_tgt"]
    else:
        a = r.get("src_id") or r.get("src")
        b = r.get("tgt_id") or r.get("tgt")
    if not a or not b:
        return None
    return (a, b) if a <= b else (b, a)

def merge_relations(*lists):
    merged: dict[tuple[str,str], dict] = {}
    for lst in lists:
        if not lst: continue
        for rel in lst:
            key = norm_edge_pair(rel)
            if not key:
                continue
            rel.setdefault("hit_sem_count", 0)
            rel.setdefault("rank", 0)
            rel.setdefault("weight", 1.0)
            rel.setdefault("description", "")
            rel.setdefault("src_id", key[0])
            rel.setdefault("tgt_id", key[1])
            rel.setdefault("src_tgt", key)
            old = merged.get(key)
            if not old:
                merged[key] = rel
            else:
                if rel.get("hit_sem_count",0) > old.get("hit_sem_count",0):
                    old["hit_sem_count"] = rel["hit_sem_count"]
                if rel.get("rank",0) > old.get("rank",0):
                    old["rank"] = rel["rank"]
                if rel.get("weight",0) > old.get("weight",0):
                    old["weight"] = rel["weight"]
                if len(rel.get("description","")) > len(old.get("description","")):
                    old["description"] = rel.get("description","")
    merged_list = list(merged.values())
    merged_list.sort(
        key=lambda r: (
            -r.get("hit_sem_count",0),
            -r.get("rank",0),
            -r.get("weight",0.0),
            r.get("description","")
        )
    )
    return merged_list

def _make_chunk_key(ck: dict):
    doc_id = ck.get("full_doc_id")
    idx = ck.get("chunk_order_index")
    if doc_id is not None and idx is not None:
        return (doc_id, idx)
    raw = ck.get("content", "")
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return ("hash", h)

def merge_text_units(*lists):
    merged: dict[tuple, dict] = {}
    for lst in lists or []:
        for ck in lst or []:
            key = _make_chunk_key(ck)
            if key[0] == "hash":
                printable_id = f"hash:{key[1]}"
            else:
                printable_id = f"{key[0]}:{key[1]}"
            ck.setdefault("_cid", printable_id)
            order = ck.get("chunk_order_index")
            ck.setdefault("order", order if order is not None else BIG_ORDER)
            ck.setdefault("relation_counts", 0)
            ck.setdefault("content", "")
            old = merged.get(key)
            if not old:
                merged[key] = ck
            else:
                if ck["order"] < old["order"]:
                    old["order"] = ck["order"]
                if ck["relation_counts"] > old["relation_counts"]:
                    old["relation_counts"] = ck["relation_counts"]
                if len(ck["content"]) > len(old["content"]):
                    old["content"] = ck["content"]
    merged_list = list(merged.values())
    merged_list.sort(
        key=lambda x: (x.get("order", BIG_ORDER),
                       -x.get("relation_counts", 0),
                       -len(x.get("content","")),
                       x.get("_cid",""))
    )
    return merged_list

def build_entities_csv(rows):
    table = [["id","entity","type","description","rank"]]
    for i, e in enumerate(rows):
        table.append([i, e["entity_name"], e.get("entity_type","UNKNOWN"), e.get("description",""), e.get("rank",0)])
    return list_of_list_to_csv(table)

def build_relations_csv(rows):
    table = [["id","source","target","description","keywords","weight","rank"]]
    for i, r in enumerate(rows):
        src = r.get("src_id") or (r.get("src_tgt")[0] if r.get("src_tgt") else r.get("src"))
        tgt = r.get("tgt_id") or (r.get("src_tgt")[1] if r.get("src_tgt") else r.get("tgt"))
        table.append([
            i, src, tgt,
            r.get("description",""),
            r.get("keywords",""),
            r.get("weight",1.0),
            r.get("rank",0),
        ])
    return list_of_list_to_csv(table)

def build_sem_units_csv(rows):
    table = [["id","summary","content"]]
    for i, su in enumerate(rows):
        summary = su.get("unit_summary","")
        content = su.get("unit_content","")
        content = " ".join(strip_leading_summary(content, summary).split())
        table.append([i, summary, content])
    return list_of_list_to_csv(table)

def build_text_csv(rows):
    table = [["id","content"]]
    for i, ck in enumerate(rows):
        table.append([i, ck.get("content","")])
    return list_of_list_to_csv(table)

async def ERS_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    semantic_unit_vdb: BaseVectorStorage,
    semantic_units_kv: BaseKVStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    low_level_context = None
    high_level_context = None
    semantic_level_context = None
    ll_sem_units = None
    hl_sem_units = None
    sl_sem_units = None
    ll_entities = None
    hl_entities = None
    sl_entities = None
    ll_relations = None
    hl_relations = None
    sl_relations = None
    ll_text_units = None
    hl_text_units = None
    sl_text_units = None
    if ll_keywords:
        low_level_context, ll_sem_units, ll_entities, ll_relations, ll_text_units = await _build_entity_level_query_context(
            ll_keywords,
            knowledge_graph_inst,
            semantic_units_kv,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    if hl_keywords:
        high_level_context, hl_sem_units, hl_entities, hl_relations, hl_text_units = await _build_relationship_level_query_context(
            hl_keywords,
            knowledge_graph_inst,
            semantic_units_kv,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    kw_prompt_temp = PROMPTS["semantic_unit_cues_from_query"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        su_keywords = keywords_data.get("semantic_unit_cues", [])
        su_keywords = ", ".join(su_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            su_keywords = keywords_data.get("semantic_unit_cues", [])
            su_keywords = ", ".join(su_keywords)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    picked_units_dict = {su["unit_id"]: su for su in (hl_sem_units + ll_sem_units)}
    picked_units = list(picked_units_dict.values())
    skip_entity_names = {e["entity_name"] for e in ll_entities} | {e["entity_name"] for e in hl_entities}
    def norm(a, b):
        return (a, b) if a <= b else (b, a)
    skip_edge_pairs = {norm(r["src_tgt"][0], r["src_tgt"][1]) for r in (ll_relations + hl_relations)}
    skip_chunk_ids: set[str] = {
        ck.get("id") or ck.get("chunk_id")
        for ck in (ll_text_units + hl_text_units)
        if ck is not None
    }
    if su_keywords:
        semantic_level_context, sl_sem_units, sl_entities, sl_relations, sl_text_units = await _build_semantic_unit_level_query_context(
            su_keywords,
            knowledge_graph_inst,
            semantic_unit_vdb,
            semantic_units_kv,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
            picked_units,
            skip_entity_names,
            skip_edge_pairs,
            skip_chunk_ids,
        )
    all_sem_units = merge_semantic_units(hl_sem_units, ll_sem_units, sl_sem_units)
    all_entities_list = merge_entities(hl_entities, ll_entities, sl_entities)
    all_relations_list = merge_relations(ll_relations, hl_relations, sl_relations)
    all_text_units_list = merge_text_units(ll_text_units, hl_text_units, sl_text_units)
    context = f"""
-----Entities-----
```csv
{build_entities_csv(all_entities_list)}
```
-----Relationships-----
```csv
{build_relations_csv(all_relations_list)}
```
-----Semantic-Units-----
```csv
{build_sem_units_csv(all_sem_units)}
```
-----Sources-----
```csv
{build_text_csv(all_text_units_list)}
```
"""
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


































