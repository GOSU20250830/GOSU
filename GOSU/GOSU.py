import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple
from openai import AsyncOpenAI
from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
    with_retry,
    openai_complete_if_cache,
)
from .operate import (
    extract_entities_directly,
    chunking_by_token_size,
    similar_semunit_unit,
    extract_semantic_units,
    extract_entities_by_semantic_units,
    merge_similar_semantic_units,
    get_target_kg_single_semunit,
    ERS_query,
)
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    TextChunkSchema,
)
from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from .kg.neo4j_impl import Neo4JStorage
from .kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
@dataclass
class GOSU:
    working_dir: str = field(
        default_factory=lambda: f"./GOSU_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "Selected Generation Model Name"
    entity_extract_max_gleaning: int = 1
    semantic_unit_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    llm_model_func: callable = gpt_4o_mini_complete
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "GOSU.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GOSU init with param:\n  {_print_config}\n")
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.semantic_unit_vdb = self.vector_db_storage_cls(
            namespace="semantic_units",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"unit_summary", "source_chunk_id"},
        )
        self.semantic_units_kv = self.key_string_value_json_storage_cls(
            namespace="semantic_units_kv",
            global_config=asdict(self),
            embedding_func=None,
        )
        _raw_llm = partial(
            self.llm_model_func,
            hashing_kv=self.llm_response_cache,
            **self.llm_model_kwargs,
        )
        _retry_llm = with_retry(_raw_llm, max_attempt=3)
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(_retry_llm)
    def _get_storage_class(self) -> dict[str, Type[BaseGraphStorage]]:
        return {
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
        }
    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))
    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage.")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs.")
            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage.")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks.")
            await self.chunks_vdb.upsert(inserting_chunks)
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities_directly(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found.")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            logger.info("[Entity Extraction by semantic unit]...")
            su_dict = await self.aextract_semantic_units(chunk_keys=list(inserting_chunks.keys()))
            if not su_dict:
                logger.warning("No new semantic units – skip entity extraction.")
            else:
                for su_id, su in su_dict.items():
                    summary_txt = su["unit_summary"]
                    chunk_ids = [cid.strip() for cid in su["source_chunk_id"].split(";;;") if cid.strip()]
                    chunk_vals = await self.text_chunks.get_by_ids(chunk_ids)
                    tmp_chunks: dict[str, dict] = {}
                    for cid, cv in zip(chunk_ids, chunk_vals):
                        if cv and "content" in cv:
                            tmp_chunks[cid] = {
                                "content": cv["content"],
                                "full_doc_id": su["full_doc_id"]
                            }
                    maybe_new_kg = await extract_entities_by_semantic_units(
                        su_id,
                        summary_txt,
                        tmp_chunks,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        relationships_vdb=self.relationships_vdb,
                        global_config=asdict(self),
                    )
                    if maybe_new_kg is not None:
                        self.chunk_entity_relation_graph = maybe_new_kg
                logger.info("[Entity Extraction] all semantic units processed.")
                update_storage = True
        except Exception as e:
            logger.error(f"[ainsert] failed: {e}")
            raise
        if update_storage:
            await self._insert_done()
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.semantic_unit_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    async def _insert_semantic_unit_done(self):
        tasks = []
        for storage_inst in [
            self.semantic_unit_vdb,
            self.semantic_units_kv,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    async def aextract_semantic_units(
            self,
            chunk_keys: list[str] | None = None,
    ) -> dict[str, dict] | None:
        if chunk_keys is None:
            keys = await self.text_chunks.all_keys()
        else:
            keys = chunk_keys
        if not keys:
            logger.warning("[Semantic Unit Extraction] No chunks found.")
            return None
        chunk_vals = await self.text_chunks.get_by_ids(keys)
        chunks = {
            k: v for k, v in zip(keys, chunk_vals) if v is not None
        }
        if not chunks:
            logger.warning("[Semantic Unit Extraction] Selected chunks not found.")
            return None
        su_dict = await extract_semantic_units(
            chunks=chunks,
            semantic_unit_vdb=self.semantic_unit_vdb,
            global_config=asdict(self),
        )
        if su_dict is None:
            return None
        final_pairs = await similar_semunit_unit(
            self,
            su_dict,
            coarse_threshold=0.70,
        )
        merged_docs = merge_similar_semantic_units(su_dict, final_pairs)
        for uid in list(merged_docs.keys()):
            _, _ = await get_target_kg_single_semunit(
                self,
                merged_docs,
                uid,
                self.chunks_vdb,
                self.text_chunks,
                top_k=5,
                threshold=0.70,
            )
        await self.semantic_unit_vdb.upsert(merged_docs)
        await self.semantic_units_kv.upsert(merged_docs)
        await self._insert_semantic_unit_done()
        return merged_docs
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "ERS":
            response = await ERS_query(
                query,
                self.chunk_entity_relation_graph,
                self.semantic_unit_vdb,
                self.semantic_units_kv,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response
    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))
    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'
        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)
            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")
    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)







