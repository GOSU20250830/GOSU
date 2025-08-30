from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .utils import EmbeddingFunc
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)
T = TypeVar("T")
@dataclass
class QueryParam:
    mode: Literal["ERS"] = "ERS"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    top_k: int = 60
    top_m: int = 30
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000
    max_token_for_semantic_unit: int = 4000
    max_token_for_semantic_context: int = 4000
    sr_top_k: int = 12
    sr_sim_threshold: float = 0.70
    sr_hit_count_threshold: int = 3
    sr_max_round: int = 2
@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict
    async def index_done_callback(self):
        pass
    async def query_done_callback(self):
        pass
@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)
    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError
    async def upsert(self, data: dict[str, dict]):
        raise NotImplementedError
@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc
    async def all_keys(self) -> list[str]:
        raise NotImplementedError
    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError
    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError
    async def filter_keys(self, data: list[str]) -> set[str]:
        raise NotImplementedError
    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError
    async def drop(self):
        raise NotImplementedError
@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError
    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError
    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError
    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError
    async def delete_node(self, node_id: str):
        raise NotImplementedError
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in GOSU.")
@dataclass
class BaseBipartiteGraphStorage(StorageNameSpace, ABC):
    embedding_func: Optional[EmbeddingFunc] = None
    @abstractmethod
    def upsert_hyperedge_node(self, hyperedge_id: str, data: dict):
        pass
    @abstractmethod
    def upsert_entity(self, entity_id: str, data: dict):
        pass
    @abstractmethod
    def connect_entity_to_hyperedge(self, entity_id: str, hyperedge_id: str, edge_data: Optional[dict] = None):
        pass
    @abstractmethod
    def get_entities_of_hyperedge(self, hyperedge_id: str) -> list[str]:
        pass
    @abstractmethod
    def get_hyperedges_of_entity(self, entity_id: str) -> list[str]:
        pass
    @abstractmethod
    def get_node(self, node_id: str) -> Union[dict, None]:
        pass
    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        pass
    @abstractmethod
    def delete_node(self, node_id: str):
        pass
    @abstractmethod
    def save(self):
        pass