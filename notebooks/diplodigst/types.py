from typing import Any

from llama_index.indices import KnowledgeGraphIndex, VectorStoreIndex
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict


class DiploIndex(BaseModel):
    """
    Represents an index for documents.

    Attributes:
        vector (VectorStoreIndex): The vector store index.
        graph (KnowledgeGraphIndex): The knowledge graph index.
        model_config (Any): The model configuration.
    """

    vector: VectorStoreIndex
    graph: KnowledgeGraphIndex

    model_config: Any = SettingsConfigDict(arbitrary_types_allowed=True)
