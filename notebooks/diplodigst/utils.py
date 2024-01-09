from typing import Literal

import tiktoken
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever, KGTableRetriever, VectorIndexRetriever
from llama_index.schema import NodeWithScore
from tiktoken import Encoding


def count_tokens(
    string: str, encoding_name: str = "cl100k_base", model_name: str | None = None
) -> int:
    """
    Counts the number of tokens in a given string.

    Args:
        string (str): The input string to count tokens from.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base".
        model_name (str | None, optional): The name of the model to use. Defaults to None.

    Returns:
        int: The number of tokens in the input string.

    Raises:
        ValueError: If there is an error counting tokens.
    """
    encoding: Encoding | None = None

    try:
        # check if model and encoding are passed
        if model_name:
            encoding = tiktoken.encoding_for_model(model_name)
        elif encoding_name:
            encoding = tiktoken.get_encoding(encoding_name)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        # count tokens
        num_tokens: int = len(encoding.encode(string))

        return num_tokens
    except Exception as e:
        raise ValueError(f"Error counting tokens: {e}")


def tokenizer(
    string: str,
    encoding_name: str = "cl100k_base",
    model_name: str | None = None,
    return_int: bool = False,
) -> list[str] | list[int]:
    """
    Tokenizes a given string.

    Args:
        string (str): The input string to tokenize.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base".
        model_name (str | None, optional): The name of the model to use. Defaults to None.

    Returns:
        list[str]: The tokens of the input string.

    Raises:
        ValueError: If there is an error tokenizing.
    """
    encoding: Encoding | None = None

    try:
        # check if model and encoding are passed
        if model_name:
            encoding = tiktoken.encoding_for_model(model_name)
        elif encoding_name:
            encoding = tiktoken.get_encoding(encoding_name)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        # tokenize
        tokens_int: list[int] = encoding.encode(string)

        if return_int:
            return tokens_int
        else:
            tokens_str: list[str] = [
                encoding.decode_single_token_bytes(token).decode(encoding="utf-8")
                for token in tokens_int
            ]

            return tokens_str
    except Exception as e:
        raise ValueError(f"Error tokenizing: {e}")


class DiploRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        graph_retriever: KGTableRetriever,
        mode: Literal["OR", "AND"] = "OR",
    ) -> None:
        """
        Initialize the Tool object.

        Args:
            vector_retriever (VectorIndexRetriever): The vector retriever object.
            graph_retriever (KGTableRetriever): The graph retriever object.
            mode (Literal["OR", "AND"], optional): The mode of operation. Defaults to "OR".

        Raises:
            ValueError: If an invalid mode is provided.

        Returns:
            None
        """
        self._vector_retriever: VectorIndexRetriever = vector_retriever
        self._graph_retriever: KGTableRetriever = graph_retriever

        # check if mode is valid
        if mode not in ["OR", "AND"]:
            raise ValueError(f"Invalid mode: {mode}")

        # set mode
        self._mode: Literal["OR", "AND"] = mode

        # initialize base class
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """
        Retrieve nodes based on the given query bundle.

        Args:
            query_bundle (QueryBundle): The query bundle containing the necessary information for retrieval.

        Returns:
            list[NodeWithScore]: A list of retrieved nodes with their scores.
        """
        # retrieve nodes from vector and kg retrievers
        vector_nodes: list[NodeWithScore] = self._vector_retriever.retrieve(query_bundle)
        graph_nodes: list[NodeWithScore] = self._graph_retriever.retrieve(query_bundle)

        # get ids of nodes
        vector_ids: set[str] = {n.node.node_id for n in vector_nodes}
        graph_ids: set[str] = {n.node.node_id for n in graph_nodes}

        # combine nodes into a single dict
        combined_dict: dict[str, NodeWithScore] = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in graph_nodes})

        # get ids of nodes to retrieve
        if self._mode == "AND":
            retrieve_ids: set[str] = vector_ids.intersection(graph_ids)
        else:
            retrieve_ids = vector_ids.union(graph_ids)

        # retrieve nodes
        retrieve_nodes: list[NodeWithScore] = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes
