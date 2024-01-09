import os
from typing import Literal, Sequence

import nest_asyncio  # type: ignore
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    get_response_synthesizer,  # type: ignore
)
from llama_index.embeddings import OpenAIEmbedding
# from llama_index.extractors import (
#     KeywordExtractor,
#     QuestionsAnsweredExtractor,
#     SummaryExtractor,
#     TitleExtractor,
# )
from llama_index.graph_stores import Neo4jGraphStore
from llama_index.indices import KnowledgeGraphIndex, VectorStoreIndex
from llama_index.indices.knowledge_graph.retrievers import KGRetrieverMode
from llama_index.indices.loading import load_index_from_storage  # type: ignore
from llama_index.ingestion import DocstoreStrategy, IngestionCache, IngestionPipeline
from llama_index.ingestion.cache import RedisCache
from llama_index.llms import OpenAI
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import Response
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.schema import BaseNode, Document
# MetadataMode
from llama_index.storage.docstore import RedisDocumentStore
from llama_index.storage.index_store import RedisIndexStore
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.vector_stores.types import VectorStoreQueryMode

from diplodigst.types import DiploIndex
from diplodigst.utils import DiploRetriever

nest_asyncio.apply()  # type: ignore


class DiploDocLoader:
    def __init__(
        self,
        embed_model: OpenAIEmbedding,
        llm_model: OpenAI,
        weaviate_host: str,
        weaviate_port: int,
        redis_host: str,
        redis_port: int,
        neo4j_host: str,
        neo4j_port: int,
        neo4j_username: str,
        neo4j_password: str,
        name: str = "DiploDigst",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the DiploDigst tool.

        Args:
            embed_model (OpenAIEmbedding): The embedding model.
            llm_model (OpenAI): The LLM model.
            weaviate_host (str): The host of the Weaviate service.
            weaviate_port (int): The port of the Weaviate service.
            redis_host (str): The host of the Redis service.
            redis_port (int): The port of the Redis service.
            neo4j_host (str): The host of the Neo4j service.
            neo4j_port (int): The port of the Neo4j service.
            neo4j_username (str): The username for Neo4j authentication.
            neo4j_password (str): The password for Neo4j authentication.
            name (str, optional): The name of the DiploDigst tool. Defaults to "DiploDigst".
            chunk_size (int, optional): The size of each chunk for processing. Defaults to 500.
            chunk_overlap (int, optional): The overlap between chunks. Defaults to 50.
            verbose (bool, optional): Whether to print verbose information. Defaults to True.
        """
        self._name: str = name
        self._vector_store_name: str = f"{self._name}VectorStore"
        self._graph_store_name: str = f"{self._name}GraphStore"
        self._doc_store_name: str = f"{self._name}DocStore"
        self._index_store_name: str = f"{self._name}IndexStore"
        self._ingest_cache_name: str = f"{self._name}IngestCache"
        self._vector_index_id: str = f"{self._name}_vector_index"
        self._graph_index_id: str = f"{self._name}_graph_index"
        self._verbose: bool = verbose

        # Initialize vector store
        if self._verbose:
            print("[ INFO ] Initializing vector store")
        self._load_vector_store(
            weaviate_host=weaviate_host,
            weaviate_port=weaviate_port,
        )
        if self._verbose:
            print("[ INFO ] Vector store initialized")

        # Initialize graph store
        if self._verbose:
            print("[ INFO ] Initializing graph store")
        self._load_graph_store(
            neo4j_host=neo4j_host,
            neo4j_port=neo4j_port,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
        )
        if self._verbose:
            print("[ INFO ] Graph store initialized")

        # Initialize document store
        if self._verbose:
            print("[ INFO ] Initializing document store")
        self._load_document_store(
            redis_host=redis_host,
            redis_port=redis_port,
        )
        if self._verbose:
            print("[ INFO ] Document store initialized")

        # Initialize index store
        if self._verbose:
            print("[ INFO ] Initializing index store")
        self._load_index_store(
            redis_host=redis_host,
            redis_port=redis_port,
        )
        if self._verbose:
            print("[ INFO ] Index store initialized")

        # Initialize ingestion cache
        if self._verbose:
            print("[ INFO ] Initializing ingestion cache")
        self._load_ingestion_cache(
            redis_host=redis_host,
            redis_port=redis_port,
        )
        if self._verbose:
            print("[ INFO ] Ingestion cache initialized")

        # Initialize LLM and embedding models
        if self._verbose:
            print("[ INFO ] Initializing LLM and embedding models")
        self.llm_model: OpenAI = llm_model
        self.embed_model: OpenAIEmbedding = embed_model
        if self._verbose:
            print("[ INFO ] LLM and embedding models initialized")

        # Initialize service contexts
        if self._verbose:
            print("[ INFO ] Initializing service context")
        self._load_service_context(chunk_size, chunk_overlap)
        if self._verbose:
            print("[ INFO ] Service context initialized")

        # Initialize storage contexts
        if self._verbose:
            print("[ INFO ] Initializing storage context")
        self._load_storage_context()
        if self._verbose:
            print("[ INFO ] Storage context initialized")

        # Initialize ingestion pipeline
        if self._verbose:
            print("[ INFO ] Initializing ingestion pipeline")
        self._load_ingest_pipeline()
        if self._verbose:
            print("[ INFO ] Ingestion pipeline initialized")

    def ingest(self, data_path: str, filename_as_id: bool = False) -> DiploIndex:
        """
        Ingests documents from the specified data path and creates indices for vector and knowledge graph.

        Args:
            data_path (str): The path to the directory containing the documents.
            filename_as_id (bool, optional): Whether to use the filename as the ID for each document. Defaults to False.

        Returns:
            DocumentIndex: The document index containing the vector and graph indices.

        Raises:
            ValueError: If there is an error indexing the documents.
        """
        try:
            # load documents
            if self._verbose:
                print("[ INFO ] Loading documents")
            docs: list[Document] = self._load_documents(data_path, filename_as_id)
            if self._verbose:
                print(f"[ INFO ] Loaded {len(docs)} documents")

            # run ingestion pipeline
            if self._verbose:
                print("[ INFO ] Running ingestion pipeline")
            nodes: Sequence[BaseNode] = self._ingest_pipeline.run(documents=docs)
            if self._verbose:
                print(f"[ INFO ] Ingested {len(nodes)} nodes")

            # create vector index
            if self._verbose:
                print("[ INFO ] Creating vector index")
            vector_index: VectorStoreIndex = self._create_vector_index(
                nodes=nodes,
                show_progress=True,
            )
            if self._verbose:
                print("[ INFO ] Vector index created")

            # create knowledge graph index
            if self._verbose:
                print("[ INFO ] Creating knowledge graph index")
            graph_index: KnowledgeGraphIndex = self._create_graph_index(
                nodes=nodes,
                max_triplets_per_chunk=10,
                show_progress=True,
            )
            if self._verbose:
                print("[ INFO ] Knowledge graph index created")

            # print list of indices
            if self._verbose:
                print("[ INFO ] Indices created")

            # return document index
            return DiploIndex(vector=vector_index, graph=graph_index)

        except Exception as e:
            raise ValueError(f"Error indexing documents: {e}")

    def load(self) -> DiploIndex:
        """
        Load the vector index and knowledge graph index, and return a DocumentIndex object.

        Returns:
            DocumentIndex: The document index containing the loaded vector index and graph index.

        Raises:
            ValueError: If there is an error loading the vector index.
        """
        try:
            # load vector index
            print("[ INFO ] Loading vector index")
            vector_index: VectorStoreIndex = self._load_vector_index()
            print("[ INFO ] Vector index loaded")

            # load graph index
            print("[ INFO ] Loading knowledge graph index")
            graph_index: KnowledgeGraphIndex = self._load_graph_index()
            print("[ INFO ] Knowledge graph index loaded")

            # return document index
            return DiploIndex(vector=vector_index, graph=graph_index)
        except Exception as e:
            raise ValueError(f"Error loading vector index: {e}")

    def _load_documents(self, data_path: str, filename_as_id: bool = False) -> list[Document]:
        """
        Load documents from the specified data path.

        Args:
            data_path (str): The path to the directory containing the documents.
            filename_as_id (bool, optional): Whether to use the filename as the document ID. Defaults to False.

        Returns:
            list[Document]: A list of loaded documents.

        Raises:
            ValueError: If the data path does not exist or if there is an error loading the documents.
        """
        # Check that the data path exists
        if not os.path.exists(data_path):
            raise ValueError("Data path does not exist")

        # Load documents
        try:
            return SimpleDirectoryReader(
                data_path, filename_as_id=filename_as_id, required_exts=[".pdf"]
            ).load_data(show_progress=self._verbose)
        except Exception as e:
            raise ValueError(f"Error loading documents: {e}")

    def _load_service_context(self, chunk_size: int, chunk_overlap: int) -> ServiceContext:
        """
        Load the service context with the specified chunk size and chunk overlap.

        Args:
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between adjacent text chunks.

        Returns:
            ServiceContext: The loaded service context.

        Raises:
            ValueError: If there is an error loading the service context.
        """
        try:
            # initialize service context
            self._service_context: ServiceContext = ServiceContext.from_defaults(  # type: ignore
                embed_model=self.embed_model,
                llm=self.llm_model,
                transformations=[
                    # TitleExtractor(llm=self.llm_model, metadata_mode=MetadataMode.EMBED),
                    # KeywordExtractor(llm=self.llm_model, metadata_mode=MetadataMode.EMBED),
                    # SummaryExtractor(
                    #     llm=self.llm_model,
                    #     summaries=["prev", "self", "next"],
                    #     metadata_mode=MetadataMode.EMBED,
                    # ),
                    # QuestionsAnsweredExtractor(
                    #     llm=self.llm_model, metadata_mode=MetadataMode.EMBED
                    # ),
                ],
                text_splitter=SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            )

            # return service context
            return self._service_context
        except Exception as e:
            raise ValueError(f"Error loading service context: {e}")

    def _load_storage_context(self) -> StorageContext:
        """
        Load the storage context with default values.

        Returns:
            The loaded storage context.

        Raises:
            ValueError: If there is an error loading the storage context.
        """
        try:
            # initialize storage context
            self._storage_context: StorageContext = StorageContext.from_defaults(  # type: ignore
                docstore=self._doc_store,
                index_store=self._index_store,
                vector_store=self._vector_store,
                graph_store=self._graph_store,
            )

            # return storage context
            return self._storage_context
        except Exception as e:
            raise ValueError(f"Error loading storage context: {e}")

    def _load_ingest_pipeline(self) -> IngestionPipeline:
        """
        Load and initialize the ingestion pipeline.

        Returns:
            IngestionPipeline: The initialized ingestion pipeline.

        Raises:
            ValueError: If there is an error loading the ingestion pipeline.
        """
        try:
            # initialize ingestion pipeline
            self._ingest_pipeline: IngestionPipeline = IngestionPipeline(
                transformations=self._service_context.transformations,
                cache=self._ingest_cache,
                docstore=self._doc_store,
                docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
            )

            # return ingestion pipeline
            return self._ingest_pipeline
        except Exception as e:
            raise ValueError(f"Error loading ingestion pipeline: {e}")

    def _load_vector_store(self, weaviate_host: str, weaviate_port: int) -> WeaviateVectorStore:
        """
        Load the Weaviate vector store.

        Args:
            weaviate_host (str): The host address of the Weaviate server.
            weaviate_port (int): The port number of the Weaviate server.

        Returns:
            WeaviateVectorStore: The loaded Weaviate vector store.

        Raises:
            ValueError: If there is an error loading the vector store.
        """
        try:
            # initialize vector store variables
            self._vector_store_host: str = weaviate_host
            self._vector_store_port: int = weaviate_port
            self._vector_store_url: str = (
                f"http://{self._vector_store_host}:{self._vector_store_port}"
            )

            # initialize vector store
            self._vector_store = WeaviateVectorStore(
                index_name=self._vector_store_name,
                url=self._vector_store_url,
            )

            # return vector store
            return self._vector_store
        except Exception as e:
            raise ValueError(f"Error loading vector store: {e}")

    def _load_graph_store(
        self,
        neo4j_host: str,
        neo4j_port: int,
        neo4j_username: str,
        neo4j_password: str,
    ) -> Neo4jGraphStore:
        """
        Load the Neo4j graph store with the provided credentials.

        Args:
            neo4j_host (str): The host address of the Neo4j server.
            neo4j_port (int): The port number of the Neo4j server.
            neo4j_username (str): The username for accessing the Neo4j server.
            neo4j_password (str): The password for accessing the Neo4j server.

        Returns:
            Neo4jGraphStore: The initialized Neo4j graph store.

        Raises:
            ValueError: If there is an error loading the graph store.
        """
        try:
            # initialize graph store variables
            self._graph_store_host: str = neo4j_host
            self._graph_store_port: int = neo4j_port
            self._graph_store_url: str = f"bolt://{self._graph_store_host}:{self._graph_store_port}"
            self._graph_store_username: str = neo4j_username
            self._graph_store_password: str = neo4j_password

            # initialize graph store
            self._graph_store = Neo4jGraphStore(
                username=self._graph_store_username,
                password=self._graph_store_password,
                url=self._graph_store_url,
                database=self._graph_store_name,
            )

            # return graph store
            return self._graph_store
        except Exception as e:
            raise ValueError(f"Error loading graph store: {e}")

    def _load_document_store(self, redis_host: str, redis_port: int) -> RedisDocumentStore:
        """
        Load the RedisDocumentStore with the specified Redis host and port.

        Args:
            redis_host (str): The host address of the Redis server.
            redis_port (int): The port number of the Redis server.

        Returns:
            RedisDocumentStore: The loaded RedisDocumentStore object.

        Raises:
            ValueError: If there is an error loading the document store.
        """
        try:
            # initialize document store variables
            self._doc_store_host: str = redis_host
            self._doc_store_port: int = redis_port
            self._doc_store_url: str = f"redis://{self._doc_store_host}:{self._doc_store_port}"

            # initialize document store
            self._doc_store: RedisDocumentStore = RedisDocumentStore.from_host_and_port(
                host=self._doc_store_host,
                port=self._doc_store_port,
                namespace=self._doc_store_name,
            )

            # return document store
            return self._doc_store
        except Exception as e:
            raise ValueError(f"Error loading document store: {e}")

    def _load_index_store(self, redis_host: str, redis_port: int) -> RedisIndexStore:
        """
        Load the RedisIndexStore with the specified Redis host and port.

        Args:
            redis_host (str): The host address of the Redis server.
            redis_port (int): The port number of the Redis server.

        Returns:
            RedisIndexStore: The loaded RedisIndexStore object.

        Raises:
            ValueError: If there is an error loading the index store.
        """
        try:
            # initialize index store variables
            self._index_store_host: str = redis_host
            self._index_store_port: int = redis_port
            self._index_store_url: str = (
                f"redis://{self._index_store_host}:{self._index_store_port}"
            )

            # initialize index store
            self._index_store: RedisIndexStore = RedisIndexStore.from_host_and_port(
                host=self._index_store_host,
                port=self._index_store_port,
                namespace=self._index_store_name,
            )

            # return index store
            return self._index_store
        except Exception as e:
            raise ValueError(f"Error loading index store: {e}")

    def _load_ingestion_cache(self, redis_host: str, redis_port: int) -> IngestionCache:
        """
        Load the ingestion cache with the specified Redis host and port.

        Args:
            redis_host (str): The host address of the Redis server.
            redis_port (int): The port number of the Redis server.

        Returns:
            IngestionCache: The loaded ingestion cache.

        Raises:
            ValueError: If there is an error loading the ingestion cache.
        """
        try:
            # initialize ingestion cache variables
            self._ingest_cache_host: str = redis_host
            self._ingest_cache_port: int = redis_port
            self._ingest_cache_url: str = (
                f"redis://{self._ingest_cache_host}:{self._ingest_cache_port}"
            )

            # initialize ingestion cache
            self._ingest_cache = IngestionCache(
                cache=RedisCache.from_host_and_port(
                    host=self._ingest_cache_host,
                    port=self._ingest_cache_port,
                ),
                collection=self._ingest_cache_name,
            )

            # return vector cache
            return self._ingest_cache
        except Exception as e:
            raise ValueError(f"Error loading vector cache: {e}")

    def _create_vector_index(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = True,
    ) -> VectorStoreIndex:
        """
        Creates a vector index for the given nodes.

        Args:
            nodes (Sequence[BaseNode]): The nodes to be indexed.
            show_progress (bool, optional): Whether to show progress during indexing. Defaults to True.

        Returns:
            VectorStoreIndex: The created vector index.

        Raises:
            ValueError: If there is an error creating the vector index.
        """
        try:
            # initialize vector index
            vector_index: VectorStoreIndex = VectorStoreIndex(
                nodes=nodes,
                service_context=self._service_context,
                storage_context=self._storage_context,
                show_progress=show_progress,
            )

            # set index id
            vector_index.set_index_id(self._vector_index_id)

            # add index struct to storage
            self._storage_context.index_store.add_index_struct(vector_index.index_struct)

            # return vector index
            return vector_index
        except Exception as e:
            raise ValueError(f"Error creating vector index: {e}")

    def _create_graph_index(
        self,
        nodes: Sequence[BaseNode],
        max_triplets_per_chunk: int = 10,
        show_progress: bool = True,
    ) -> KnowledgeGraphIndex:
        """
        Creates a knowledge graph index.

        Args:
            nodes (Sequence[BaseNode] | None, optional): The nodes to be indexed. Defaults to None.
            max_triplets_per_chunk (int, optional): The maximum number of triplets per chunk. Defaults to 10.
            show_progress (bool, optional): Whether to show progress during indexing. Defaults to True.

        Returns:
            KnowledgeGraphIndex: The created knowledge graph index.

        Raises:
            ValueError: If there is an error creating the graph index.
        """
        try:
            # initialize graph index
            graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex(
                nodes=nodes,
                max_triplets_per_chunk=max_triplets_per_chunk,
                service_context=self._service_context,
                storage_context=self._storage_context,
                show_progress=show_progress,
            )

            # set index id
            graph_index.set_index_id(self._graph_index_id)

            # add index struct to storage
            self._storage_context.index_store.add_index_struct(graph_index.index_struct)

            # return graph index
            return graph_index
        except Exception as e:
            raise ValueError(f"Error creating graph index: {e}")

    def _load_vector_index(self) -> VectorStoreIndex:
        """
        Load the vector index from storage.

        Returns:
            VectorStoreIndex: The loaded vector index.

        Raises:
            ValueError: If there is an error loading the vector index.
        """
        try:
            # initialize vector index
            vector_index: VectorStoreIndex = load_index_from_storage(  # type: ignore
                storage_context=self._storage_context,
                index_id=self._vector_index_id,
            )

            # return vector index
            return vector_index
        except Exception as e:
            raise ValueError(f"Error loading vector index: {e}")

    def _load_graph_index(self) -> KnowledgeGraphIndex:
        """
        Load the knowledge graph index from storage.

        Returns:
            The loaded knowledge graph index.

        Raises:
            ValueError: If there is an error loading the graph index.
        """
        try:
            # initialize graph index
            graph_index: KnowledgeGraphIndex = load_index_from_storage(  # type: ignore
                storage_context=self._storage_context,
                index_id=self._graph_index_id,
            )

            # return graph index
            return graph_index
        except Exception as e:
            raise ValueError(f"Error loading graph index: {e}")


class DiploDocRetriever:
    def __init__(
        self,
        llm_model: OpenAI,
        embed_model: OpenAIEmbedding,
        doc_index: DiploIndex,
        retriever_mode: Literal["OR", "AND"] = "OR",
        response_mode: ResponseMode = ResponseMode.COMPACT,
        vector_query_alpha: float | None = 0.5,
        vector_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.HYBRID,
        graph_query_mode: KGRetrieverMode = KGRetrieverMode.HYBRID,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the Tool object.

        Args:
            llm_model (OpenAI): The LLM model.
            embed_model (OpenAIEmbedding): The embedding model.
            doc_index (DocumentIndex): The document index.
            retriever_mode (Literal["OR", "AND"], optional): The retriever mode. Defaults to "OR".
            response_mode (ResponseMode, optional): The response mode. Defaults to ResponseMode.COMPACT.
            vector_query_alpha (float | None, optional): The vector query alpha value. Defaults to 0.5.
            vector_query_mode (VectorStoreQueryMode, optional): The vector query mode. Defaults to VectorStoreQueryMode.HYBRID.
            graph_query_mode (KGRetrieverMode, optional): The graph query mode. Defaults to KGRetrieverMode.HYBRID.
            verbose (bool, optional): Whether to print verbose information. Defaults to True.
        """
        self._retriever_mode: Literal["OR", "AND"] = retriever_mode
        self._response_mode: ResponseMode = response_mode
        self._vector_query_alpha: float | None = vector_query_alpha
        self._vector_query_mode: VectorStoreQueryMode = vector_query_mode
        self._graph_query_mode: KGRetrieverMode = graph_query_mode
        self._verbose: bool = verbose

        # initialize vector and graph indices
        if self._verbose:
            print("[ INFO ] Initializing vector and graph indices")
        self._vector_index: VectorStoreIndex = doc_index.vector
        self._graph_index: KnowledgeGraphIndex = doc_index.graph
        if self._verbose:
            print("[ INFO ] Vector and graph indices initialized")

        # initialize vector and graph retrievers
        if self._verbose:
            print("[ INFO ] Initializing vector and graph retrievers")
        self._load_vector_retriever()
        self._load_graph_retriever()
        if self._verbose:
            print("[ INFO ] Vector and graph retrievers initialized")

        # initialize retriever
        if self._verbose:
            print("[ INFO ] Initializing retriever")
        self._load_retriever()
        if self._verbose:
            print("[ INFO ] Retriever initialized")

        # initialize llm and embedding models
        self.llm_model: OpenAI = llm_model
        self.embed_model: OpenAIEmbedding = embed_model

        # initialize service context
        self._load_service_context()

        # initialize response synthesizer
        self._load_response_synthesizer()

        # initialize query engine
        if self._verbose:
            print("[ INFO ] Initializing query engine")
        self._load_query_engine()
        if self._verbose:
            print("[ INFO ] Query engine initialized")

    def query(self, query: str) -> Response:
        """
        Executes a query on the query engine and returns the response.

        Args:
            query (str): The query to be executed.

        Returns:
            Response: The response from the query engine.

        Raises:
            ValueError: If there is an error retrieving documents.
        """
        try:
            # query the query engine
            _response: Response = self._query_engine.query(query)  # type: ignore

            # return response
            return _response
        except Exception as e:
            raise ValueError(f"Error retrieving documents: {e}")

    def _load_retriever(self) -> DiploRetriever:
        """
        Load the DiploRetriever object.

        Returns:
            DiploRetriever: The loaded DiploRetriever object.

        Raises:
            ValueError: If there is an error loading the retriever.
        """
        try:
            # initialize retriever
            self._retriever = DiploRetriever(
                vector_retriever=self._vector_retriever,
                graph_retriever=self._graph_retriever,
                mode=self._retriever_mode,
            )

            # return retriever
            return self._retriever
        except Exception as e:
            raise ValueError(f"Error loading retriever: {e}")

    def _load_service_context(self) -> ServiceContext:
        """
        Load the service context.

        This method initializes the service context with default values for the embed model and llm model.
        It returns the initialized service context.

        Raises:
            ValueError: If there is an error loading the service context.

        Returns:
            ServiceContext: The initialized service context.
        """
        try:
            # initialize service context
            self._service_context: ServiceContext = ServiceContext.from_defaults(  # type: ignore
                embed_model=self.embed_model,
                llm=self.llm_model,
            )

            # return service context
            return self._service_context
        except Exception as e:
            raise ValueError(f"Error loading service context: {e}")

    def _load_response_synthesizer(self) -> BaseSynthesizer:
        """
        Load and initialize the response synthesizer.

        Returns:
            The initialized response synthesizer.

        Raises:
            ValueError: If there is an error loading the response synthesizer.
        """
        try:
            # initialize response synthesizer
            self._response_synthesizer: BaseSynthesizer = get_response_synthesizer(
                service_context=self._service_context,
                response_mode=self._response_mode,
            )

            # return response synthesizer
            return self._response_synthesizer
        except Exception as e:
            raise ValueError(f"Error loading response synthesizer: {e}")

    def _load_query_engine(self) -> RetrieverQueryEngine:
        """
        Load the query engine.

        Returns:
            RetrieverQueryEngine: The loaded query engine.

        Raises:
            ValueError: If there is an error loading the query engine.
        """
        try:
            # initialize query engine
            self._query_engine: RetrieverQueryEngine = RetrieverQueryEngine(  # type: ignore
                retriever=self._retriever,
                response_synthesizer=self._response_synthesizer,
            )

            # return query engine
            return self._query_engine
        except Exception as e:
            raise ValueError(f"Error loading query engine: {e}")

    def _load_vector_retriever(self) -> VectorIndexRetriever:
        """
        Load the vector retriever.

        Returns:
            VectorIndexRetriever: The loaded vector retriever.

        Raises:
            ValueError: If there is an error loading the vector retriever.
        """
        try:
            # initialize vector retriever
            self._vector_retriever = VectorIndexRetriever(  # type: ignore
                index=self._vector_index,
                alpha=self._vector_query_alpha,
                vector_store_query_mode=self._vector_query_mode,
                include_text=False,
            )

            # return vector retriever
            return self._vector_retriever
        except Exception as e:
            raise ValueError(f"Error loading vector retriever: {e}")

    def _load_graph_retriever(self) -> KGTableRetriever:
        """
        Load the graph retriever.

        Returns:
            KGTableRetriever: The loaded graph retriever.

        Raises:
            ValueError: If there is an error loading the graph retriever.
        """
        try:
            # initialize graph retriever
            self._graph_retriever = KGTableRetriever(  # type: ignore
                index=self._graph_index, retriever_mode=self._graph_query_mode, include_text=False
            )

            # return graph retriever
            return self._graph_retriever
        except Exception as e:
            raise ValueError(f"Error loading graph retriever: {e}")
