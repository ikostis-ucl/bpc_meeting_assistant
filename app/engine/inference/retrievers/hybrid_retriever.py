import re
import unicodedata
from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.retrievers.bm25 import BM25Retriever


class HybridRetriever(BaseRetriever):
    """
    Custom hybrid retriever combining dense vector search and enhanced BM25 for French text.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both retrievers,
    with enhanced French text preprocessing for BM25 and timespan filtering.
    """

    def __init__(
            self,
            vector_index,
            nodes: List,
            similarity_top_k: int = 50,
            alpha: float = 0.5,
            rrf_k: int = 60,
            start_date: Optional[int] = None,
            end_date: Optional[int] = None,
            callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """
        Initialize the hybrid French retriever.

        Args:
            vector_index: Vector index for dense retrieval
            nodes: List of nodes for BM25 retrieval (will be filtered if dates provided)
            similarity_top_k: Number of top results to retrieve from each method
            alpha: Weight for combining scores (1.0 = only dense, 0.0 = only sparse)
            rrf_k: RRF parameter for rank fusion
            start_date: Start timestamp for timespan filtering
            end_date: End timestamp for timespan filtering
            callback_manager: Callback manager for tracing
        """
        self._vector_index = vector_index
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._start_date = start_date
        self._end_date = end_date

        # Filter nodes by timespan if dates are provided
        if self._start_date is not None and self._end_date is not None:
            self._nodes = self._filter_nodes_by_timespan(nodes)

        super().__init__(callback_manager)

    def _filter_nodes_by_timespan(self, nodes: List) -> List:
        """Filter nodes based on meeting datetime within the specified timespan."""
        filtered_nodes = []
        for node in nodes:
            if node.metadata.get("meeting_datetime"):
                meeting_time = node.metadata["meeting_datetime"]
                if self._start_date <= meeting_time <= self._end_date:
                    filtered_nodes.append(node)
        return filtered_nodes

    def _create_metadata_filters(self) -> Optional[MetadataFilters]:
        """Create metadata filters for timespan filtering."""
        if self._start_date is None or self._end_date is None:
            return None

        filters = [
            MetadataFilter(
                key="meeting_datetime",
                value=self._start_date,
                operator=FilterOperator.GTE
            ),
            MetadataFilter(
                key="meeting_datetime",
                value=self._end_date,
                operator=FilterOperator.LTE
            )
        ]
        return MetadataFilters(filters=filters)

    @staticmethod
    def preprocessing_fr(text: str) -> str:
        """Enhanced text preprocessing optimized for French text."""
        # Convert to lowercase
        text = text.lower()

        # Normalize Unicode characters (NFD normalization for French accents)
        text = unicodedata.normalize('NFD', text)

        # Remove punctuation but keep French accents and hyphens
        text = re.sub(r'[^\w\sàâäæéèêëïîôöùûüÿçñ-]', ' ', text)

        # Handle French contractions and common patterns
        text = re.sub(r"\bl'", ' le ', text)  # l'école -> le école
        text = re.sub(r"\bd'", ' de ', text)  # d'accord -> de accord
        text = re.sub(r"\bqu'", ' que ', text)  # qu'est -> que est
        text = re.sub(r"\bc'", ' ce ', text)  # c'est -> ce est
        text = re.sub(r"\bs'", ' se ', text)  # s'agit -> se agit
        text = re.sub(r"\bm'", ' me ', text)  # m'aide -> me aide
        text = re.sub(r"\bt'", ' te ', text)  # t'aide -> te aide
        text = re.sub(r"\bn'", ' ne ', text)  # n'est -> ne est

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _bm25_retriever(self, nodes: List) -> BM25Retriever:
        """Create BM25 retriever with enhanced French preprocessing."""
        # Create copies of nodes to avoid modifying originals
        processed_nodes = []
        for node in nodes:
            processed_node = node.copy()
            processed_node.text = self.preprocessing_fr(node.text)
            processed_nodes.append(processed_node)

        return BM25Retriever.from_defaults(
            nodes=processed_nodes,
            similarity_top_k=self._similarity_top_k
        )

    def _combine_retrievers_rrf(
            self,
            dense_nodes: List[NodeWithScore],
            sparse_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        Args:
            dense_nodes: Nodes from vector retrieval.
            sparse_nodes: Nodes from BM25 retrieval.

        Returns:
            List[NodeWithScore]: Combined and ranked nodes.
        """
        # Create rank dictionaries
        dense_ranks = {node.id_: i + 1 for i, node in enumerate(dense_nodes)}
        sparse_ranks = {node.id_: i + 1 for i, node in enumerate(sparse_nodes)}

        # Get all unique node IDs
        all_node_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Calculate RRF scores
        rrf_scores = {}
        for node_id in all_node_ids:
            dense_rrf = 1 / (self._rrf_k + dense_ranks.get(node_id, len(dense_nodes) + 1))
            sparse_rrf = 1 / (self._rrf_k + sparse_ranks.get(node_id, len(sparse_nodes) + 1))
            rrf_scores[node_id] = self._alpha * dense_rrf + (1 - self._alpha) * sparse_rrf

        # Create node dictionary for lookup
        node_dict = {}
        for node in dense_nodes + sparse_nodes:
            if node.id_ not in node_dict:
                node_dict[node.id_] = node

        # Create result nodes with RRF scores
        result_nodes = []
        for node_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if node_id in node_dict:
                node = node_dict[node_id]
                node.score = score
                result_nodes.append(node)

        return result_nodes

    @trace_method("HybridRetriever.retrieve")
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid approach with timespan filtering.

        Args:
            query_bundle: Query bundle containing the search query.

        Returns:
            List[NodeWithScore]: Retrieved and ranked nodes.
        """
        query_str = query_bundle.query_str

        # Create vector retriever with metadata filters
        metadata_filters = self._create_metadata_filters()
        vector_retriever = VectorIndexRetriever(
            index=self._vector_index,
            similarity_top_k=self._similarity_top_k,
            filters=metadata_filters
        )

        # Create enhanced BM25 retriever (already filtered nodes in __init__)
        bm25_retriever = self._bm25_retriever(self._nodes)

        # Retrieve from both methods
        dense_nodes = vector_retriever.retrieve(query_str)  # Original query for embeddings
        processed_query = self.preprocessing_fr(query_str)
        sparse_nodes = bm25_retriever.retrieve(processed_query)  # Processed query for BM25

        # Combine using RRF
        combined_nodes = self._combine_retrievers_rrf(dense_nodes, sparse_nodes)

        return combined_nodes
