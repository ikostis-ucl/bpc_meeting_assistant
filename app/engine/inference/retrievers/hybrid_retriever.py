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
            similarity_top_k: int = None,
            alpha: float = 0.5,
            rrf_k: int = 60,
            callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """
        Initialize the hybrid French retriever.

        Args:
            vector_index: Vector index for dense retrieval
            nodes: List of nodes for BM25 retrieval (pre-filtered by document batch)
            similarity_top_k: Number of top results to retrieve from each method (defaults to 50 if None)
            alpha: Weight for combining scores (1.0 = only dense, 0.0 = only sparse)
            rrf_k: RRF parameter for rank fusion
            callback_manager: Callback manager for tracing
        """
        self._vector_index = vector_index
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k if similarity_top_k is not None else 50
        self._alpha = alpha
        self._rrf_k = rrf_k

        super().__init__(callback_manager)

    def _create_metadata_filters(self, timestamp_batch: List) -> Optional[MetadataFilters]:
        """Create metadata filters for timestamp batch filtering."""
        if not timestamp_batch:
            return None

        filters = [
            MetadataFilter(
                key="meeting_datetime",
                value=timestamp_batch,
                operator=FilterOperator.IN
            )
        ]
        return MetadataFilters(filters=filters)

    @staticmethod
    def preprocessing_fr(text: str) -> str:
        """Enhanced text preprocessing optimized for French text."""
        text = text.lower()
        text = unicodedata.normalize('NFD', text)

        # Remove punctuation but keep French accents and hyphens
        text = re.sub(r'[^\w\sàâäæéèêëïîôöùûüÿçñ-]', ' ', text)

        # Handle French contractions and common patterns
        text = re.sub(r"\bl'", ' le ', text)
        text = re.sub(r"\bd'", ' de ', text)
        text = re.sub(r"\bqu'", ' que ', text)
        text = re.sub(r"\bc'", ' ce ', text)
        text = re.sub(r"\bs'", ' se ', text)
        text = re.sub(r"\bm'", ' me ', text)
        text = re.sub(r"\bt'", ' te ', text)
        text = re.sub(r"\bn'", ' ne ', text)

        text = ' '.join(text.split())

        return text

    def _bm25_retriever(self, nodes: List) -> BM25Retriever:
        """Create BM25 retriever with enhanced French preprocessing."""
        processed_nodes = []
        for node in nodes:
            processed_node = node.copy()
            processed_node.text = self.preprocessing_fr(node.text)
            processed_nodes.append(processed_node)

        effective_top_k = min(self._similarity_top_k, len(processed_nodes))

        return BM25Retriever.from_defaults(
            nodes=processed_nodes,
            similarity_top_k=effective_top_k
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
        dense_ranks = {node.id_: i + 1 for i, node in enumerate(dense_nodes)}
        sparse_ranks = {node.id_: i + 1 for i, node in enumerate(sparse_nodes)}

        all_node_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Calculate RRF scores
        rrf_scores = {}
        for node_id in all_node_ids:
            dense_rrf = 1 / (self._rrf_k + dense_ranks.get(node_id, len(dense_nodes) + 1))
            sparse_rrf = 1 / (self._rrf_k + sparse_ranks.get(node_id, len(sparse_nodes) + 1))
            rrf_scores[node_id] = self._alpha * dense_rrf + (1 - self._alpha) * sparse_rrf

        node_dict = {}
        for node in dense_nodes + sparse_nodes:
            if node.id_ not in node_dict:
                node_dict[node.id_] = node

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
        Retrieve nodes using hybrid approach with document batch filtering.

        Args:
            query_bundle: Query bundle containing the search query.

        Returns:
            List[NodeWithScore]: Retrieved and ranked nodes.
        """

        query_str = query_bundle.query_str
        timestamp_batch = getattr(query_bundle, 'query_batch', [])
        metadata_filters = self._create_metadata_filters(timestamp_batch)

        vector_retriever = VectorIndexRetriever(
            index=self._vector_index,
            similarity_top_k=self._similarity_top_k,
            filters=metadata_filters
        )

        bm25_retriever = self._bm25_retriever(self._nodes)

        dense_nodes = vector_retriever.retrieve(query_str)
        processed_query = self.preprocessing_fr(query_str)
        sparse_nodes = bm25_retriever.retrieve(processed_query)

        combined_nodes = self._combine_retrievers_rrf(dense_nodes, sparse_nodes)
        return combined_nodes
