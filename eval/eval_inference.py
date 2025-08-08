import heapq
from typing import List, Tuple

from llama_index.core.schema import QueryBundle

from app.engine.inference.base_inference import BaseInference
from app.engine.inference.retrievers.hybrid_retriever import HybridRetriever


class EvalInference(BaseInference):
    """
    Base evaluator class for retrieval-only evaluation operations.

    Inherits from BaseInference but focuses on retrieval evaluation
    without LLM generation, useful for benchmark testing.
    """

    def __init__(self, args, questions=None):
        """
        Initialize the evaluator with the same components as BaseInference.

        Args:
            args: Configuration arguments containing API keys, model paths, and other settings.
        """
        super().__init__(args)
        self.questions = questions

    def evaluate_retriever(self, query_string: str, alpha: float = 0.5) -> List[Tuple]:
        """
        Retrieval-only evaluation method for benchmark testing.

        Performs hybrid retrieval and reranking without LLM generation,
        making it suitable for evaluating retrieval performance.

        Args:
            query_string: User's query text
            alpha: Weight for combining dense and sparse scores (0.0 = only sparse, 1.0 = only dense)

        Returns:
            List of tuples with (None, metadata, timespan) - no LLM generation
        """
        results = []
        reranker = self._get_reranker()
        query_bundle = QueryBundle(query_str=query_string)

        # Get all nodes once for efficiency
        all_node_ids = [node.id_ for node in self.index.docstore.docs.values()]
        all_nodes = self.index.docstore.get_nodes(all_node_ids)

        for start_date, end_date in self.timespans:
            # Create custom hybrid retriever with timespan filtering
            hybrid_retriever = HybridRetriever(
                vector_index=self.index,
                nodes=all_nodes,
                similarity_top_k=50,
                alpha=alpha,
                start_date=start_date,
                end_date=end_date,
                callback_manager=self.callback_manager
            )

            # Retrieve using custom retriever
            combined_nodes = hybrid_retriever.retrieve(query_bundle)

            # Apply reranking to top candidates
            top_candidates = combined_nodes[:20]
            reranked_nodes = reranker.postprocess_nodes(top_candidates, query_bundle)

            # Process metadata with scores
            metadata = {}
            for node in reranked_nodes:
                metadata[node.id_] = {
                    'score': node.score,
                    'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'metadata': node.metadata
                }

            # No LLM generation - return None for answer
            results.append((None, metadata, (start_date, end_date)))

        # Keep top 5 nodes based on scores for each result
        for i in range(len(results)):
            metadata = results[i][1]
            if metadata:
                top_nodes = heapq.nlargest(5, metadata.items(), key=lambda item: item[1]['score'])
                top_metadata = {node_id: data for node_id, data in top_nodes}
                results[i] = (results[i][0], top_metadata, results[i][2])

        self._cleanup_resources()
        return results
