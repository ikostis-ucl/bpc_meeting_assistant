import heapq
from typing import List, Tuple

from llama_index.core.schema import QueryBundle

from app.engine.inference.base_inference import BaseInference
from app.engine.inference.retrievers.hybrid_retriever import HybridRetriever
from app.utils.app_utils import pprint_hline
from app.utils.inference_utils import pprint_qa


class EvalInference(BaseInference):
    """
    Base evaluator class for retrieval-only evaluation operations.

    Inherits from BaseInference but focuses on retrieval evaluation
    without LLM generation, useful for benchmark testing.
    """

    def __init__(self, args):
        """
        Initialize the evaluator with the same components as BaseInference.

        Args:
            args: Configuration arguments containing API keys, model paths, and other settings.
        """
        super().__init__(args)
        self.questions = None

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

        # Apply judge-based result cleaning (without LLM answers)
        cleaned_results = self._clean_retrieval_results(results)

        # Keep top 3 nodes based on scores for each result
        for i in range(len(cleaned_results)):
            metadata = cleaned_results[i][1]
            if metadata:
                top_nodes = heapq.nlargest(5, metadata.items(), key=lambda item: item[1]['score'])
                top_metadata = {node_id: data for node_id, data in top_nodes}
                cleaned_results[i] = (cleaned_results[i][0], top_metadata, cleaned_results[i][2])

        self._cleanup_resources()
        return cleaned_results

    def _clean_retrieval_results(self, results: List[Tuple]) -> List[Tuple]:
        """
        Clean retrieval results by merging similar timespans based on metadata overlap.

        Since we don't have LLM answers to compare, we merge based on
        overlapping retrieved documents/metadata.

        Args:
            results: List of (None, metadata, timespan) tuples

        Returns:
            List of cleaned results with merged timespans where appropriate
        """
        if not results:
            return results

        cleaned_results = [results[0]]

        for i in range(1, len(results)):
            current_metadata = results[i][1] or {}
            previous_metadata = cleaned_results[-1][1] or {}

            # Check for metadata overlap (similar retrieved documents)
            overlap_ratio = self._calculate_metadata_overlap(current_metadata, previous_metadata)

            # Merge if significant overlap (threshold can be adjusted)
            if overlap_ratio > 0.3:
                # Merge metadata
                merged_metadata = previous_metadata.copy()
                merged_metadata.update(current_metadata)

                # Extend timespan
                cleaned_results[-1] = (
                    None,  # No LLM answer
                    merged_metadata,
                    (cleaned_results[-1][2][0], results[i][2][1])  # Extend timespan
                )
            else:
                cleaned_results.append(results[i])

        return cleaned_results

    @staticmethod
    def _calculate_metadata_overlap(metadata1: dict, metadata2: dict) -> float:
        """
        Calculate overlap ratio between two metadata dictionaries.

        Args:
            metadata1: First metadata dictionary
            metadata2: Second metadata dictionary

        Returns:
            Float between 0 and 1 representing overlap ratio
        """
        if not metadata1 or not metadata2:
            return 0.0

        keys1 = set(metadata1.keys())
        keys2 = set(metadata2.keys())

        if not keys1 or not keys2:
            return 0.0

        intersection = len(keys1.intersection(keys2))
        union = len(keys1.union(keys2))

        return intersection / union if union > 0 else 0.0

    def run(self):
        """
        Execute the benchmark evaluation workflow for retrieval-only testing.

        Processes queries using hybrid retrieval and reranking without LLM generation,
        making it suitable for evaluating retrieval performance without API costs.
        """
        # Process each question using retrieval-only evaluation
        for question in self.questions:
            results = self.evaluate_retriever(query_string=question)
            # Display results using existing utilities
            pprint_qa(question, results)
            pprint_hline(token="=")
