from typing import List
import gc
import heapq
from abc import ABC, abstractmethod
from datetime import datetime

from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from app.engine.data_processing.data_loaders import load_index
from app.engine.guardrails.input_guardrails import InputGuardrails
from app.engine.inference.Judge import Judge
from app.engine.inference.retrievers.hybrid_retriever import HybridRetriever
from app.utils.app_utils import pprint_debug
from app.utils.eval_utils import timed_operation
from app.utils.inference_utils import throttle_requests


class BaseInference(ABC):
    """
    Base class for inference operations in the meeting minutes assistant.

    Handles initialization of the document index, timespans generation,
    and embedding model setup. Provides base functionality for document querying.
    """

    def __init__(self, args):
        """
        Initialize the inference engine with required components.

        Args:
            args: Configuration arguments containing API keys, model paths, and other settings.
        """
        self.args = args

        self.model = self._create_model(args)

        self.index, (self.start_date, self.end_date) = load_index(args)
        self.document_batches = None
        self.generate_document_batches()

        self.ts_doc_index = {}
        for node in self.index.docstore.docs.values():
            if hasattr(node, 'metadata') and node.metadata:
                timestamp = node.metadata.get("meeting_datetime")
                file_name = node.metadata.get("file_name")
                if timestamp is not None and file_name is not None:
                    if timestamp not in self.ts_doc_index:
                        self.ts_doc_index[timestamp] = []
                    if file_name not in self.ts_doc_index[timestamp]:
                        self.ts_doc_index[timestamp].append(file_name)

        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                    cache_folder=args.embeddings_cache_dir)

        self.input_guardrails = InputGuardrails(args)

        self.prompt_template = PromptTemplate(
            "Vous analysez des comptes-rendus de réunions de projet. "
            "Répondez de manière factuelle et concise aux questions en utilisant uniquement les informations présentes dans les documents fournis.\n\n"
            "Requête: {query_string}\n\n"
            "Instructions:\n"
            "- Réponse directe sans formules de politesse\n"
            "- Citez les documents sources (nom du fichier)\n"
            "- Si l'information est incomplète, indiquez: 'Informations limitées. Consultez [nom du document]'\n"
            "- Pas de spéculation ou d'interprétation\n"
            "- Format: énumérations ou paragraphes courts\n\n"
            "Réponse:"
        )

        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.token_counter])
        Settings.callback_manager = self.callback_manager
        self.model_tpm = self.args.groq_model_inference_tpm

        self.judge = Judge(args)

        self.reranker = None

        if self.args.benchmark_mode:
            self.timing_data = {
                'retrieval_times': [],
                'reranker_times': [],
                'synthesis_times': [],
                'judge_times': [],
                'total_query_times': []
            }

    @abstractmethod
    def _create_model(self, args):
        """
        Create and return the LLM model instance.

        Args:
            args: Configuration arguments

        Returns:
            The initialized LLM model
        """
        pass

    def _get_reranker(self):
        """
        Get a ColbertRerank instance, creating it if necessary.
        
        Returns:
            ColbertRerank: A reranker instance

        DEV NOTE: In a production setting, consider using a more robust reranker,
        e.g. Cohere Rerank.
        """
        if self.reranker is None:
            self.reranker = ColbertRerank(top_n=5)
        return self.reranker

    def _cleanup_resources(self):
        """
        Clean up memory-intensive resources to prevent memory leaks.
        Called after processing queries to free up memory.
        """
        if self.reranker is not None:
            del self.reranker
            self.reranker = None

        gc.collect()

    def generate_document_batches(self):
        """
        Generate document batches by splitting unique timestamps into groups.
        """
        timestamps = set()
        for node in self.index.docstore.docs.values():
            if hasattr(node, 'metadata') and node.metadata:
                timestamp = node.metadata.get("meeting_datetime")
                if timestamp is not None:
                    timestamps.add(timestamp)

        sorted_timestamps = sorted(timestamps)

        self.document_batches = []
        for i in range(0, len(sorted_timestamps), self.args.n_batch):
            batch = sorted_timestamps[i:i + self.args.n_batch]
            self.document_batches.append(batch)

    @timed_operation('retrieval_times', timespan_aware=True)
    def _retrieve_nodes(self, hybrid_retriever, query_bundle, timespan_idx=None):
        """Wrapper for retrieval operation."""
        return hybrid_retriever.retrieve(query_bundle)

    @timed_operation('reranker_times', timespan_aware=True)
    def _rerank_nodes(self, reranker, top_candidates, query_bundle, timespan_idx=None):
        """Wrapper for reranking operation."""
        return reranker.postprocess_nodes(top_candidates, query_bundle)

    @timed_operation('synthesis_times', timespan_aware=True)
    def _synthesize_response(self, response_synthesizer, query_string, reranked_nodes, timespan_idx=None):
        """Wrapper for response synthesis."""
        return response_synthesizer.synthesize(
            query=self.prompt_template.format(query_string=query_string),
            nodes=reranked_nodes
        )

    def _filter_nodes_by_timestamp_batch(self, nodes: List, timestamp_batch: List) -> List:
        """Filter nodes based on timestamp batch."""
        timestamp_set = set(timestamp_batch)
        filtered_nodes = []
        for node in nodes:
            meeting_time = node.metadata.get("meeting_datetime")
            if meeting_time in timestamp_set:
                filtered_nodes.append(node)
        return filtered_nodes

    @timed_operation('judge_times')
    def _filter_results(self, query_string, results):
        """Wrapper for judge operation."""
        cleaned_results = [results[0]]
        for i in range(1, len(results)):
            answer_prev = cleaned_results[-1][0]
            answer_next = results[i][0]
            judgement = self.judge.run(query_string, answer_prev, answer_next)

            if judgement:
                min_timestamp = min(cleaned_results[-1][3][0], results[i][3][0])
                max_timestamp = max(cleaned_results[-1][3][1], results[i][3][1])

                if cleaned_results[-1][1] is None:
                    cleaned_results[-1] = (
                        cleaned_results[-1][0],
                        {},
                        cleaned_results[-1][2],
                        (min_timestamp, max_timestamp)
                    )
                if results[i][1] is not None:
                    cleaned_results[-1][1].update(results[i][1])

                merged_batch_idx = cleaned_results[-1][2][0]
                merged_timestamp_batch = list(set(cleaned_results[-1][2][1] + results[i][2][1]))

                cleaned_results[-1] = (
                    cleaned_results[-1][0],
                    cleaned_results[-1][1],
                    (merged_batch_idx, merged_timestamp_batch),
                    (min_timestamp, max_timestamp)
                )
            else:
                cleaned_results.append(results[i])

        return cleaned_results

    @throttle_requests()
    @timed_operation('total_query_times')
    def query_llm(self, query_string: str, alpha: float = 0.5):
        """
        Process a query using enhanced hybrid retrieval with French preprocessing and RRF.

        Args:
            query_string: User's query text.
            alpha: Weight for combining dense and sparse scores (0.0 = only sparse, 1.0 = only dense).

        Returns:
            list: List of tuples containing (answer, metadata, timespan) for each processed result
        """

        if not self.args.disable_guardrails:
            is_valid, validation_reason = self.input_guardrails.validate_query(query_string)

            if not is_valid:
                rejection_message = self.input_guardrails.get_rejection_message(validation_reason)
                pprint_debug(f"Query rejected: {rejection_message}")

                if self.args.anon:
                    _pdf_path = "app/assets/idle_screen_anon.pdf"
                else:
                    _pdf_path = "app/assets/idle_screen.pdf"

                # Placeholder metadata for GUI rendering
                placeholder_metadata = {
                    "placeholder_node": {
                        'score': 0.0,
                        'text': "There is no relevant information to your query",
                        'metadata': {
                            'file_name': "Home",
                            'file_path': _pdf_path,
                            'page_number': 1,
                            'meeting_datetime': round(datetime.now().timestamp())
                        }
                    }
                }
                current_timestamp = round(datetime.now().timestamp())
                return [(rejection_message, placeholder_metadata, (0, [current_timestamp]),
                         (current_timestamp, current_timestamp))]
            else:
                pprint_debug("Valid query received for processing.")

        results = []
        reranker = self._get_reranker()
        # query_bundle = QueryBundle(query_str=query_string) # TODO: Remove this line if unnecessary

        all_node_ids = [node.id_ for node in self.index.docstore.docs.values()]
        all_nodes = self.index.docstore.get_nodes(all_node_ids)

        for batch_idx, query_batch in enumerate(self.document_batches):
            batch_nodes = self._filter_nodes_by_timestamp_batch(all_nodes, query_batch)

            hybrid_retriever = HybridRetriever(
                vector_index=self.index,
                nodes=batch_nodes,
                similarity_top_k=50,
                alpha=alpha,
                callback_manager=self.callback_manager
            )

            query_bundle = QueryBundle(query_str=query_string)
            query_bundle.query_batch = query_batch

            combined_nodes = self._retrieve_nodes(hybrid_retriever, query_bundle, timespan_idx=batch_idx)

            top_candidates = combined_nodes[:50]
            reranked_nodes = self._rerank_nodes(reranker, top_candidates, query_bundle, timespan_idx=batch_idx)

            response_synthesizer = get_response_synthesizer(llm=self.model)
            answer = self._synthesize_response(response_synthesizer, query_string, reranked_nodes,
                                               timespan_idx=batch_idx)

            metadata = {}
            for node in reranked_nodes:
                metadata[node.id_] = {
                    'score': node.score,
                    'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'metadata': node.metadata
                }

            min_batch_timestamp = min(query_batch)
            max_batch_timestamp = max(query_batch)
            results.append(
                (answer.response, metadata, (batch_idx, query_batch), (min_batch_timestamp, max_batch_timestamp)))

        cleaned_results = self._filter_results(query_string, results)

        # Limit the output to the user to 3 nodes for readability. This is a heuristic.
        for i in range(len(cleaned_results)):
            metadata = cleaned_results[i][1]
            top_nodes = heapq.nlargest(3, metadata.items(), key=lambda item: item[1]['score'])
            top_metadata = {node_id: data for node_id, data in top_nodes}
            cleaned_results[i] = (
                cleaned_results[i][0],
                top_metadata,
                cleaned_results[i][2],
                cleaned_results[i][3]
            )

        self._cleanup_resources()
        return cleaned_results
