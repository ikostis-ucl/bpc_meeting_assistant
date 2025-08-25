import calendar
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
        self.timespans = None
        self.generate_timespans(self.start_date, self.end_date, args.time_freq)

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
        # Release reranker resources
        if self.reranker is not None:
            del self.reranker
            self.reranker = None

        # Force garbage collection to reclaim memory
        gc.collect()

    def generate_timespans(self, starting_month_timestamp, ending_month_timestamp, time_freq):
        """
        Generate time spans between start and end dates based on specified frequency.

        Args:
            starting_month_timestamp: Start date timestamp.
            ending_month_timestamp: End date timestamp.
            time_freq: Time frequency in months.
        """

        def add_months(source_date, months):
            """
            Helper function to add months to a date while handling month/year transitions.

            Args:
                source_date: Base date to add months to.
                months: Number of months to add.

            Returns:
                datetime: New date after adding specified months.
            """
            month = source_date.month - 1 + months
            year = source_date.year + month // 12
            month = month % 12 + 1
            day = min(source_date.day, calendar.monthrange(year, month)[1])
            return datetime(year, month, day)

        self.timespans = []

        # Convert timestamps to datetime objects
        start_date = datetime.fromtimestamp(starting_month_timestamp)
        end_date = datetime.fromtimestamp(ending_month_timestamp)

        current_start = start_date

        # Generate timespan intervals
        while current_start < end_date:
            current_end = add_months(current_start, time_freq)
            if current_end > end_date:
                current_end = end_date
            self.timespans.append((int(current_start.timestamp()), int(current_end.timestamp())))
            current_start = current_end

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

    @timed_operation('judge_times')
    def _filter_results(self, query_string, results):
        """Wrapper for judge operation."""
        cleaned_results = [results[0]]
        for i in range(1, len(results)):
            answer_prev = cleaned_results[-1][0]
            answer_next = results[i][0]
            judgement = self.judge.run(query_string, answer_prev, answer_next)

            if judgement:
                if cleaned_results[-1][1] is None:
                    cleaned_results[-1] = (cleaned_results[-1][0], {}, cleaned_results[-1][2])
                if results[i][1] is not None:
                    cleaned_results[-1][1].update(results[i][1])
                cleaned_results[-1] = (
                    cleaned_results[-1][0],
                    cleaned_results[-1][1],
                    (cleaned_results[-1][2][0], results[i][2][1])
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

                # Create placeholder metadata for GUI rendering
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

                return [(rejection_message, placeholder_metadata, (0, round(datetime.now().timestamp())))]
            else:
                pprint_debug("Valid query received for processing.")

        results = []
        reranker = self._get_reranker()
        query_bundle = QueryBundle(query_str=query_string)

        all_node_ids = [node.id_ for node in self.index.docstore.docs.values()]
        all_nodes = self.index.docstore.get_nodes(all_node_ids)

        for timespan_idx, (start_date, end_date) in enumerate(self.timespans):
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

            combined_nodes = self._retrieve_nodes(hybrid_retriever, query_bundle, timespan_idx=timespan_idx)

            top_candidates = combined_nodes[:20]
            reranked_nodes = self._rerank_nodes(reranker, top_candidates, query_bundle, timespan_idx=timespan_idx)

            response_synthesizer = get_response_synthesizer(llm=self.model)
            answer = self._synthesize_response(response_synthesizer, query_string, reranked_nodes,
                                               timespan_idx=timespan_idx)

            # Process metadata with RRF scores
            metadata = {}
            for node in reranked_nodes:
                metadata[node.id_] = {
                    'score': node.score,
                    'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'metadata': node.metadata
                }

            results.append((answer.response, metadata, (start_date, end_date)))

        cleaned_results = self._filter_results(query_string, results)

        # Keep top 3 nodes based on scores for showcasing to the user.
        for i in range(len(cleaned_results)):
            metadata = cleaned_results[i][1]
            top_nodes = heapq.nlargest(3, metadata.items(), key=lambda item: item[1]['score'])
            top_metadata = {node_id: data for node_id, data in top_nodes}
            cleaned_results[i] = (cleaned_results[i][0], top_metadata, cleaned_results[i][2])

        self._cleanup_resources()
        return cleaned_results
