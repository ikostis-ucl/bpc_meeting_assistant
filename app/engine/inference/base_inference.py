import calendar
import gc
import heapq
import time
from datetime import datetime
from typing import List

from halo import Halo
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from app.engine.data_processing.data_loaders import load_index
from app.engine.inference.Judge import Judge
from app.engine.inference.retrievers.hybrid_retriever import HybridRetriever
from app.utils.app_utils import fmt_string, Color
from app.utils.inference_utils import throttle_requests


class BaseInference:
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

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                    cache_folder=args.embeddings_cache_dir)

        # Set up prompt template for queries in French
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

        # Set up token counting and callbacks
        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.token_counter])
        Settings.callback_manager = self.callback_manager

        # Initialize judge component
        self.judge = Judge(args)

        # Don't initialize reranker until needed
        self.reranker = None

        if args.benchmark_mode:
            self.timing_data = {
                'retrieval_times': [],
                'reranker_times': [],
                'synthesis_times': [],
                'judge_times': [],
                'total_query_times': []
            }

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

    def _get_filtered_nodes_for_timespan(self, start_date: int, end_date: int) -> List:
        """Get filtered nodes for a specific timespan."""
        all_node_ids = [node.id_ for node in self.index.docstore.docs.values()]
        all_nodes = self.index.docstore.get_nodes(all_node_ids)

        filtered_nodes = []
        for node in all_nodes:
            if node.metadata.get("meeting_datetime"):
                meeting_time = node.metadata["meeting_datetime"]
                if start_date <= meeting_time <= end_date:
                    filtered_nodes.append(node)

        return filtered_nodes

    def _record_timing(self, operation: str, duration: float, timespan_idx: int = None):
        """Record timing data for benchmark analysis."""
        if not self.args.benchmark_mode:
            return

        timing_entry = {
            'duration': duration,
            'timespan_idx': timespan_idx,
            'timestamp': time.time()
        }

        if operation in self.timing_data:
            self.timing_data[operation].append(timing_entry)

    def get_timing_summary(self):
        """Get summary of timing data for benchmark analysis."""
        if not self.args.benchmark_mode:
            return None

        summary = {}
        for operation, timings in self.timing_data.items():
            if timings:
                durations = [t['duration'] for t in timings]
                summary[operation] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'average': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        return summary

    @Halo(text=fmt_string(Color.CYAN, '[CONSOLE] Querying model...'),
          placement='right', animation='bounce', spinner='moon')
    @throttle_requests()
    def query_llm(self, query_string: str, alpha: float = 0.5):
        """
        Process a query using enhanced hybrid retrieval with French preprocessing and RRF.

        Args:
            query_string: User's query text.
            alpha: Weight for combining dense and sparse scores (0.0 = only sparse, 1.0 = only dense).

        Returns:
            list: List of tuples containing (answer, metadata, timespan) for each processed result.
        """
        query_start_time = time.time() if self.args.benchmark_mode else None

        results = []
        reranker = self._get_reranker()
        query_bundle = QueryBundle(query_str=query_string)

        # Get all nodes once for efficiency
        all_node_ids = [node.id_ for node in self.index.docstore.docs.values()]
        all_nodes = self.index.docstore.get_nodes(all_node_ids)

        for timespan_idx, (start_date, end_date) in enumerate(self.timespans):
            # Create custom hybrid retriever with timespan filtering
            hybrid_retriever = HybridRetriever(
                vector_index=self.index,
                nodes=all_nodes,  # Pass all nodes, retriever will filter
                similarity_top_k=50,
                alpha=alpha,
                start_date=start_date,
                end_date=end_date,
                callback_manager=self.callback_manager
            )

            # Retrieve using custom retriever
            retrieval_start = time.time() if self.args.benchmark_mode else None
            combined_nodes = hybrid_retriever.retrieve(query_bundle)
            if self.args.benchmark_mode:
                retrieval_time = time.time() - retrieval_start
                self._record_timing('retrieval_times', retrieval_time, timespan_idx)

            # Apply reranking to top candidates
            top_candidates = combined_nodes[:20]
            reranker_start = time.time() if self.args.benchmark_mode else None
            reranked_nodes = reranker.postprocess_nodes(top_candidates, query_bundle)
            if self.args.benchmark_mode:
                reranker_time = time.time() - reranker_start
                self._record_timing('reranker_times', reranker_time, timespan_idx)

            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(llm=self.model)

            # Generate answer
            synthesis_start = time.time() if self.args.benchmark_mode else None
            answer = response_synthesizer.synthesize(
                query=self.prompt_template.format(query_string=query_string),
                nodes=reranked_nodes
            )
            if self.args.benchmark_mode:
                synthesis_time = time.time() - synthesis_start
                self._record_timing('synthesis_times', synthesis_time, timespan_idx)

            # Process metadata with RRF scores
            metadata = {}
            for node in reranked_nodes:
                metadata[node.id_] = {
                    'score': node.score,
                    'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'metadata': node.metadata
                }

            results.append((answer.response, metadata, (start_date, end_date)))

        judge_start = time.time() if self.args.benchmark_mode else None
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

        if self.args.benchmark_mode:
            judge_time = time.time() - judge_start
            self._record_timing('judge_times', judge_time)

        # Keep top 3 nodes based on scores
        for i in range(len(cleaned_results)):
            metadata = cleaned_results[i][1]
            top_nodes = heapq.nlargest(3, metadata.items(), key=lambda item: item[1]['score'])
            top_metadata = {node_id: data for node_id, data in top_nodes}
            cleaned_results[i] = (cleaned_results[i][0], top_metadata, cleaned_results[i][2])

        if self.args.benchmark_mode:
            total_query_time = time.time() - query_start_time
            self._record_timing('total_query_times', total_query_time)

        self._cleanup_resources()
        return cleaned_results
