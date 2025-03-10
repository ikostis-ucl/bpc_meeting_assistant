import calendar
import heapq
from datetime import datetime

from halo import Halo
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from app.engine.data_processing.data_loaders import load_index
from app.engine.inference.Judge import Judge
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
        # Load document index and get timespan boundaries
        self.model = None

        self.index, (self.start_date, self.end_date) = load_index(args)
        self.timespans = None
        self.generate_timespans(self.start_date, self.end_date, args.time_freq)

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                    cache_folder=args.embeddings_cache_dir)

        # Set up prompt template for queries in French
        self.prompt_template = PromptTemplate(
            "Vous êtes un(e) assistant(e) qui aide un chef de projet à extraire des informations de documents qui "
            "incluent le déroulement de réunions autour d'un certain projet. Il s'agit de la requête du chef "
            "de projet:\n"
            "Requête: {query_string}\n"
            "Répondez de manière aussi cohérente que possible. Votre réponse doit être rédigée en français."
        )

        # Set up token counting and callbacks
        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.token_counter])
        Settings.callback_manager = self.callback_manager

        # Initialize reranker and judge components
        self.reranker = ColbertRerank(top_n=5)
        self.judge = Judge(args)

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

    @Halo(text=fmt_string(Color.CYAN, '[CONSOLE] Querying model...'),
          placement='right', animation='bounce', spinner='moon')
    @throttle_requests()
    def query_llm(self, query_string):
        """
        Process a query through the LLM and clean results.

        Args:
            query_string: User's query text.

        Returns:
            list: List of tuples containing (answer, metadata, timespan) for each processed result.
        """
        results = []

        # Process query for each timespan
        for start_date, end_date in self.timespans:
            # Configure query engine with filters and processors
            query_engine = self.index.as_query_engine(
                llm=self.model,
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(key="meeting_datetime",
                                       value=start_date,
                                       operator=FilterOperator.GTE),
                        MetadataFilter(key="meeting_datetime",
                                       value=end_date,
                                       operator=FilterOperator.LTE),
                    ]
                ),
                similarity_top_k=50,
                node_postprocessors=[self.reranker],
            )

            # Execute query and store results
            answer = query_engine.query(self.prompt_template.format(query_string=query_string))

            # Attach node.score to each corresponding node in the metadata
            if answer.metadata:
                for node in answer.source_nodes:
                    if node.id_ in answer.metadata:
                        answer.metadata[node.id_]['score'] = node.score

            results.append((answer.response, answer.metadata, (start_date, end_date)))

        # Clean and merge similar results using the judge
        cleaned_results = [results[0]]
        for i in range(1, len(results)):
            answer_prev = cleaned_results[-1][0]
            answer_next = results[i][0]
            judgement = self.judge.run(query_string, answer_prev, answer_next)

            if judgement:
                # Merge metadata and extend timespans for similar answers
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

        # Keep metadata of top 3 nodes based on their scores
        for i in range(len(cleaned_results)):
            metadata = cleaned_results[i][1]
            top_nodes = heapq.nlargest(3, metadata.items(), key=lambda item: item[1]['score'])
            top_metadata = {node_id: data for node_id, data in top_nodes}
            cleaned_results[i] = (cleaned_results[i][0], top_metadata, cleaned_results[i][2])

        return cleaned_results
