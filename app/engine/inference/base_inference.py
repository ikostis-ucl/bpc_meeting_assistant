import calendar
from datetime import datetime

from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.engine.data_processing.data_loaders import load_index


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
