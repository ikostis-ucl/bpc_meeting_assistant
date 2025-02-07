import calendar
from datetime import datetime

from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from app.engine.data_processing.data_loaders import load_index


class BaseInference:
    def __init__(self, args):
        self.model = Groq(model="llama3-70b-8192", api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.0)
        self.model_tpm = 6000

        self.index, (_start_date, _end_date) = load_index(args)
        self.timespans = None
        self._generate_timespans(_start_date, _end_date, args.time_freq)

        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                    cache_folder=args.embeddings_cache_dir)
        self.prompt_template = PromptTemplate(
            "Vous êtes un(e) assistant(e) qui aide un chef de projet à extraire des informations de documents qui "
            "incluent le déroulement de réunions autour d'un certain projet. Il s'agit de la requête du chef "
            "de projet:\n"
            "Requête: {query_string}\n"
            "Répondez de manière aussi cohérente que possible. Votre réponse doit être rédigée en français."
        )

    def _generate_timespans(self, starting_month_timestamp, ending_month_timestamp, time_freq):
        def add_months(source_date, months):
            month = source_date.month - 1 + months
            year = source_date.year + month // 12
            month = month % 12 + 1
            day = min(source_date.day, calendar.monthrange(year, month)[1])
            return datetime(year, month, day)

        self.timespans = []

        start_date = datetime.fromtimestamp(starting_month_timestamp)
        end_date = datetime.fromtimestamp(ending_month_timestamp)

        current_start = start_date

        while current_start < end_date:
            current_end = add_months(current_start, time_freq)
            if current_end > end_date:
                current_end = end_date
            self.timespans.append((int(current_start.timestamp()), int(current_end.timestamp())))
            current_start = current_end
