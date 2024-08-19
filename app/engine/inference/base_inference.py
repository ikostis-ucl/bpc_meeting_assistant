from halo import Halo
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.engine.data_processing.data_loaders import load_index
from app.engine.retriever.temporal_retriever import TemporalRetriever
from app.utils.app_utils import fmt_string, Color


class BaseInference:
    def __init__(self, args):
        self.model = None
        self.prompt_template = None

        self.index = load_index(args)

        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model)

        self.retriever = TemporalRetriever(embed_model=self.embedding_model, index=self.index)

        self.prompt_template = PromptTemplate(
            "Vous êtes un(e) assistant(e) qui aide un chef de projet à extraire des informations de documents qui "
            "incluent le déroulement de réunions autour d'un certain projet. Il s'agit de la requête du chef "
            "de projet:\n"
            "Requête: {query_string}\n"
            "Répondez de manière aussi cohérente que possible. Votre réponse doit être rédigée en français."
        )

    @Halo(text=fmt_string(Color.CYAN, 'Querying model...'), placement='right', animation='bounce', spinner='moon')
    def query_llm(self, query_string, start_date, end_date):
        self.retriever.set_datetime_span(start_date=start_date, end_date=end_date)
        query_engine = RetrieverQueryEngine.from_args(retriever=self.retriever, llm=self.model)

        answer = query_engine.query(self.prompt_template.format(query_string=query_string))

        return answer.response, answer.metadata
