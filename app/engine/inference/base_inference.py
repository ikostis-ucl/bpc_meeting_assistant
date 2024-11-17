import datetime

from halo import Halo
from llama_index.core import PromptTemplate
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from app.engine.data_processing.data_loaders import load_index
from app.utils.app_utils import fmt_string, Color, datetime_to_timestamp


class BaseInference:
    def __init__(self, args):
        self.model = None

        self.index = load_index(args)

        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model)

        self.prompt_template = PromptTemplate(
            "Vous êtes un(e) assistant(e) qui aide un chef de projet à extraire des informations de documents qui "
            "incluent le déroulement de réunions autour d'un certain projet. Il s'agit de la requête du chef "
            "de projet:\n"
            "Requête: {query_string}\n"
            "Répondez de manière aussi cohérente que possible. Votre réponse doit être rédigée en français."
        )

        self.reranker = CohereRerank(api_key=args.cohere_api_key, top_n=5)

    @Halo(text=fmt_string(Color.CYAN, 'Querying model...'), placement='right', animation='bounce', spinner='moon')
    def query_llm(self, query_string, start_date, end_date):

        if start_date is None:
            start_date = datetime.datetime(1970, 1, 1)
        if end_date is None:
            end_date = datetime.datetime.now()

        query_engine = self.index.as_query_engine(llm=self.model,
                                                  filters=MetadataFilters(
                                                      filters=[
                                                          MetadataFilter(key="meeting_datetime",
                                                                         value=datetime_to_timestamp(start_date),
                                                                         operator=FilterOperator.GTE),
                                                          MetadataFilter(key="meeting_datetime",
                                                                         value=datetime_to_timestamp(end_date),
                                                                         operator=FilterOperator.LTE),
                                                      ]
                                                  ),
                                                  vector_store_query_mode="hybrid",
                                                  sparse_top_k=10,
                                                  similarity_top_k=10,
                                                  node_postprocessors=[self.reranker]
                                                  )

        answer = query_engine.query(self.prompt_template.format(query_string=query_string))

        return answer.response, answer.metadata, (start_date, end_date)
