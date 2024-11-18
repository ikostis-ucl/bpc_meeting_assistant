import datetime

from halo import Halo
from llama_index.core.query_engine import RetrieverQueryEngine

from app.engine.inference.base_inference import BaseInference
from app.engine.retriever.temporal_retriever import TemporalRetriever
from app.utils.app_utils import fmt_string, Color, datetime_to_timestamp


class TRAGInference(BaseInference):
    def __init__(self, args):
        super().__init__(args)

        self.cutoff_percentage = args.cutoff_percentage
        self.__api_key = args.cohere_api_key
        self.retriever = None

    @Halo(text=fmt_string(Color.CYAN, 'Querying model...'), placement='right', animation='bounce', spinner='moon')
    def query_llm(self, query_string, start_date, end_date):
        if start_date is None:
            start_date = datetime.datetime(1970, 1, 1)
        if end_date is None:
            end_date = datetime.datetime.now()

        self.retriever = TemporalRetriever(embed_model=self.embedding_model,
                                           index=self.index,
                                           cutoff_percentage=self.cutoff_percentage,
                                           key=self.__api_key)

        self.retriever.datetime_span = {"start_date": datetime_to_timestamp(start_date),
                                        "end_date": datetime_to_timestamp(end_date)}

        query_engine = RetrieverQueryEngine.from_args(retriever=self.retriever, llm=self.model)

        answer = query_engine.query(self.prompt_template.format(query_string=query_string))

        return answer.response, answer.metadata, (start_date, end_date)
