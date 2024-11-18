import datetime

from halo import Halo
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from app.engine.inference.base_inference import BaseInference
from app.utils.app_utils import fmt_string, Color, datetime_to_timestamp


class RAGInference(BaseInference):
    def __init__(self, args):
        super().__init__(args)

        self.reranker = ColbertRerank(top_n=5)

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
                                                  similarity_top_k=50,
                                                  node_postprocessors=[self.reranker]
                                                  )

        answer = query_engine.query(self.prompt_template.format(query_string=query_string))

        return answer.response, answer.metadata, (start_date, end_date)
