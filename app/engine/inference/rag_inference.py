import time

from halo import Halo
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from app.engine.inference.base_inference import BaseInference
from app.utils.app_utils import fmt_string, Color, pprint_debug


class RAGInference(BaseInference):
    def __init__(self, args):
        super().__init__(args)

        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.token_counter])
        Settings.callback_manager = self.callback_manager
        self.model.callback_manager = Settings.callback_manager

        self.reranker = ColbertRerank(top_n=5)

    @Halo(text=fmt_string(Color.YELLOW, '[CONSOLE] Querying model...'),
          placement='right', animation='bounce', spinner='moon')
    def query_llm(self, query_string):
        results = []
        start_time = time.time()

        for start_date, end_date in self.timespans:
            query_engine = self.index.as_query_engine(llm=self.model,
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

            answer = query_engine.query(self.prompt_template.format(query_string=query_string))
            results.append((answer.response, answer.metadata, (start_date, end_date)))

            elapsed_time = time.time() - start_time
            if self.token_counter.total_llm_token_count >= self.model_limiters["tokens_per_minute"]:
                sleep_time = 60 - elapsed_time
                if sleep_time > 0:
                    with Halo(text=fmt_string(Color.YELLOW, '[CONSOLE] API rate limit reached, waiting...'),
                              placement='right', animation='bounce', spinner='dots'):
                        time.sleep(sleep_time)
                self.token_counter.reset_counts()
                start_time = time.time()

        pprint_debug("Checkpoint: Querying model... Done.")
        return results
