from halo import Halo
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from app.engine.inference.Judge import Judge
from app.engine.inference.base_inference import BaseInference
from app.utils.app_utils import fmt_string, Color
from app.utils.inference_utils import throttle_requests


class RAGInference(BaseInference):
    def __init__(self, args):
        super().__init__(args)

        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.token_counter])
        Settings.callback_manager = self.callback_manager
        self.model.callback_manager = Settings.callback_manager

        self.reranker = ColbertRerank(top_n=5)
        self.judge = Judge(args)  # Initialize the Judge

    @Halo(text=fmt_string(Color.CYAN, '[CONSOLE] Querying model...'),
          placement='right', animation='bounce', spinner='moon')
    @throttle_requests()
    def query_llm(self, query_string):
        results = []

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

        # Clean up results based on judgements
        cleaned_results = [results[0]]
        for i in range(1, len(results)):
            answer_prev = cleaned_results[-1][0]
            answer_next = results[i][0]
            judgement = self.judge.run(query_string, answer_prev, answer_next)
            if judgement:
                # Merge metadata and extend timespan
                if cleaned_results[-1][1] is None:
                    cleaned_results[-1] = (cleaned_results[-1][0], {}, cleaned_results[-1][2])
                if results[i][1] is not None:
                    cleaned_results[-1][1].update(results[i][1])
                cleaned_results[-1] = (
                    cleaned_results[-1][0], cleaned_results[-1][1], (cleaned_results[-1][2][0], results[i][2][1]))
            else:
                cleaned_results.append(results[i])

        print("Checkpoint: Querying model... Done.")
        return cleaned_results
