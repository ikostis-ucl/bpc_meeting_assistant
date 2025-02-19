# groq_inference.py
from halo import Halo
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.groq import Groq
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from app.engine.inference.Judge import Judge
from app.engine.inference.base_inference import BaseInference
from app.utils.app_utils import fmt_string, Color
from app.utils.inference_utils import throttle_requests


class GroqInference(BaseInference):
    """
    Groq-based inference implementation for the meeting minutes assistant.

    Extends BaseInference to provide specific implementation using Groq's LLM.
    Handles query processing, reranking, and result cleaning through a judge system.

    Dev Note:
    To scale this class, one has to figure out how to instantiate the Groq model and the ColbertRerank postprocessor.
    Multiple API keys, or a single API key with a high rate limit, are required to handle the number of queries.

    Implementing a local Inference is viable through Ollama, if scaling through the API is not an option.
    https://docs.llamaindex.ai/en/stable/api_reference/llms/ollama/
    """

    def __init__(self, args):
        """
        Initialize Groq inference with models and processors.

        Args:
            args: Configuration arguments for models and API settings.
        """
        super().__init__(args)
        # Initialize Groq model with specific settings
        self.model = Groq(model="llama3-70b-8192", api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.0)
        self.model_tpm = 6000

        # Set up token counting and callbacks
        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager([self.token_counter])
        Settings.callback_manager = self.callback_manager
        self.model.callback_manager = Settings.callback_manager

        # Initialize reranker and judge components
        self.reranker = ColbertRerank(top_n=5)
        self.judge = Judge(args)

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

        return cleaned_results
