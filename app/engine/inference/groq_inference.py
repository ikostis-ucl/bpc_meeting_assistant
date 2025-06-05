from halo import Halo
from llama_index.core import Settings
from llama_index.llms.groq import Groq

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
        self.model = Groq(model=args.groq_model_inference,
                          api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.0)
        self.model_tpm = args.groq_model_inference_tpm

        # Set up token counting and callbacks
        self.model.callback_manager = Settings.callback_manager

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
        return super().query_llm(query_string)
