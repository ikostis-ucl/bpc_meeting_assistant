from llama_index.core import Settings
from llama_index.llms.groq import Groq

from app.engine.inference.base_inference import BaseInference


class GroqInference(BaseInference):
    """
    Groq-based inference implementation for the meeting minutes assistant.

    Extends BaseInference to provide specific implementation using Groq's LLM.
    Handles query processing, reranking, and result cleaning through a judge system.
    """

    def __init__(self, args):
        """
        Initialize Groq inference with models and processors.

        Args:
            args: Configuration arguments for models and API settings.
        """
        self.model_tpm = args.groq_model_inference_tpm

        super().__init__(args)

    def _create_model(self, args):
        """
        Create Groq model instance.

        Args:
            args: Configuration arguments

        Returns:
            Groq: The initialized Groq model
        """
        model = Groq(
            model=args.groq_model_inference,
            api_key=args.groq_api_key,
            model_kwargs={"seed": 42},
            temperature=0.0
        )
        model.callback_manager = Settings.callback_manager
        return model
