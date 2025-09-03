from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.groq import Groq
from pydantic import BaseModel, ValidationError

from app.utils.app_utils import pprint_error


class Response(BaseModel):
    """
    Pydantic model for judge response validation.

    Attributes:
        judgement (bool): Boolean indicating if answers are similar.
    """
    judgement: bool


class Judge:
    """
    Judge class for evaluating similarity between answers.

    Uses a language model to determine if two answers to the same query
    are essentially the same or contain meaningful differences.
    """

    def __init__(self, args):
        """
        Initialize the Judge with model and prompt configuration.

        Args:
            args: Configuration arguments containing API keys and model settings.
        """
        self.args = args
        self.model = Groq(model=args.groq_model_inference_judge,
                          api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.0)

        self.prompt_template = """Question posée: {query_string}

        Réponse 1: {answer_prev}

        Réponse 2: {answer_next}

        Ces deux réponses à la même question contiennent-elles la même information contextuelle?

        Retournez True si les réponses:
        - Répondent à la question de manière équivalente
        - Donnent les mêmes informations factuelles pertinentes
        - Arrivent aux mêmes conclusions principales

        Retournez False si:
        - Une réponse contient des informations pertinentes absentes de l'autre
        - Les conclusions diffèrent significativement
        - L'une répond mieux à la question posée

        Réponse (True/False seulement):"""

        self.program = LLMTextCompletionProgram.from_defaults(
            output_cls=Response,
            prompt_template_str=self.prompt_template,
            llm=self.model,
            verbose=False,
        )

    def run(self, query_string, answer_prev, answer_next):
        """
        Compare two consecutive answers for contextual similarity.

        Args:
            query_string: Original user query for context
            answer_prev: Previous timespan answer
            answer_next: Next timespan answer

        Returns:
            bool: True if answers are contextually similar, False otherwise
        """
        try:
            response = self.program(
                query_string=query_string,
                answer_prev=answer_prev,
                answer_next=answer_next
            )
            return response.judgement
        except ValidationError as e:
            pprint_error(f"Judge validation error: {e}")
            return False
        except Exception as e:
            pprint_error(f"Judge execution error: {e}")
            return False
