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
        # Initialize Groq model for similarity judgement
        self.model = Groq(model=args.groq_model_inference_judge,
                          api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.0)

        # Define prompt template for comparison
        self.prompt_template = """\
        Quand je pose la question suivante \
        ------
        Question: \
        {query_string} \
        ------ \
        j'obtiens deux réponses : \
        ------ \
        Réponse 1 : {answer_prev} \
        ------ \
        Réponse 2 : {answer_next} \
        ------ \
        En prenant tout en considération, si les deux réponses sont essentiellement les mêmes, \
        j'aimerais que vous me répondiez "True". S'il y a même de petites différences dans les détails \
        fournis dans les réponses qui différencient le sens de la réponse, j'aimerais que vous me répondiez "False".\
        Me répondre en n'utilisant que les mots "True" ou "False".
        """

        # Set up LLM program with validation
        self.program = LLMTextCompletionProgram.from_defaults(
            llm=self.model,
            output_cls=Response,
            prompt_template_str=self.prompt_template,
            verbose=True,
        )

    def run(self, query_string, answer_prev, answer_next):
        """
        Compare two answers to determine if they are essentially the same.

        Args:
            query_string: Original query that generated the answers.
            answer_prev: First answer to compare.
            answer_next: Second answer to compare.

        Returns:
            bool: True if answers are similar, False otherwise.
        """
        try:
            output = self.program(query_string=query_string,
                                  answer_prev=answer_prev,
                                  answer_next=answer_next)
        except ValidationError as exc:
            pprint_error(f"{exc}")
            return False
        return output.judgement
