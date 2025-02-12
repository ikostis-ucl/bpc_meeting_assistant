from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.groq import Groq
from pydantic import BaseModel, ValidationError
from app.utils.app_utils import pprint_error

class Response(BaseModel):
    judgement: bool


class Judge:
    def __init__(self, args):
        self.args = args
        self.model = Groq(model="llama-3.3-70b-specdec", api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.0)

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

        self.program = LLMTextCompletionProgram.from_defaults(
            llm=self.model,
            output_cls=Response,
            prompt_template_str=self.prompt_template,
            verbose=True,
        )

    def run(self, query_string, answer_prev, answer_next):
        try:
            output = self.program(query_string=query_string,
                                  answer_prev=answer_prev,
                                  answer_next=answer_next)
        except ValidationError as exc:
            pprint_error(f"{exc}")
            return False
        return output.judgement
