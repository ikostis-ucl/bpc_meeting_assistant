import os
import warnings

from app.engine.inference.groq_inference import GroqInference as InferenceEngine
from configuration import config_parser
from app.utils.app_utils import pprint_console

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = config_parser()
    args = parser.parse_args()

    agent = InferenceEngine(args=args)
    questions = ["Quelle est la demande de la STAB en ce qui concerne les escaliers? Quand a-t-elle présenté cette demande ?"]
    start_dates = ["01/09/2022"]
    end_dates = ["01/11/2022"]
    for question, start_date, end_date in zip(questions, start_dates, end_dates):
        results = agent.query_llm(query_string=question, start_date=start_date, end_date=end_date)
        pprint_console(results)
