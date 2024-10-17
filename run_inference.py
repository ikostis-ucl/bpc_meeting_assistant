import os
import warnings

from app.engine.inference.groq_inference import GroqInference as InferenceEngine
from app.utils.app_utils import pprint_qa
from configuration import config_parser

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = config_parser()
    args = parser.parse_args()

    agent = InferenceEngine(args=args)
    questions = [
        "Liste des décisions prises concernant le carrelage des salles des bains (SDBs) et les dates (jour/mois/année) auxquelles elles ont été prises.",
        "Quelle est la couleur choisie (RAL) pour les châssis ?",
        "Quelle est la date de remise des parties communes ?"]
    start_dates = [None, "2023-01-01", "2023-02-15", "2023-12-31"]
    end_dates = [None, "2023-02-15", "2023-12-31", None]
    for question in questions:
        for start_date, end_date in zip(start_dates, end_dates):
            answer, metadata = agent.query_llm(query_string=question, start_date=start_date, end_date=end_date)
            pprint_qa(question, answer, metadata, dates=[agent.retriever.datetime_span['start_date'],
                                                         agent.retriever.datetime_span['end_date']])
