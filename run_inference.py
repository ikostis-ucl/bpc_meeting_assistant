import os
import warnings

from app.engine.inference.groq_inference import GroqInference as InferenceEngine
from app.utils.app_utils import pprint_qa
from configuration import config_parser


class Demo:
    def __init__(self):
        self.questions = []
        self.start_dates = []
        self.end_dates = []
        self.agent = None

    def inputs(self):
        ...

    def run(self):
        self.inputs()

        for question in self.questions:
            for start_date, end_date in zip(self.start_dates, self.end_dates):
                answer, metadata = self.agent.query_llm(query_string=question, start_date=start_date, end_date=end_date)
                pprint_qa(question, answer, metadata, dates=[self.agent.retriever.datetime_span['start_date'],
                                                             self.agent.retriever.datetime_span['end_date']])


class Test(Demo):
    def __init__(self):
        super().__init__()
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.args.input_path = "data/tests"
        self.args.storage_dir = "data/tests/vector_db_test"

        self.agent = InferenceEngine(args=self.args)

        self.questions = ["Quelles sont les décisions prises en matière de recharge des véhicules électriques ? "
                          "Indiquez-moi les dates (jj/mm/aaaa) auxquelles ces décisions ont été prises."]
        self.start_dates = [None, None, "2022-09-01", "2022-10-31"]
        self.end_dates = [None, "2022-10-01", "2022-10-31", None]


class Benchmark(Demo):
    def __init__(self):
        super().__init__()
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"

        self.agent = InferenceEngine(args=self.args)

        self.questions = [
            "Quelle est la date de remise des espaces communs ?",
            "Liste des décisions prises concernant le carrelage des salles des bains (SDBs) et les dates (jour/mois/année) auxquelles elles ont été prises.",
            "Quelle est la couleur choisie (RAL) pour les châssis ?"
        ]

        self.start_dates = [None, "2023-01-01", "2023-02-15", "2024-01-01"]
        self.end_dates = [None, "2023-02-15", "2023-12-31", None]


class InteractiveQuery(Demo):
    def __init__(self):
        super().__init__()
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"

        self.agent = InferenceEngine(args=self.args)

        self.questions = []
        self.start_dates = []
        self.end_dates = []

    def inputs(self):
        while True:
            question = input("Question: ")
            start_date = input("Start date (yyyy-mm-dd): ")
            end_date = input("End date (yyyy-mm-dd): ")

            if start_date == "":
                start_date = None
            if end_date == "":
                end_date = None

            self.questions.append(question)
            self.start_dates.append(start_date)
            self.end_dates.append(end_date)

            response = input("Do you want to ask another question? (y/[n]): ")
            if response == '' or response == 'n':
                break


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    scenario = Test()
    scenario.run()
