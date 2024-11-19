import os
import time
import warnings

from app.engine.inference.rag_inference import RAGInference
from app.engine.inference.trag_inference import TRAGInference
from app.utils.app_utils import pprint_qa
from configuration import config_parser


class Demo:
    def __init__(self):
        self.questions = []
        self.dates = []
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.agent = None

    def inputs(self):
        ...

    def run(self):
        self.inputs()

        for question in self.questions:
            start_time = time.time()
            for start_date, end_date in self.dates:
                answer, metadata, (s_date, e_date) = self.agent.query_llm(query_string=question, start_date=start_date,
                                                                          end_date=end_date)
                pprint_qa(question, answer, metadata, dates=[s_date, e_date])
                elapsed_time = time.time() - start_time
                if elapsed_time < 20:
                    time.sleep(20 - elapsed_time) # Failsafe for the Cohere API rate limit


class Test(Demo):
    def __init__(self):
        super().__init__()
        self.args.input_path = "data/tests"
        self.args.storage_dir = "data/tests/vector_db_test"
        self.questions = [
            "Quelles sont les décisions prises en matière des stations du recharge des véhicules électriques ? Indiquez-moi les dates (jj/mm/aaaa) auxquelles ces décisions ont été prises."]
        self.dates = [(None, "2022-10-01"), ("2022-09-01", "2022-10-31"), ("2022-10-31", None), (None, None)]


class TestRAG(Test):
    def __init__(self):
        super().__init__()
        self.agent = RAGInference(args=self.args)


class TestTRAG(Test):
    def __init__(self):
        super().__init__()
        self.agent = TRAGInference(args=self.args)


class Benchmark(Demo):
    def __init__(self):
        super().__init__()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        self.questions = [
            "Quelle est la couleur choisie (RAL) pour les châssis ?",
            "Liste des décisions prises concernant le carrelage des salles des bains (SDBs) et les dates (jour/mois/année) auxquelles elles ont été prises.",
            "Quelle est la date de remise des espaces communs ?"
        ]
        self.dates = [("2023-01-01", "2023-02-01"), ("2023-02-15", "2023-12-31"), ("2024-01-01", None), (None, None)]


class BenchmarkRAG(Benchmark):
    def __init__(self):
        super().__init__()
        self.agent = RAGInference(args=self.args)


class BenchmarkTRAG(Benchmark):
    def __init__(self):
        super().__init__()
        self.agent = TRAGInference(args=self.args)


class InteractiveQueryRAG(Demo):
    def __init__(self):
        super().__init__()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"

        self.agent = RAGInference(args=self.args)

        self.questions = []
        self.dates = []

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
            self.dates.append((start_date, end_date))

            response = input("Do you want to ask another question? (y/[n]): ")
            if response == '' or response == 'n':
                break


class InteractiveQueryTRAG(InteractiveQueryRAG):
    def __init__(self):
        super().__init__()
        self.agent = TRAGInference(args=self.args)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    scenario = BenchmarkTRAG()
    scenario.run()
