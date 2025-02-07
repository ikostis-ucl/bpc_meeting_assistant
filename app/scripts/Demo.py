import time

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

            results = self.agent.query_llm(query_string=question)
            pprint_qa(question, results)
            elapsed_time = time.time() - start_time
            if elapsed_time < 20:
                time.sleep(20 - elapsed_time)  # Failsafe for the Cohere API rate limit
