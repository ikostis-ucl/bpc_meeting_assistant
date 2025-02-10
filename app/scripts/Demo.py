import time
import sys

from app.utils.app_utils import pprint_qa, pprint_hline
from configuration import config_parser


class Demo:
    def __init__(self):
        self.questions = []
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

            pprint_hline(token="=")

            elapsed_time = time.time() - start_time
            remainder = elapsed_time % 60
            sleep_time = 60 - remainder
            if sleep_time > 0:
                for remaining in range(int(sleep_time), 0, -1):
                    sys.stdout.write("\r")
                    sys.stdout.write(f"Waiting for {remaining} seconds...")
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\rWaiting complete!                \n")
