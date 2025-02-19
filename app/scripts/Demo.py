from app.utils.app_utils import pprint_hline
from app.utils.inference_utils import pprint_qa
# from app.utils.inference_utils import throttle_cross_query_requests
from configuration import config_parser


class Demo:
    def __init__(self):
        self.questions = []
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.agent = None

    def inputs(self):
        ...

    # @throttle_cross_query_requests
    def run(self):
        self.inputs()

        for question in self.questions:
            results = self.agent.query_llm(query_string=question)
            pprint_qa(question, results)
            pprint_hline(token="=")
