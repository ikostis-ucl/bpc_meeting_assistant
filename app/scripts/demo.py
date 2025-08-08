from app.utils.app_utils import pprint_hline
from app.utils.inference_utils import pprint_qa
from configuration import config_parser


class Demo:
    """
    Base demonstration class for the question-answering system.
    Provides framework for running queries and displaying results.
    """

    def __init__(self):
        """
        Initialize demo configuration.
        Sets up argument parser and empty question list.
        """
        self.questions = []
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.agent = None

    def inputs(self):
        """
        Placeholder method for handling input questions.
        To be implemented by child classes.
        """
        ...

    def run(self):
        """
        Execute the demonstration workflow.
        Processes each question and displays results with formatting.
        """
        self.inputs()

        # Process each question and display results
        for question in self.questions:
            results = self.agent.query_llm(query_string=question)
            pprint_qa(question, results)
            pprint_hline(token="=")
