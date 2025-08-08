from app.engine.inference.groq_inference import GroqInference
from app.scripts.demo import Demo
from app.utils.app_utils import pprint_console

class InteractiveQueryRAG(Demo):
    """
    Interactive query system using RAG (Retrieval-Augmented Generation).
    Allows users to input questions dynamically and receive responses.
    """

    def __init__(self):
        """
        Initialize interactive query system.
        Sets up paths and Groq inference agent.
        """
        super().__init__()

        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        if self.args.anon:
            pprint_console("Running in --anon mode.")
            self.args.input_path = "./data/input_anonymised"
            self.args.storage_dir = "./data/vector_db_anonymised"

        self.agent = GroqInference(args=self.args)
        self.questions = []

    def inputs(self):
        """
        Handle interactive user input.
        Continuously accepts questions until user chooses to stop.
        """
        while True:
            # Get user question
            question = input("Question: ")
            self.questions.append(question)

            # Check if user wants to continue
            response = input("Do you want to ask another question? (y/[n]): ")
            if response == '' or response == 'n':
                break
