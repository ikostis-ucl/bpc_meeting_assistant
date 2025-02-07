from app.engine.inference.rag_inference import RAGInference
from app.scripts.Demo import Demo


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
