from app.engine.inference.rag_inference import RAGInference
from app.scripts.Demo import Demo


class InteractiveQueryRAG(Demo):
    def __init__(self):
        super().__init__()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"

        self.agent = RAGInference(args=self.args)

        self.questions = []

    def inputs(self):
        while True:
            question = input("Question: ")

            self.questions.append(question)

            response = input("Do you want to ask another question? (y/[n]): ")
            if response == '' or response == 'n':
                break
