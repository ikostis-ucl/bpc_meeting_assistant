from app.engine.inference.rag_inference import RAGInference
from app.scripts.Demo import Demo


class Test(Demo):
    def __init__(self):
        super().__init__()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        self.questions = [
            "Quelles sont les décisions prises en matière des stations du recharge des véhicules électriques ? Indiquez-moi les dates (jj/mm/aaaa) auxquelles ces décisions ont été prises."]

class TestRAG(Test):
    def __init__(self):
        super().__init__()
        self.agent = RAGInference(args=self.args)
