from app.engine.inference.groq_inference import GroqInference
from app.scripts.Demo import Demo


class Test(Demo):
    def __init__(self):
        super().__init__()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        self.questions = [
            "Quelles finitions pour les halls d'entrée ?"]

class TestRAG(Test):
    def __init__(self):
        super().__init__()
        self.agent = GroqInference(args=self.args)
