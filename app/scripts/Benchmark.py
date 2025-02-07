from app.engine.inference.rag_inference import RAGInference
from app.scripts.Demo import Demo


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
