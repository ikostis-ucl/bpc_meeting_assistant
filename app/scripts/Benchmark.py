from app.engine.inference.groq_inference import GroqInference
from app.scripts.Demo import Demo


class Benchmark(Demo):
    def __init__(self):
        super().__init__()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        self.questions = [
            "Quelle est la couleur choisie (RAL) pour les châssis ?",
            "Liste des décisions prises concernant le carrelage des salles des bains (SDBs) et les dates (jour/mois/année) auxquelles elles ont été prises.",
            "Quelle est la date de remise des espaces communs ?",
            "Quelles sont les décisions qui ont étés prises pour les acrotères des terrasses ?",
            "Pourrais-je avoir toutes les informations concernant l'ascenseur vélo ?",
            "Pourrais-je avoir un historique concernant les décisions prises pour les couvre-murs ?",
            "Informations concernant les faux-plafonds ?",
            "Quelles finitions pour les halls d'entrée ?",
            "Quelle isolation a été choisi pour les plafonds du sous-sols -1 ?",
            "Peux-je avoir une liste remarques SECO ?"
        ]


class BenchmarkRAG(Benchmark):
    def __init__(self):
        super().__init__()
        self.agent = GroqInference(args=self.args)
