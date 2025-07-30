from app.engine.inference.groq_inference import GroqInference
from app.scripts.Demo import Demo
from app.utils.app_utils import pprint_console

class Benchmark(Demo):
    """
    Benchmark class for evaluating the system with predefined questions.
    Inherits from Demo class and provides a standard set of test questions.
    """

    def __init__(self):
        """Initialize benchmark configuration with predefined questions and paths."""
        super().__init__()

        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        if self.args.anon:
            pprint_console("Running in --anon mode.")
            self.args.input_path = "./data/input_anonymised"
            self.args.storage_dir = "./data/vector_db_anonymised"

        # Standard set of benchmark questions covering different aspects
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
    """
    RAG-specific implementation of the benchmark system.
    Uses GroqInference for query processing.
    """

    def __init__(self):
        """Initialize benchmark with Groq inference agent."""
        super().__init__()
        self.agent = GroqInference(args=self.args)
