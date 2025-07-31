from app.engine.inference.groq_inference import GroqInference
from app.scripts.Demo import Demo
from app.utils.app_utils import pprint_console
from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX


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
        self.questions = list(BENCHMARK_QUESTIONS_INDEX.values())
        # TODO: override the .run() method to process the questions index and compare with GT.


class BenchmarkRAG(Benchmark):
    """
    RAG-specific implementation of the benchmark system.
    Uses GroqInference for query processing.
    """

    def __init__(self):
        """Initialize benchmark with Groq inference agent."""
        super().__init__()
        self.agent = GroqInference(args=self.args)
