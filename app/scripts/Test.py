from app.engine.inference.groq_inference import GroqInference
from app.scripts.Demo import Demo


class Test(Demo):
    """
    Test class for system evaluation with a single test question.
    Inherits from Demo class and provides minimal test configuration.
    """

    def __init__(self):
        """
        Initialize test configuration.
        Sets up paths and defines single test question.
        """
        super().__init__()
        self.args.input_path = "data/tests"
        self.args.storage_dir = "data/tests/vector_db_test"
        self.questions = [
            "Peux-je avoir une liste remarques SECO ?"
        ]


class TestRAG(Test):
    """
    RAG-specific implementation of the test system.
    Uses GroqInference for processing the test question.
    """

    def __init__(self):
        """Initialize test with Groq inference agent."""
        super().__init__()
        self.agent = GroqInference(args=self.args)
