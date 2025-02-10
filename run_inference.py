import os
import warnings

from app.scripts.Test import TestRAG
from app.scripts.Benchmark import BenchmarkRAG

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    scenario = BenchmarkRAG()
    scenario.run()
