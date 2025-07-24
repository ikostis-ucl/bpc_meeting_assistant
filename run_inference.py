import os
import warnings

# from app.scripts.Test import TestRAG as Scenario
from app.scripts.Benchmark import BenchmarkRAG as Scenario

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    scenario = Scenario()
    scenario.run()
