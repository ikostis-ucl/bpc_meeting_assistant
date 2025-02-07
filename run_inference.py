import os
import warnings

from app.scripts.Test import TestRAG

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    scenario = TestRAG()
    scenario.run()
