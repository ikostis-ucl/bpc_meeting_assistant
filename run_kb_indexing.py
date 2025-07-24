import os
import warnings

# from app.scripts.Indexing import IndexingTest as IndexingRoutine
from app.scripts.Indexing import IndexingFull as IndexingRoutine

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    indexing_run = IndexingRoutine()
