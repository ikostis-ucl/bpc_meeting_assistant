from app.engine.data_processing.index_storage import Storage
from configuration import config_parser


class IndexingFull:
    """
    Full indexing implementation for processing all documents.
    Creates and manages vector storage for the complete document set.
    """

    def __init__(self):
        """
        Initialize full indexing configuration.
        Sets up paths for input documents and vector storage.
        """
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"

        # Initialize storage agent and run indexing
        self.agent = Storage(args=self.args)
        self.agent.run()


class IndexingTest:
    """
    Test indexing implementation for processing test documents.
    Creates and manages vector storage for test document subset.
    """

    def __init__(self):
        """
        Initialize test indexing configuration.
        Sets up paths for test documents and test vector storage.
        """
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.args.input_path = "data/tests"
        self.args.storage_dir = "data/tests/vector_db_test"

        # Initialize storage agent and run indexing
        self.agent = Storage(args=self.args)
        self.agent.run()
