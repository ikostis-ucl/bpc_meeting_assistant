from app.engine.data_processing.index_storage import Storage
from configuration import config_parser


class IndexingFull:
    def __init__(self):
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"

        self.agent = Storage(args=self.args)

        self.agent.run()


class IndexingTest:
    def __init__(self):
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.args.input_path = "data/tests"
        self.args.storage_dir = "data/tests/vector_db_test"

        self.agent = Storage(args=self.args)

        self.agent.run()


if __name__ == "__main__":
    indexing_run = IndexingTest()
