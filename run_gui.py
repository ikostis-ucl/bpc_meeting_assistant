from datetime import datetime

from app.engine.gui.gui import GUI
from app.engine.inference.rag_inference import RAGInference as InferenceEngine
from app.utils.app_utils import pprint_hline, pprint_console
from configuration import config_parser

if __name__ == "__main__":
    pprint_hline("=")
    pprint_console(f"Chat started at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    parser = config_parser()
    args = parser.parse_args()

    # args.input_path = "data/tests"
    # args.storage_dir = "data/tests/vector_db_test"

    agent = InferenceEngine(args=args)
    app = GUI(args=args, conv_agent=agent)
    app.run()

    pprint_console(f"Chat finished at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    pprint_hline('=')
