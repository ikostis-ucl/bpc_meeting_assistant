import os
import warnings
from datetime import datetime

from app.engine.gui.gui import GUI
from app.engine.inference.groq_inference import GroqInference as InferenceEngine
from app.utils.app_utils import pprint_hline, pprint_console
from configuration import config_parser

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pprint_hline("=")
    pprint_console(f"Chat started at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    parser = config_parser()
    args = parser.parse_args()

    if args.anon:
        pprint_console("Running in --anon mode.")
        args.input_path = "./data/input_anonymised"
        args.storage_dir = "./data/vector_db_anonymised"

    agent = InferenceEngine(args=args)
    app = GUI(args=args, conv_agent=agent)
    app.run()

    pprint_console(f"Chat finished at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    pprint_hline('=')
