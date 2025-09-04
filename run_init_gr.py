from app.engine.guardrails.domains_extractor import DomainsExtractor
from app.utils.app_utils import pprint_console
from configuration import config_parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    if args.anon:
        pprint_console("Running in --anon mode.")
        args.input_path = "./data/input_anonymised"
        args.storage_dir = "./data/vector_db_anonymised"

    extractor = DomainsExtractor(args)

    _domains = extractor.run_extraction()

    pprint_console("Domain extraction completed!")
