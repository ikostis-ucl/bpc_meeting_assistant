from app.engine.guardrails.extract_thematics import DomainsExtractor
from app.utils.app_utils import pprint_console
from configuration import config_parser


def main():
    """Main function to run keyword extraction."""

    # Parse arguments
    parser = config_parser()
    args = parser.parse_args()

    if args.anon:
        pprint_console("Running in --anon mode.")
        args.input_path = "./data/input_anonymised"
        args.storage_dir = "./data/vector_db_anonymised"

    # Initialize extractor
    extractor = DomainsExtractor(args)

    # Run extraction
    keywords = extractor.run_extraction()

    pprint_console("Keyword extraction completed!")
    return keywords


if __name__ == "__main__":
    main()
