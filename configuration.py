"""
Ioannis KOSTIS
ioannis.kostis@uclouvain.be | ioannis.aris.kostis@gmail.com

Config object that handles the various parameters and configurations of the system.
"""

import configargparse


def config_parser():
    parser = configargparse.ArgumentParser(
        description="BPC_Group_meeting_assistant.")
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--keys", is_config_file=True, required=False,
                        help='Path to the API keys file.',
                        default='./app/assets/.api_keys/keys.txt')

    # I/O params
    parser.add_argument('--input_path', type=str,
                        default="./data/input/tests",
                        help='File/Directory input path.')
    parser.add_argument('--temp_path', type=str,
                        default="./data/temp/",
                        help='Temporary files directory path.')
    parser.add_argument('--storage_dir', type=str,
                        default='./data/vector_db_test',
                        help='Vector DB directory path.')

    parser.add_argument("--home_path", type=str, default=None,
                        help="Home path of the user. Used in GUI.")

    # Embeddings
    parser.add_argument('--embeddings_model', type=str, default="OrdalieTech/Solon-embeddings-large-0.1",
                        help='Embedding model.')

    # Retriever
    parser.add_argument('--cutoff_percentage', type=float, default=0.05,
                        help="Percentage of retrieved nodes to contribute as context to the answer.")

    # API Keys
    parser.add_argument('--llama_parse_key', type=str, help='Your LlamaParse token key (llx-<...>)')
    parser.add_argument('--openai_api_key', type=str, help='Your OPENAI API token key (sk-<...>)')
    parser.add_argument('--groq_api_key', type=str, help='Your Groq API token key gsk_<...>)')
    parser.add_argument('--cohere_api_key', type=str, help='Your Cohere API key <...>)')

    return parser
