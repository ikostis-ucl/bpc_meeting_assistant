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
    parser.add_argument('--input', type=str,
                        default="./data/input",
                        help='File/Directory input path.')
    parser.add_argument('--output_path', type=str,
                        default="./data/output/",
                        help='Output directory path.')
    parser.add_argument('--storage_dir', type=str,
                        default='./data/vector_db',
                        help='Vector DB directory path.')

    parser.add_argument("--home_path", type=str, default=None,
                        help="Home path of the user. Used in GUI.")

    # Embeddings
    parser.add_argument('--embeddings_model', type=str, default="OrdalieTech/Solon-embeddings-large-0.1",
                        help='Embedding model.')

    return parser
