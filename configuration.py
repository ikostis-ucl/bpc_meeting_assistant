"""
Ioannis KOSTIS
ioannis.kostis@uclouvain.be | ioannis.aris.kostis@gmail.com

Config object that handles the various parameters and configurations of the system.
"""

import os

import configargparse
from dotenv import load_dotenv

load_dotenv(dotenv_path='./app/assets/.env/.env')


def config_parser():
    parser = configargparse.ArgumentParser(
        description="BPC_Group_meeting_assistant.")
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # I/O params
    parser.add_argument('--input_path', type=str,
                        default="./data/input",
                        help='File/Directory input path.')
    parser.add_argument('--storage_dir', type=str,
                        default='./data/vector_db',
                        help='Vector DB directory path.')

    parser.add_argument("--home_path", type=str, default=None,
                        help="Home path of the user. Used in GUI.")

    # Embeddings
    parser.add_argument('--embeddings_model', type=str, default="OrdalieTech/Solon-embeddings-large-0.1",
                        help='Embedding model.')
    parser.add_argument('--embeddings_cache_dir', type=str, default="./resources/.emb_models",
                        help='Embeddings cache directory path.')

    # Retriever
    parser.add_argument('--time_freq', type=int, default=5,
                        help='Duration of timespans, counted in months.')
    parser.add_argument('--cutoff_percentage', type=float, default=0.05,
                        help="Percentage of retrieved nodes to contribute as context to the answer.")

    # Deployment
    parser.add_argument('--prod', action='store_true', help='Run in production mode')

    # API Keys
    parser.add_argument('--llama_parse_key', type=str, default=os.getenv('LLAMA_PARSE_KEY'),
                        help='Your LlamaParse token key (llx-<...>)')
    parser.add_argument('--openai_api_key', type=str, default=os.getenv('OPENAI_API_KEY'),
                        help='Your OPENAI API token key (sk-<...>)')
    parser.add_argument('--groq_api_key', type=str, default=os.getenv('GROQ_API_KEY'),
                        help='Your Groq API token key gsk_<...>)')

    # Memory management
    parser.add_argument('--memory_monitor', action='store_true',
                        help='Enable memory monitoring')
    parser.add_argument('--cache_clear_interval', type=int, default=60,
                        help='Interval in seconds for clearing Gradio cache')

    return parser
