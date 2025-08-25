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

    # GUI
    parser.add_argument("--home_path", type=str, default=None,
                        help="Home path of the user. Used in GUI.")

    # Benchmarking
    parser.add_argument('--benchmark_mode', action='store_true',
                        help='Run in benchmark evaluation mode')
    parser.add_argument('--benchmark_gt_path', type=str,
                        default="eval/gt/meeting_minutes_publication_gt.csv",
                        help='Path to ground truth CSV file')

    # Embeddings
    parser.add_argument('--embeddings_model', type=str, default="OrdalieTech/Solon-embeddings-large-0.1",
                        help='Embedding model.')
    parser.add_argument('--embeddings_cache_dir', type=str, default="./resources/.emb_models",
                        help='Embeddings cache directory path.')

    # Guardrails
    parser.add_argument('--disable_guardrails', action='store_true',
                        help="Use the argument to bypass the input guardrails during inference.",)
    parser.add_argument('--thematics_storage_path', type=str,
                        default="./app/engine/guardrails/archive/meeting_assistant_thematics.json",
                        help='Storage path for the thematics file.')

    # Groq LLM API
    parser.add_argument('--groq_model_inference', type=str,
                        default="llama-3.3-70b-versatile",
                        help='Model name for Groq inference.')
    parser.add_argument('--groq_model_inference_tpm', type=int, default=300000,
                        help='Tokens per minute for Groq model.')

    parser.add_argument('--groq_model_inference_judge', type=str,
                        default="llama-3.3-70b-versatile",
                        help="Groq model name used in assessing similarity between consecutive answers.")

    parser.add_argument('--groq_model_indexing_extraction', type=str,
                        default="llama-3.3-70b-versatile",
                        help='Groq model name used in information extraction while indexing.')
    parser.add_argument('--groq_model_indexing_extraction_tpm', type=int, default=300000,
                        help='Tokens per minute for Groq model.')
    parser.add_argument('--groq_model_indexing_kw', type=str,
                        default="llama-3.1-8b-instant",
                        help='Groq model name used in information extraction while indexing.')
    parser.add_argument('--groq_model_indexing_kw_tpm', type=int, default=250000,
                        help='Tokens per minute for Groq model.')

    parser.add_argument('--groq_model_gr', type=str,
                        default='meta-llama/llama-4-maverick-17b-128e-instruct',
                        help='Groq model name used in guardrails.')
    parser.add_argument('--groq_model_gr_tpm', type=int, default=300000,
                        help='Tokens per minute for Groq model used in guardrails.')

    # Retriever
    parser.add_argument('--time_freq', type=int, default=5,
                        help='Duration of timespans, counted in months.')

    # Deployment
    parser.add_argument('--prod', action='store_true', help='Run in production mode')
    parser.add_argument('--anon', action='store_true', help='Run in anonymized mode')

    # API Keys
    parser.add_argument('--llama_parse_key', type=str, default=os.getenv('LLAMA_PARSE_KEY'),
                        help='Your LlamaParse token key (llx-<...>)')
    parser.add_argument('--openai_api_key', type=str, default=os.getenv('OPENAI_API_KEY'),
                        help='Your OPENAI API token key (sk-<...>)')
    parser.add_argument('--groq_api_key', type=str, default=os.getenv('GROQ_API_KEY'),
                        help='Your Groq API token key gsk_<...>)')

    return parser
