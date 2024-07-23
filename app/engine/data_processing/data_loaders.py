import os

from halo import Halo
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.utils.app_utils import fmt_string, Color, pprint_console, pprint_error


def load_index(args):
    # Settings.embed_model is needed to load the index.
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embeddings_model)

    if os.path.exists(args.storage_dir):
        with Halo(text=f"{fmt_string(Color.BLUE, 'Loading index from storage...')}",
                  placement='right', animation='bounce', spinner='moon'):
            storage_context = StorageContext.from_defaults(persist_dir=args.storage_dir)
            index = load_index_from_storage(storage_context)
        pprint_console("Loaded index from storage.")
        return index, storage_context
    else:
        pprint_error("Index storage directory not found. Parse and store the index first.")
        exit()
