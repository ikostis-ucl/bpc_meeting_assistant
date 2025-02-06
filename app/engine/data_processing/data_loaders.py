import os

from halo import Halo
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.utils.app_utils import fmt_string, Color, pprint_console, pprint_error


def load_index(args, transformations=None):
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                cache_folder=args.embeddings_cache_dir)
    if transformations is not None:
        Settings.transformations = transformations
    Settings.show_progress = False

    if os.path.exists(args.storage_dir):
        with Halo(text=f"{fmt_string(Color.YELLOW, '[CONSOLE] Loading index from storage...')}",
                  placement='right', animation='bounce', spinner='moon'):
            storage_context = StorageContext.from_defaults(persist_dir=args.storage_dir)
            index = load_index_from_storage(storage_context, show_progress=False)
        pprint_console("Loaded index from storage.")

        _timestamps = [node.metadata.get('meeting_datetime') for node in index.docstore.docs.values()]
        return index, (min(_timestamps), max(_timestamps))
    else:
        pprint_error("Index storage directory not found. Parse and store the index first.")
        exit()
