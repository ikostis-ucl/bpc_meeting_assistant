import os

from halo import Halo
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.utils.app_utils import fmt_string, Color, pprint_console, pprint_error


def load_index(args, transformations=None):
    """
    Load the index from storage.

    Args:
        args: Command line arguments containing storage directory and embedding model details.
        transformations: Optional transformations to apply to the settings.

    Returns:
        Tuple containing the index and a tuple of minimum and maximum timestamps.

    Raises:
        SystemExit: If the storage directory is not found.
    """
    # Set the embedding model and cache folder
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                cache_folder=args.embeddings_cache_dir)
    if transformations is not None:
        Settings.transformations = transformations
    Settings.show_progress = False

    # Check if the storage directory exists
    if os.path.exists(args.storage_dir):
        with Halo(text=f"{fmt_string(Color.CYAN, '[CONSOLE] Loading index from storage...')}",
                  placement='right', animation='bounce', spinner='moon'):
            storage_context = StorageContext.from_defaults(persist_dir=args.storage_dir)
            index = load_index_from_storage(storage_context, show_progress=False)
        pprint_console("Loaded index from storage.")

        # Extract timestamps from the index
        _timestamps = [node.metadata.get('meeting_datetime') for node in index.docstore.docs.values()]
        return index, (min(_timestamps), max(_timestamps))
    else:
        pprint_error("Index storage directory not found. Parse and store the index first.")
        exit()
