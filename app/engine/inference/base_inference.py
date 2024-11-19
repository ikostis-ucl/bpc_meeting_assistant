from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from app.engine.data_processing.data_loaders import load_index


class BaseInference:
    def __init__(self, args):
        self.model = Groq(model="llama-3.1-70b-versatile", api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.0)
        self.index = load_index(args)
        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                    cache_folder=args.embeddings_cache_dir)
        self.prompt_template = PromptTemplate(
            "Vous êtes un(e) assistant(e) qui aide un chef de projet à extraire des informations de documents qui "
            "incluent le déroulement de réunions autour d'un certain projet. Il s'agit de la requête du chef "
            "de projet:\n"
            "Requête: {query_string}\n"
            "Répondez de manière aussi cohérente que possible. Votre réponse doit être rédigée en français."
        )
