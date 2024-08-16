from llama_index.llms.groq import Groq

from app.engine.inference.base_inference import BaseInference


class GroqInference(BaseInference):
    def __init__(self, args):
        super().__init__(args)
        # llama3-70b-8192
        self.model = Groq(model="llama-3.1-70b-versatile", api_key=args.groq_api_key,
                          model_kwargs={"seed": 42}, temperature=0.1)
