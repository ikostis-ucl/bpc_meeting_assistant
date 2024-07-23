import os
import pickle
import re
import shutil

import fitz
import nest_asyncio
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
from tqdm import tqdm

from app.utils.app_utils import pprint_console, simplify_path, empty_dir
from app.engine.data_processing.data_loaders import load_index


class Storage:
    def __init__(self, args):
        nest_asyncio.apply()

        self.args = args
        self.transformations = []

        # Document parser
        self.doc_parser = LlamaParse(api_key=args.llama_parse_key,
                                     verbose=False,
                                     language='fr')

        # Embeddings
        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model)

        # Text LLM
        self.llm = Groq(model="llama3-70b-8192", api_key=args.groq_api_key,
                        model_kwargs={"seed": 42}, temperature=0.0)

        # Node Parser
        self.node_parser = MarkdownNodeParser(llm=self.llm, num_workers=8)

        # Index
        self.index = None

        os.makedirs(self.args.storage_dir, exist_ok=True)

    def directory_confirmation(self):
        if empty_dir(self.args.storage_dir):
            self.index = {}
        else:
            while True:
                user_input = input("An index of your documents already exists. "
                                   "Do you want to (p)urge it, or (l)oad it? (['l']/'p'): ").strip().lower()
                if user_input == 'p':
                    shutil.rmtree(self.args.storage_dir)
                    os.makedirs(self.args.storage_dir, exist_ok=True)
                    self.index = {}
                    break
                elif user_input == 'l' or user_input == '':
                    self.index = load_index(self.args)
                    break
                else:
                    print("Invalid input. Please reply with 'l' for loading the index "
                          "or 'p' for purging it (default option: 'l').")

    def apply_transformations(self):
        pass

    def parse_documents_omni(self):
        """
        This uses gpt-4o. It works way better, and it has integrated multimodal capabilities.
        It costs!
        """

        self.doc_parser = LlamaParse(api_key=self.args.llama_parse_key,
                                     result_type="markdown",
                                     verbose=False,
                                     language='fr',
                                     gpt4o_mode=True,
                                     gpt4o_api_key=self.args.openai_api_key
                                     )

        if os.path.isdir(self.args.input_data):
            file_paths = [simplify_path(os.path.join(self.args.input_data, f))
                          for f in os.listdir(self.args.input_data)
                          if os.path.isfile(os.path.join(self.args.input_data, f))]
        else:
            file_paths = [self.args.input_data]

        nodes = []

        for fpath in tqdm(file_paths, desc=f"Parsing files"):
            pages_dir = f"{self.args.storage_dir}/pdf_pages"
            os.makedirs(pages_dir, exist_ok=True)

            fname_no_xt = os.path.basename(fpath).removesuffix(".pdf")

            pdf_document = fitz.open(fpath)

            # TODO: From first page, extract the date of the document.

            # TODO: Remove per page parsing.

            # Iterate through each page and create a new PDF for each page
            for page_number in range(len(pdf_document)):
                pdf_writer = fitz.open()  # Create a new PDF in memory
                pdf_writer.insert_pdf(pdf_document, from_page=page_number, to_page=page_number)

                # Create the output PDF file path
                output_pdf_path = f"{pages_dir}/{fname_no_xt}_page_{page_number + 1:02d}.pdf"

                # Save the single page to the new PDF
                pdf_writer.save(output_pdf_path)
                pdf_writer.close()

            # Close the input PDF
            pdf_document.close()

            p_paths = [simplify_path(os.path.join(pages_dir, f))
                       for f in os.listdir(pages_dir)
                       if os.path.isfile(os.path.join(pages_dir, f))]
            p_paths.sort()
            for p_path in tqdm(p_paths, desc="Parsing pages", leave=True):
                document = self.doc_parser.load_data(file_path=p_path)[0]

                pattern = rf'{pages_dir}/{fname_no_xt}_page_(\d+)\.pdf'
                match = re.search(pattern, p_path)
                if match:
                    pg_num = str(match.group(1))
                else:
                    pg_num = ""

                # TODO: Add date
                # TODO: Remove page
                document.metadata = {
                    'pg_num': pg_num,
                    'file_path': fpath,
                    'file_name': os.path.basename(fpath)
                }

                nodes.extend(self.node_parser.get_nodes_from_documents(documents=[document]))

            shutil.rmtree(pages_dir)

        return nodes

    def metadata_extraction(self, nodes):
        raise NotImplementedError

    def store_index(self, nodes):
        pprint_console("Storing index...")
        docstore = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(docstore=docstore)

        _index = VectorStoreIndex(nodes=nodes, storage_context=storage_context,
                                  embed_model=self.embedding_model, show_progress=True)
        storage_context.persist(persist_dir=self.args.storage_dir)

    def run(self):
        self.directory_confirmation()
        self.apply_transformations()

        cache_dir = f"{self.args.storage_dir}/cache"
        os.makedirs(cache_dir, exist_ok=True)

        nodes = self.parse_documents_omni()

        self.store_index(nodes)

        pprint_console("Finished storing index.")
