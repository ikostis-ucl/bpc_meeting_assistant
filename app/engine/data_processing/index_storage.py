import os
import shutil
import tempfile
import warnings

import fitz
import nest_asyncio
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from tqdm import tqdm

from app.engine.data_processing.data_loaders import load_index
from app.engine.data_processing.metadata_extraction.InvolvedPartiesExtractor import InvolvedPartiesExtractor
from app.engine.data_processing.metadata_extraction.TitleNodeFilter import TitleNodeFilter
from app.engine.data_processing.metadata_extraction.parse_first_page_llm import parse_first_page
from app.utils.app_utils import pprint_console, simplify_path, empty_dir, fmt_string, Color, pprint_error
from app.utils.data_processing_utils import datetime_to_timestamp


class Storage:
    """
    A class for handling document storage, parsing, and indexing operations.

    This class manages PDF document processing, OCR, metadata extraction, and vector storage indexing.
    It handles document parsing, date extraction, and storage of processed documents.
    """

    def __init__(self, args):
        """
        Initialize the Storage instance with necessary components.

        Args:
            args: Configuration arguments containing API keys, model paths, and other settings.
        """

        nest_asyncio.apply()
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.args = args

        self.doc_parser = LlamaParse(api_key=self.args.llama_parse_key,
                                     result_type="markdown",
                                     verbose=False,
                                     language='fr',
                                     max_timeout=600,
                                     use_vendor_multimodal_model=False,  # Change this to True to use a VLM/MMLM
                                     # vendor_multimodal_model_name='openai-gpt4o',
                                     # vendor_multimodal_api_key=self.args.openai_api_key
                                     )

        self.index = None

        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                    cache_folder=args.embeddings_cache_dir)
        # TODO: Remove this if unnecessary
        # from llama_index.llms.groq import Groq
        # self.llm = Groq(model=args.groq_model_indexing_kw,
        #                 api_key=args.groq_api_key,
        #                 model_kwargs={"seed": 42}, temperature=0.0)

        self.node_parser = MarkdownNodeParser()
        self.title_filter = TitleNodeFilter()

        os.makedirs(self.args.storage_dir, exist_ok=True)

    def directory_confirmation(self, skip=False):
        """
        Handle storage directory confirmation and loading existing index if present.

        Args:
            skip (bool): If True, skip confirmation and load existing index.
        """
        if skip:
            self.index, _ = load_index(args=self.args, transformations=[self.node_parser])
            return

        if not empty_dir(self.args.storage_dir):
            while True:
                user_input = input(fmt_string(Color.BLUE,
                                              "An index of your documents already exists. "
                                              "Do you want to (p)urge it, or (l)oad it? (['l']/'p'): ")).strip().lower()
                if user_input == 'p':
                    shutil.rmtree(self.args.storage_dir)
                    os.makedirs(self.args.storage_dir, exist_ok=True)
                    break
                elif user_input == 'l' or user_input == '':
                    self.index, _ = load_index(args=self.args, transformations=[self.node_parser, self.title_filter])
                    break
                else:
                    pprint_error("Invalid input. Please reply with 'l' for loading the index "
                                 "or 'p' for purging it (default option: 'l').")

    def parse_documents(self):
        """
        Parse and process PDF documents, extracting metadata and storing in the index.

        This method handles:
        - Document loading and preprocessing
        - Date and participant extraction using LLM
        - Document indexing and storage
        - Skipping already indexed documents
        """
        Settings.embed_model = self.embedding_model

        if os.path.isdir(self.args.input_path):
            file_paths = [simplify_path(os.path.join(self.args.input_path, f))
                          for f in os.listdir(self.args.input_path)
                          if os.path.isfile(os.path.join(self.args.input_path, f)) and f.lower().endswith('.pdf')]
        else:
            file_paths = [self.args.input_path]

        if not file_paths:
            pprint_console("The input folder is empty. Exiting...")
            exit()

        already_indexed_files = set()
        doc_ref_ids = []
        if self.index is not None:
            doc_ref_ids = list(set([doc.ref_doc_id for doc in self.index.docstore.docs.values()]))
            for doc_id in doc_ref_ids:
                already_indexed_files.add(f"{doc_id}.pdf")

        file_paths = tqdm(file_paths)
        for f_path in file_paths:
            fname = os.path.basename(f_path)
            fname_no_xt = fname.removesuffix(".pdf")
            file_paths.set_description(f"Processing file {fname_no_xt}")
            doc_id = fname_no_xt.lower().replace(" ", "_")

            if fname.lower() in already_indexed_files or doc_id in doc_ref_ids:
                pprint_console(f"File {fname} already indexed, skipping...")
                continue

            # Process first page to extract date and involved parties
            formatted_datetime, involved_parties = parse_first_page(file_path=f_path,
                                                                    llm_name=self.args.groq_model_indexing_extraction,
                                                                    llm_key=self.args.groq_api_key,
                                                                    doc_parser=self.doc_parser)

            if formatted_datetime is None:
                pprint_console(f"Could not find date on file {fname_no_xt}.pdf, skipping...")
                continue

            # Process rest of document
            pdf_document = fitz.open(f_path)
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as rest_pdf:
                temp_pdf_path = rest_pdf.name
                rest_pdf_doc = fitz.open()
                for page_num in range(1, pdf_document.page_count):
                    rest_pdf_doc.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                rest_pdf_doc.save(rest_pdf)

                rest_pdf.flush()

                documents = self.doc_parser.load_data(temp_pdf_path)

            if documents:
                # Attach metadata
                for page_num, doc in enumerate(documents, start=2):
                    doc.metadata = {
                        'meeting_datetime': datetime_to_timestamp(formatted_datetime),
                        'file_path': f"{self.args.input_path}/{fname}",
                        'file_name': fname,
                        'page_number': page_num
                    }
                    doc.excluded_embed_metadata_keys = ["meeting_datetime", "file_path", "file_name", "page_number"]
                    doc.excluded_llm_metadata_keys = ["meeting_datetime", "file_path", "file_name", "page_number"]
                    doc.id_ = doc_id

                if self.index is None:
                    self.index = VectorStoreIndex.from_documents(documents=documents,
                                                                 transformations=[self.node_parser,
                                                                                  self.title_filter,
                                                                                  InvolvedPartiesExtractor(
                                                                                      entities=involved_parties)],
                                                                 embed_model=self.embedding_model,
                                                                 show_progress=False)

                    self.index.storage_context.persist(persist_dir=self.args.storage_dir)
                    doc_ref_ids.append(doc_id)
                    already_indexed_files.add(fname.lower())

                else:
                    for doc in documents:
                        self.index.insert(document=doc)

                    self.index.storage_context.persist(persist_dir=self.args.storage_dir)
                    doc_ref_ids.append(doc_id)
                    already_indexed_files.add(fname.lower())

        pprint_console("All documents processed and indexed.")

    def run(self):

        self.directory_confirmation()

        self.parse_documents()

        pprint_console("Finished storing index.")
