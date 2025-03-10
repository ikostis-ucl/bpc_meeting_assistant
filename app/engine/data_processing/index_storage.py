import os
import re
import shutil
import tempfile
import warnings

import cv2
import easyocr
import fitz
import nest_asyncio
import numpy as np
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
from tqdm import tqdm
from unidecode import unidecode

from app.engine.data_processing.data_loaders import load_index
from app.engine.data_processing.metadata_extractors import InvolvedPartiesExtractor, KeywordExtractor
from app.utils.app_utils import pprint_console, simplify_path, empty_dir, fmt_string, Color, pprint_error
from app.utils.data_processing_utils import is_match, datetime_to_timestamp


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

        # Apply async support and suppress warnings
        nest_asyncio.apply()
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.args = args

        # Initialize document parser with LlamaParse
        self.doc_parser = LlamaParse(api_key=self.args.llama_parse_key,
                                     result_type="markdown",
                                     verbose=False,
                                     language='fr',
                                     max_timeout=600,
                                     use_vendor_multimodal_model=False, # Change this to True to use a VLM/MMLM
                                     vendor_multimodal_model_name='openai-gpt4o',
                                     vendor_multimodal_api_key=self.args.openai_api_key
                                     )

        self.index = None

        # Initialize OCR, embedding model, and language model
        self.ocr_reader = easyocr.Reader(['fr'], verbose=False)
        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model,
                                                    cache_folder=args.embeddings_cache_dir)
        self.llm = Groq(model="gemma2-9b-it", api_key=args.groq_api_key,
                        model_kwargs={"seed": 42}, temperature=0.0)

        self.node_parser = MarkdownNodeParser()

        # Ensure storage directory exists
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

        # Check if directory is not empty and handle user choice
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
                    self.index, _ = load_index(args=self.args, transformations=[self.node_parser])
                    break
                else:
                    pprint_error("Invalid input. Please reply with 'l' for loading the index "
                                 "or 'p' for purging it (default option: 'l').")

    def parse_documents(self):
        """
        Parse and process PDF documents, extracting metadata and storing in the index.

        This method handles:
        - Document loading and preprocessing
        - OCR text extraction
        - Date and participant extraction
        - Document indexing and storage
        """
        Settings.embed_model = self.embedding_model

        # Get list of files to process
        if os.path.isdir(self.args.input_path):
            file_paths = [simplify_path(os.path.join(self.args.input_path, f))
                          for f in os.listdir(self.args.input_path)
                          if os.path.isfile(os.path.join(self.args.input_path, f))]
        else:
            file_paths = [self.args.input_path]

        if not file_paths:
            pprint_console("The input folder is empty. Exiting...")
            exit()

        # Process each file
        file_paths = tqdm(file_paths)
        for f_path in file_paths:
            fname_no_xt = os.path.basename(f_path).removesuffix(".pdf")
            file_paths.set_description(f"Processing file {fname_no_xt}")

            dir_path = os.path.dirname(f_path)

            # Open PDF and process first page
            pdf_document = fitz.open(f_path)

            # From the first page, extract the date and time of the meeting, and the abbreviations of the attendees.
            first_page = pdf_document.load_page(0)

            # Prepare first page image for OCR
            first_page_pixmap = first_page.get_pixmap(dpi=450, colorspace='gray')
            first_page_image = np.frombuffer(first_page_pixmap.samples, dtype=np.uint8).reshape(
                first_page_pixmap.height,
                first_page_pixmap.width,
                first_page_pixmap.n)
            first_page_image = cv2.fastNlMeansDenoising(src=first_page_image)
            _, first_page_image = cv2.threshold(first_page_image, 170, 255, cv2.THRESH_BINARY)

            # Extract text using OCR
            ocr_result = []
            ocr_res_temp = self.ocr_reader.readtext(first_page_image, width_ths=1.5)
            for text_box in ocr_res_temp:
                [upper_left, upper_right, lower_left, _], text = text_box[0], text_box[1]
                x, y = int(upper_left[0]), int(upper_left[1])
                w, h = int(upper_right[0]) - x, int(lower_left[1]) - y
                ocr_result.append([x, y, w, h, text])

            # Sort OCR results and calculate margins
            ocr_result = sorted(ocr_result, key=lambda x: x[1])
            h_error_margin = int(np.mean([line[3] for line in ocr_result]) * 0.2)
            w_error_margin = int(np.mean([line[2] for line in ocr_result]) * 0.2)
            page_height = first_page_image.shape[0]

            # Define regions of interest
            y_upper_limit, y_lower_limit = int(page_height * (126 / 1333)), first_page_image.shape[1]
            roi_date_y_upper_limit, roi_date_y_lower_limit, roi_date_x_right_limit = 0, 0, 0
            roi_poi_y_upper_limit, roi_poi_x_right_limit, roi_poi_x_left_limit = 0, 0, 0

            # Process and index documents

            # Define the Region of Interest (RoI) of the document (general & date)
            for bbox in ocr_result:
                if is_match(text=bbox[4], phrase="intervenant de diffuser", threshold=60):
                    y_lower_limit = bbox[1] - h_error_margin
                    break

            for bbox in ocr_result:
                if y_lower_limit >= bbox[1] >= y_upper_limit:
                    if is_match(text=bbox[4], phrase="Place, Date, Heure", threshold=60):
                        roi_date_y_upper_limit = bbox[1] - h_error_margin
                        roi_date_y_lower_limit = bbox[1] + bbox[3] + h_error_margin
                        roi_date_x_right_limit = bbox[0] + bbox[2] + w_error_margin
                        break

            for bbox in ocr_result:
                if y_lower_limit >= bbox[1] >= y_upper_limit:
                    if is_match(text=bbox[4], phrase="ABREV"):
                        roi_poi_y_upper_limit = bbox[1] + bbox[3] + h_error_margin
                        roi_poi_x_right_limit = bbox[0] + bbox[2] + w_error_margin
                        roi_poi_x_left_limit = bbox[0] - w_error_margin
                        break

            # Save available abbreviations and document date
            d_time = []
            for bbox in ocr_result:
                if y_lower_limit >= bbox[1] >= y_upper_limit:
                    if (roi_date_y_lower_limit >= bbox[1] >= roi_date_y_upper_limit and
                            bbox[0] >= roi_date_x_right_limit):
                        d_time.append(bbox)

            involved_parties = []
            for bbox in ocr_result:
                if y_lower_limit >= bbox[1] >= y_upper_limit:
                    if (bbox[1] >= roi_poi_y_upper_limit and
                            roi_poi_x_right_limit - bbox[2] >= bbox[0] >= roi_poi_x_left_limit):
                        involved_parties.append(bbox[4])

            # Format the date of the document
            d_time = sorted(d_time, key=lambda x: x[0])
            datetime_sting = ' '.join(x[4] for x in d_time)
            datetime_sting = unidecode(datetime_sting).lower().replace("o", "0")

            date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
            date_match = re.search(date_pattern, datetime_sting)
            if date_match:
                date = date_match.group()
                formatted_datetime = date.split('/')  # [dd, mm, yyyy]
            else:
                pprint_console(f"Could not find date on file {fname_no_xt}.pdf, skipping...")
                continue

            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as rest_pdf:
                temp_pdf_path = rest_pdf.name
                rest_pdf_doc = fitz.open()
                for page_num in range(1, pdf_document.page_count):
                    rest_pdf_doc.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                rest_pdf_doc.save(rest_pdf)

                rest_pdf.flush()

                documents = self.doc_parser.load_data(temp_pdf_path)

            if documents:
                # Attach document date onto the document as metadata
                for page_num, doc in enumerate(documents, start=2):
                    doc.metadata = {
                        'meeting_datetime': datetime_to_timestamp(formatted_datetime),
                        'file_path': f"{self.args.input_path}/files_archive/{os.path.basename(f_path)}",
                        'file_name': os.path.basename(f_path),
                        'page_number': page_num
                    }
                    doc.excluded_embed_metadata_keys = ["meeting_datetime"]
                    doc.excluded_llm_metadata_keys = ["meeting_datetime"]
                    doc.id_ = f'{fname_no_xt.lower().replace(" ", "_")}'

                if self.index is None:
                    self.index = VectorStoreIndex.from_documents(documents=documents,
                                                                 transformations=[self.node_parser,
                                                                                  KeywordExtractor(keywords=3,
                                                                                                   llm=self.llm,
                                                                                                   show_progress=False),
                                                                                  InvolvedPartiesExtractor(
                                                                                      entities=involved_parties)],
                                                                 embed_model=self.embedding_model,
                                                                 show_progress=False)

                    self.index.storage_context.persist(persist_dir=self.args.storage_dir)

                else:
                    doc_ref_ids = list(set([doc.ref_doc_id for doc in self.index.docstore.docs.values()]))
                    for doc in documents:
                        if doc.id_ in doc_ref_ids:
                            self.index.delete_ref_doc(ref_doc_id=doc.id_, delete_from_docstore=True)
                            doc_ref_ids.remove(doc.id_)
                        self.index.insert(document=doc)

                    self.index.storage_context.persist(persist_dir=self.args.storage_dir)

                shutil.move(src=f_path, dst=f"{dir_path}/_completed/")

    def run(self):
        """
        Execute the main storage workflow.

        This method coordinates the directory confirmation and document parsing process.
        """

        self.directory_confirmation()

        self.parse_documents()

        pprint_console("Finished storing index.")
