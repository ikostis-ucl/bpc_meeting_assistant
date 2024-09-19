import os
import re
import shutil

import cv2
import easyocr
import fitz
import nest_asyncio
import numpy as np
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from tqdm import tqdm
from unidecode import unidecode

from app.engine.data_processing.data_loaders import load_index
from app.utils.app_utils import pprint_console, simplify_path, empty_dir, fmt_string, Color, pprint_error
from app.utils.data_processing_utils import is_match


class Storage:
    def __init__(self, args):
        nest_asyncio.apply()

        self.args = args

        # Document parser
        self.doc_parser = LlamaParse(api_key=args.llama_parse_key,
                                     verbose=False,
                                     language='fr')

        # Index
        self.index = None

        self.args = args

        self.embedding_model = HuggingFaceEmbedding(model_name=args.embeddings_model)

        self.node_parser = MarkdownNodeParser()

        os.makedirs(self.args.storage_dir, exist_ok=True)

    def directory_confirmation(self, skip=False):
        if skip:
            self.index = load_index(args=self.args, transformations=[self.node_parser])
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
                    self.index = load_index(args=self.args, transformations=[self.node_parser])
                    break
                else:
                    pprint_error("Invalid input. Please reply with 'l' for loading the index "
                                 "or 'p' for purging it (default option: 'l').")

    def parse_documents(self):
        self.doc_parser = LlamaParse(api_key=self.args.llama_parse_key,
                                     result_type="markdown",
                                     verbose=False,
                                     language='fr',
                                     max_timeout=600,
                                     use_vendor_multimodal_model=True,
                                     vendor_multimodal_model_name='openai-gpt4o',
                                     vendor_multimodal_api_key=self.args.openai_api_key
                                     )

        Settings.embed_model = self.embedding_model

        if os.path.isdir(self.args.input_path):
            file_paths = [simplify_path(os.path.join(self.args.input_path, f))
                          for f in os.listdir(self.args.input_path)
                          if os.path.isfile(os.path.join(self.args.input_path, f))]
        else:
            file_paths = [self.args.input_path]

        if not file_paths:
            pprint_console("The input folder is empty. Exiting...")
            exit()

        file_paths = tqdm(file_paths)
        for f_path in file_paths:
            fname_no_xt = os.path.basename(f_path).removesuffix(".pdf")
            file_paths.set_description(f"Processing file {fname_no_xt}")

            dir_path = os.path.dirname(f_path)

            pdf_document = fitz.open(f_path)

            # From the first page, extract the date and time of the meeting, and the abbreviations of the attendees.
            first_page = pdf_document.load_page(0)
            first_page_pixmap = first_page.get_pixmap(dpi=450, colorspace='gray')
            first_page_image = np.frombuffer(first_page_pixmap.samples, dtype=np.uint8).reshape(
                first_page_pixmap.height,
                first_page_pixmap.width,
                first_page_pixmap.n)
            first_page_image = cv2.fastNlMeansDenoising(src=first_page_image)
            _, first_page_image = cv2.threshold(first_page_image, 170, 255, cv2.THRESH_BINARY)

            # Apply OCR on the first page
            ocr_reader = easyocr.Reader(['fr'], verbose=False)
            ocr_result = []
            ocr_res_temp = ocr_reader.readtext(first_page_image, width_ths=1.5)
            for text_box in ocr_res_temp:
                [upper_left, upper_right, lower_left, _], text = text_box[0], text_box[1]
                x, y = int(upper_left[0]), int(upper_left[1])
                w, h = int(upper_right[0]) - x, int(lower_left[1]) - y
                ocr_result.append([x, y, w, h, text])

            ocr_result = sorted(ocr_result, key=lambda x: x[1])

            h_error_margin = int(np.mean([line[3] for line in ocr_result]) * 0.2)
            w_error_margin = int(np.mean([line[2] for line in ocr_result]) * 0.2)
            page_height = first_page_image.shape[0]
            y_upper_limit, y_lower_limit = int(page_height * (126 / 1333)), first_page_image.shape[1]
            roi_date_y_upper_limit, roi_date_y_lower_limit, roi_date_x_right_limit = 0, 0, 0

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

            # Save available abbreviations and document date
            d_time = []
            for bbox in ocr_result:
                if y_lower_limit >= bbox[1] >= y_upper_limit:
                    if (roi_date_y_lower_limit >= bbox[1] >= roi_date_y_upper_limit and
                            bbox[0] >= roi_date_x_right_limit):
                        d_time.append(bbox)

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

            documents = self.doc_parser.load_data(f_path)

            if documents:
                # Attach document date onto the document as metadata
                for doc in documents:
                    doc.metadata = {
                        'meeting_datetime': formatted_datetime,
                        'file_path': f_path,
                        'file_name': os.path.basename(f_path)
                    }
                    doc.excluded_embed_metadata_keys = ["meeting_datetime"]
                    doc.excluded_llm_metadata_keys = ["meeting_datetime"]
                    doc.id_ = f'{fname_no_xt.lower().replace(" ", "_")}'

                if self.index is None:
                    self.index = VectorStoreIndex.from_documents(documents=documents,
                                                                 transformations=[self.node_parser],
                                                                 embed_model=self.embedding_model,
                                                                 show_progress=False)

                    self.index.storage_context.persist(persist_dir=self.args.storage_dir)

                else:
                    # TODO: Move storage to index here and test
                    doc_ref_ids = list(set([doc.ref_doc_id for doc in self.index.docstore.docs.values()]))
                    for doc in documents:
                        if doc.id_ in doc_ref_ids:
                            self.index.delete_ref_doc(ref_doc_id=doc.id_, delete_from_docstore=True)
                            doc_ref_ids.remove(doc.id_)
                        self.index.insert(document=doc)

                    self.index.storage_context.persist(persist_dir=self.args.storage_dir)

                shutil.move(src=f_path, dst=f"{dir_path}/_completed/")

    def run(self):
        self.directory_confirmation()

        self.parse_documents()

        pprint_console("Finished storing index.")
