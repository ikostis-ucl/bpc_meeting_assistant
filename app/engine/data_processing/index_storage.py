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

from app.engine.data_processing.data_loaders import load_index
from app.utils.app_utils import pprint_console, simplify_path, empty_dir, fmt_string, Color
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

        # self.llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=args.groq_api_key,
        #                 model_kwargs={"seed": 42}, temperature=0.0)

        self.node_parser = MarkdownNodeParser()

        os.makedirs(self.args.storage_dir, exist_ok=True)

    def directory_confirmation(self):
        if not empty_dir(self.args.storage_dir):
            while True:
                user_input = input(fmt_string(Color.GREEN,
                                              "An index of your documents already exists. "
                                              "Do you want to (p)urge it, or (l)oad it? (['l']/'p'): ")).strip().lower()
                if user_input == 'p':
                    shutil.rmtree(self.args.storage_dir)
                    os.makedirs(self.args.storage_dir, exist_ok=True)
                    break
                elif user_input == 'l' or user_input == '':
                    self.index = load_index(self.args)
                    break
                else:
                    print("Invalid input. Please reply with 'l' for loading the index "
                          "or 'p' for purging it (default option: 'l').")

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

        if os.path.isdir(self.args.input_path):
            file_paths = [simplify_path(os.path.join(self.args.input_path, f))
                          for f in os.listdir(self.args.input_path)
                          if os.path.isfile(os.path.join(self.args.input_path, f))]
        else:
            file_paths = [self.args.input_path]

        data = []

        for fpath in tqdm(file_paths, desc=f"Parsing documents"):
            fname_no_xt = os.path.basename(fpath).removesuffix(".pdf")

            pdf_document = fitz.open(fpath)

            """
            From the first page, extract the date and time of the meeting, and the abbreviations of the attendees.
            """
            first_page = pdf_document.load_page(0)
            first_page_pixmap = first_page.get_pixmap(dpi=450, colorspace='gray')
            first_page_image = np.frombuffer(first_page_pixmap.samples, dtype=np.uint8).reshape(
                first_page_pixmap.height,
                first_page_pixmap.width,
                first_page_pixmap.n)
            first_page_image = cv2.fastNlMeansDenoising(src=first_page_image)
            _, first_page_image = cv2.threshold(first_page_image, 170, 255, cv2.THRESH_BINARY)

            ocr_reader = easyocr.Reader(['fr'])
            ocr_result = []
            ocr_res_temp = ocr_reader.readtext(first_page_image, width_ths=0.7)
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

            # Define the Region of Interest (RoI) of the document (general, datetime, abbreviations)
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

            # Save available abbreviations and document date/time
            datetime = []
            for bbox in ocr_result:
                if y_lower_limit >= bbox[1] >= y_upper_limit:
                    if (roi_date_y_lower_limit >= bbox[1] >= roi_date_y_upper_limit and
                            bbox[0] >= roi_date_x_right_limit):
                        datetime.append(bbox[4])

            datetime_sting = ' '.join(x for x in datetime)
            datetime_sting = datetime_sting.lower().replace("o", "0")
            pattern = r'(\d{2}/\d{2}/\d{4}), de \d{1,2}h\d{0,2}\w*\s*(?:a|à\s*)?(\d{1,2})h(\d{2})'
            match = re.search(pattern, datetime_sting)
            if match:
                date = match.group(1)
                hour = match.group(2).zfill(2)
                minute = match.group(3).zfill(2)
                formatted_datetime = f"{date} {hour}:{minute}"
            else:
                raise ValueError("Unable to extract date from document.")

            documents = self.doc_parser.load_data(fpath)

            # Attach document date/time and available abbreviations onto the document as metadata
            for doc in documents:
                doc.metadata = {
                    'meeting_datetime': formatted_datetime,
                    'file_path': fpath,
                    'file_name': os.path.basename(fpath)
                }
                doc.excluded_llm_metadata_keys = ["meeting_datetime", "file_path", "file_name", "entities"]
                doc.id_ = f'{fname_no_xt.lower().replace(" ", "_")}'

            data.extend(documents)

        shutil.rmtree(f"{self.args.storage_dir}/pdfs")

        Settings.embed_model = self.embedding_model
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(data,
                                                         transformations=[self.node_parser],
                                                         embed_model=self.embedding_model,
                                                         show_progress=True)
        else:
            raise NotImplementedError("Updatable graph index is not yet implemented.")
            # self.index = index
            # self.index.insert_nodes(data)

    def run(self):
        self.directory_confirmation()

        temp_dir = f"{self.args.storage_dir}/pdfs"
        os.makedirs(temp_dir, exist_ok=True)

        self.parse_documents_omni()

        self.index.storage_context.persist(persist_dir=self.args.storage_dir)

        pprint_console("Finished storing index.")
