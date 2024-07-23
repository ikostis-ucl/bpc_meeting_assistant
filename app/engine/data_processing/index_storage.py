import os
import re
import shutil

import cv2
import easyocr
import fitz
import nest_asyncio
import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
from tqdm import tqdm

from app.engine.data_processing.data_loaders import load_index
from app.utils.app_utils import pprint_console, simplify_path, empty_dir, fmt_string, Color
from app.utils.data_processing_utils import is_match


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
        self.index = VectorStoreIndex([], embed_model=self.embedding_model, show_progress=True)

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

        for fpath in tqdm(file_paths, desc=f"Parsing documents"):
            fname_no_xt = os.path.basename(fpath).removesuffix(".pdf")

            pdf_document = fitz.open(fpath)

            # TODO: From first page, extract the date and the entity index of the document.
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
            roi_entity_y_upper_limit, roi_entity_x_left_limit, roi_entity_x_right_limit = 0, 0, 0

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
                    if is_match(text=bbox[4], phrase="ABREV", threshold=100):
                        roi_entity_y_upper_limit = bbox[1] - h_error_margin
                        roi_entity_x_left_limit = bbox[0] - w_error_margin
                        roi_entity_x_right_limit = bbox[0] + bbox[2] + w_error_margin
                        break

            datetime = []
            entities = ["TOUS"]
            for bbox in ocr_result:
                if y_lower_limit >= bbox[1] >= y_upper_limit:
                    if (roi_date_y_lower_limit >= bbox[1] >= roi_date_y_upper_limit and
                            bbox[0] >= roi_date_x_right_limit):
                        datetime.append(bbox[4])
                    if (bbox[1] >= roi_entity_y_upper_limit and
                            bbox[0] >= roi_entity_x_left_limit and
                            bbox[0] + bbox[2] <= roi_entity_x_right_limit):
                        if bbox[4] == 'MOICOM':
                            entities.append("MO/COM")  # FIXME: EasyOCR issue, cannot recognize /
                        else:
                            entities.append(bbox[4])

            datetime_sting = ' '.join(x for x in datetime)
            datetime_sting = datetime_sting.lower().replace("o", "0")
            pattern = r"(\d{2}/\d{2}/\d{4}), de \d{1,2}h\d{2} à (\d{2})h(\d{2})"
            match = re.search(pattern, datetime_sting)
            if match:
                date = match.group(1)
                hour = match.group(2)
                minute = match.group(3)
                formatted_datetime_string = f"{date} {hour}:{minute}"
            else:
                raise RuntimeError("Unable to extract date from document.")

            # TODO: From the rest of the document, extract the text
            rest_doc = fitz.open()
            rest_doc.insert_pdf(pdf_document, from_page=1, to_page=pdf_document.page_count - 1)
            rest_doc_path = f"{self.args.storage_dir}/pdfs/{fname_no_xt}.pdf"
            rest_doc.save(rest_doc_path)
            rest_doc.close()
            pdf_document.close()

            documents = self.doc_parser.load_data(rest_doc_path)
            # TODO: Add entities and graph
            for doc in documents:
                doc.metadata = {
                    'datetime': formatted_datetime_string,
                    'file_path': fpath,
                    'file_name': os.path.basename(fpath)
                }
                doc.id_ = f'{fname_no_xt.lower().replace(" ", "_")}'
                self.index.update_ref_doc(doc, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}, )

            os.remove(rest_doc_path)

    def run(self):
        self.directory_confirmation()

        temp_dir = f"{self.args.storage_dir}/pdfs"
        os.makedirs(temp_dir, exist_ok=True)

        self.parse_documents_omni()

        self.index.storage_context.persist(persist_dir=self.args.storage_dir)

        pprint_console("Finished storing index.")
