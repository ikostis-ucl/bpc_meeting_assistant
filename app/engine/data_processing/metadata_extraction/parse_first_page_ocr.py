import re

import cv2
import easyocr
import fitz
import numpy as np
from unidecode import unidecode

from app.utils.data_processing_utils import is_match


def parse_first_page(file_path):
    """
    Process the first page of a PDF document to extract datetime and involved parties.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        tuple: (formatted_datetime, involved_parties) where:
            - formatted_datetime (list): Date in [dd, mm, yyyy] format
            - involved_parties (list): List of abbreviations of involved parties
    """

    ocr_reader = easyocr.Reader(['fr'], verbose=False)

    # Open PDF and process first page
    pdf_document = fitz.open(file_path)
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
    ocr_res_temp = ocr_reader.readtext(first_page_image, width_ths=1.5)
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
        return formatted_datetime, involved_parties
    else:
        return None, involved_parties
