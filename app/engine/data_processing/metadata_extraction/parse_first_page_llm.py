import json
import re
import tempfile

import fitz
from llama_index.llms.groq import Groq
from pydantic import BaseModel, Field


def parse_first_page(file_path, llm_name, llm_key, doc_parser):
    """
    Process the first page of a PDF document to extract datetime and involved parties using LLM.

    This method uses LlamaParse and LlamaIndex with Groq for structured extraction.

    Args:
        file_path (str): Path to the PDF file
        llm_name (str): Name of the Groq LLM model to use
        llm_key (str): API key for Groq LLM
        doc_parser: LlamaParse document parser instance

    Returns:
        tuple: (formatted_datetime, involved_parties) where:
            - formatted_datetime (list): Date in [dd, mm, yyyy] format
            - involved_parties (list): List of abbreviations of involved parties
    """

    extraction_llm = Groq(model=llm_name,
                          api_key=llm_key,
                          model_kwargs={"seed": 42}, temperature=0.0)

    # Open PDF and extract just the first page to a temporary file
    pdf_document = fitz.open(file_path)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as first_page_pdf:
        temp_pdf_path = first_page_pdf.name
        first_page_doc = fitz.open()
        first_page_doc.insert_pdf(pdf_document, from_page=0, to_page=0)
        first_page_doc.save(first_page_pdf)
        first_page_pdf.flush()

        first_page_content = doc_parser.load_data(temp_pdf_path)

    if not first_page_content:
        return None, []

    # Pydantic model for information extraction
    class MeetingInfo(BaseModel):
        date: str = Field(description="Meeting date in DD/MM/YYYY format")
        involved_parties: list[str] = Field(
            description="List of abbreviations of parties involved in the meeting")

    text = first_page_content[0].text if first_page_content else ""

    extraction_prompt = f"""
    You are an expert at extracting structured information from construction meeting minutes.

    Extract the following information from the text:
    1. The meeting date in DD/MM/YYYY format
    2. The abbreviations of all parties involved in the meeting (usually listed as "ABREV")

    Only extract information that's explicitly stated in the text.
    If you can't find the date, return an empty string for date.
    If you can't find the parties, return an empty list.

    Format your response as valid JSON with these exact keys:
    {{
        "date": "DD/MM/YYYY",
        "involved_parties": ["ABBR1", "ABBR2", ...]
    }}

    Here's the text:
    {text}

    Return only the JSON object, with no additional explanation or text.
    """

    response = None
    try:
        response = extraction_llm.complete(extraction_prompt)

        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, response.text)

        if json_match:
            json_str = json_match.group()

            # Clean up potential LLM formatting issues
            json_str = json_str.replace("```json", "").replace("```", "").strip()

            extracted_data = json.loads(json_str)

            meeting_info = MeetingInfo(**extracted_data)

            date_str = meeting_info.date
            if date_str and re.match(r"\d{2}/\d{2}/\d{4}", date_str):
                formatted_datetime = date_str.split('/')
                return formatted_datetime, meeting_info.involved_parties
            else:
                return None, meeting_info.involved_parties
        else:
            # Fallback policy if JSON extraction fails
            print(f"Failed to extract JSON from response: {response.text[:100]}...")

            date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
            date_match = re.search(date_pattern, response.text)

            party_pattern = r'\b[A-Z]{2,5}\b'
            parties = re.findall(party_pattern, response.text)

            if date_match:
                date_str = date_match.group()
                formatted_datetime = date_str.split('/')
                return formatted_datetime, parties
            else:
                return None, parties

    except Exception as e:
        print(f"Error extracting information: {e}")
        if response is not None and hasattr(response, 'text'):
            print(f"Response excerpt: {response.text[:200]}")
        else:
            print("No response text available")
        return None, []
