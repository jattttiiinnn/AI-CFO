# core/utils.py

import io
import json
import PyPDF2
import google.generativeai as genai
from django.conf import settings

# Initialize Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)

# 1. Extract text from PDF
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF file (given as bytes).
    """
    text = ""
    with io.BytesIO(file_bytes) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

# 2. Call AI Model (OpenAI example)
def call_ai_model(prompt, model="gemini-pro"):
    """
    Calls Gemini API with a given prompt and returns text response.
    """
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)

    if hasattr(response, "text"):
        return response.text
    return str(response)