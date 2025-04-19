import os
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredImageLoader

# Initialize global imports lazily to improve startup time
_pytesseract = None
_easyocr_reader = None

def _get_pytesseract():
    global _pytesseract
    if _pytesseract is None:
        import pytesseract
        _pytesseract = pytesseract
    return _pytesseract

def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        from easyocr import Reader
        _easyocr_reader = Reader(["en"])
    return _easyocr_reader

def extract_text_with_pyPDF(pdf_file):
    """Extract text from PDF using PyPDF2"""
    try:
        pdf_reader = PdfReader(pdf_file)
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        return raw_text
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")
        return ''

def extract_text_with_langchain_pdf(pdf_file):
    """Extract text from PDF using Langchain PDF loader"""
    try:
        loader = UnstructuredFileLoader(pdf_file)
        documents = loader.load()
        pdf_pages_content = '\n'.join(doc.page_content for doc in documents)
        return pdf_pages_content
    except Exception as e:
        print(f"Error extracting text with Langchain PDF: {e}")
        return ''

def extract_text_with_pytesseract(list_dict_final_images):
    """Extract text from images using pytesseract OCR"""
    try:
        pytesseract = _get_pytesseract()
        image_list = [list(data.values())[0] for data in list_dict_final_images]
        image_content = []

        for index, image_bytes in enumerate(image_list):
            image = Image.open(BytesIO(image_bytes))
            raw_text = str(pytesseract.image_to_string(image))
            image_content.append(raw_text)

        return "\n".join(image_content)
    except Exception as e:
        print(f"Error extracting text with pytesseract: {e}")
        return ''

def extract_text_with_easyocr(list_dict_final_images):
    """Extract text from images using EasyOCR"""
    try:
        reader = _get_easyocr_reader()
        image_list = [list(data.values())[0] for data in list_dict_final_images]
        image_content = []

        for index, image_bytes in enumerate(image_list):
            image = Image.open(BytesIO(image_bytes))
            raw_text = reader.readtext(image)
            raw_text = "\n".join([res[1] for res in raw_text])
            image_content.append(raw_text)

        return "\n".join(image_content)
    except Exception as e:
        print(f"Error extracting text with EasyOCR: {e}")
        return ''

def extract_text_with_langchain_image(list_dict_final_images):
    """Extract text from images using Langchain image loader"""
    try:
        image_list = [list(data.values())[0] for data in list_dict_final_images]
        image_content = []

        for index, image_bytes in enumerate(image_list):
            with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_file_path = temp_file.name

            loader = UnstructuredImageLoader(temp_file_path)
            data = loader.load()
            raw_text = data[0].page_content if data else ''
            image_content.append(raw_text)

            os.remove(temp_file_path)

        return "\n".join(image_content)
    except Exception as e:
        print(f"Error extracting text with Langchain image: {e}")
        return ''
