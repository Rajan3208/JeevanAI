from PIL import Image
from io import BytesIO
import os
from tempfile import NamedTemporaryFile
from PyPDF2 import PdfReader
import concurrent.futures

# Import OCR libraries only when needed
def import_ocr_libraries():
    try:
        import pytesseract
        from easyocr import Reader
        from langchain_community.document_loaders import UnstructuredImageLoader, UnstructuredFileLoader
        return True
    except ImportError:
        print("Warning: Some OCR libraries are not available")
        return False

def extract_text_with_pypdf(pdf_file):
    """Extract text using PyPDF2"""
    pdf_reader = PdfReader(pdf_file)
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

def extract_text_with_pytesseract(list_dict_final_images):
    """Extract text from images using PyTesseract"""
    from pytesseract import image_to_string
    
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)

def extract_text_with_easyocr(list_dict_final_images):
    """Extract text from images using EasyOCR"""
    from easyocr import Reader
    
    language_reader = Reader(["en"])
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    # Process images in smaller batches
    batch_size = 3  # Process fewer images at once to reduce memory usage
    
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        batch_content = []
        
        for image_bytes in batch:
            image = Image.open(BytesIO(image_bytes))
            raw_text = language_reader.readtext(image)
            raw_text = "\n".join([res[1] for res in raw_text])
            batch_content.append(raw_text)
            
        image_content.extend(batch_content)

    return "\n".join(image_content)

def extract_text_with_langchain(pdf_file):
    """Extract text using Langchain's UnstructuredFileLoader"""
    from langchain_community.document_loaders import UnstructuredFileLoader
    
    loader = UnstructuredFileLoader(pdf_file)
    documents = loader.load()
    pdf_pages_content = '\n'.join(doc.page_content for doc in documents)
    return pdf_pages_content

def extract_text_with_langchain_image(list_dict_final_images):
    """Extract text from images using Langchain's UnstructuredImageLoader"""
    from langchain_community.document_loaders import UnstructuredImageLoader
    
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

def extract_text_parallel(pdf_path, images, use_ocr=True):
    """Extract text using multiple methods in parallel for speed"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Always run PyPDF extraction
        pdf_future = executor.submit(extract_text_with_pypdf, pdf_path)
        
        futures = []
        
        # Optionally run OCR methods
        if use_ocr and import_ocr_libraries():
            try:
                # Start OCR tasks in parallel
                tesseract_future = executor.submit(extract_text_with_pytesseract, images)
                easyocr_future = executor.submit(extract_text_with_easyocr, images)
                langchain_future = executor.submit(extract_text_with_langchain, pdf_path)
                langchain_img_future = executor.submit(extract_text_with_langchain_image, images)
                
                futures = [tesseract_future, easyocr_future, langchain_future, langchain_img_future]
            except Exception as e:
                print(f"Error setting up OCR extraction: {e}")
        
        # Get results
        results['pypdf'] = pdf_future.result()
        
        # Get OCR results if available
        if futures:
            try:
                results['pytesseract'] = tesseract_future.result()
                results['easyocr'] = easyocr_future.result()
                results['langchain_pdf'] = langchain_future.result()
                results['langchain_img'] = langchain_img_future.result()
            except Exception as e:
                print(f"Error in OCR processing: {e}")
    
    # Combine all available text
    combined_text = " ".join(results.values())
    
    return results, combined_text