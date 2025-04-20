import os
import uuid
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from multiprocessing import Process, Queue
from utils.pdf_processor import convert_pdf_to_images
from utils.text_extractor import (
    extract_text_with_pyPDF,
    extract_text_with_langchain_pdf,
    extract_text_with_pytesseract,
    extract_text_with_easyocr
)
from utils.insights_generator import generate_comprehensive_insights
from utils.model_handler import load_models, upload_to_gcs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Adjust origins in production

# Configuration
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', '/tmp/uploads')
app.config['MODELS_FOLDER'] = os.environ.get('MODELS_FOLDER', 'models')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Results cache for async processing
results_cache = {}

# Model cache for loaded models
model_cache = {}

# Initialize models function
def initialize_models():
    try:
        logger.info("Loading models...")
        global model_cache
        model_cache = load_models(app.config['MODELS_FOLDER'])
        logger.info(f"Loaded {len(model_cache)} models successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Call model initialization during startup
@app.before_request
def ensure_models_loaded():
    global model_cache
    if not model_cache:
        try:
            initialize_models()
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'models_loaded': bool(model_cache),
        'pending_jobs': len(results_cache)
    }), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if not (file.filename.lower().endswith('.pdf') or 
            file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'File must be a PDF or image (PNG, JPG)'}), 400
    
    try:
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        logger.info(f"Saving file to {file_path}")
        file.save(file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"File not saved: {file_path}")
            return jsonify({'error': 'Failed to save file'}), 500
            
        logger.info(f"File uploaded: {unique_filename}")
        return jsonify({
            'message': 'File uploaded successfully',
            'file_id': unique_filename
        }), 200
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed failed upload: {file_path}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

def process_file_async(file_id, file_path, result_queue=None):
    """Process file asynchronously and store results in cache"""
    try:
        logger.info(f"Starting async processing for {file_id}")
        results_cache[file_id]['status'] = 'processing'
        
        is_image = file_path.lower().endswith(('.png', '.jpg', '.jpeg'))
        pypdf_text = ""
        langchain_pdf_text = ""
        pytesseract_text = ""
        easyocr_text = ""
        combined_text = ""
        
        if is_image:
            logger.info(f"Processing image file: {file_id}")
            images = [file_path]
            
            try:
                logger.info("Starting Pytesseract extraction")
                pytesseract_text = extract_text_with_pytesseract(images)
                logger.info(f"Pytesseract extraction completed with {len(pytesseract_text)} chars")
            except Exception as e:
                logger.error(f"Pytesseract extraction failed: {str(e)}", exc_info=True)
                
            try:
                results_cache[file_id]['status'] = 'extracting_with_easyocr'
                logger.info("Starting EasyOCR extraction")
                easyocr_text = extract_text_with_easyocr(images)
                logger.info(f"EasyOCR extraction completed with {len(easyocr_text)} chars")
            except Exception as e:
                logger.error(f"EasyOCR extraction failed: {str(e)}", exc_info=True)
            
            combined_text = pytesseract_text + "\n" + easyocr_text
            
            if len(combined_text.strip()) < 50:
                logger.warning(f"Very little text extracted from image: {file_id}")
        else:
            logger.info(f"Processing PDF file: {file_id}")
            
            try:
                results_cache[file_id]['status'] = 'extracting_with_pypdf'
                logger.info("Starting PyPDF extraction")
                pypdf_text = extract_text_with_pyPDF(file_path)
                logger.info(f"PyPDF extraction completed with {len(pypdf_text)} chars")
            except Exception as e:
                logger.error(f"PyPDF extraction failed: {str(e)}", exc_info=True)
            
            if len(pypdf_text.strip()) > 200:
                logger.info(f"Using PyPDF extraction for {file_id}")
                combined_text = pypdf_text
            else:
                logger.info(f"PyPDF extraction insufficient, trying Langchain for {file_id}")
                try:
                    results_cache[file_id]['status'] = 'extracting_with_langchain'
                    logger.info("Starting Langchain extraction")
                    langchain_pdf_text = extract_text_with_langchain_pdf(file_path)
                    logger.info(f"Langchain extraction completed with {len(langchain_pdf_text)} chars")
                except Exception as e:
                    logger.error(f"Langchain extraction failed: {str(e)}", exc_info=True)
                
                if len(langchain_pdf_text.strip()) > 200:
                    combined_text = langchain_pdf_text
                else:
                    logger.info(f"Text extraction failed, using OCR for {file_id}")
                    try:
                        results_cache[file_id]['status'] = 'extracting_with_ocr'
                        logger.info("Converting PDF to images for OCR")
                        images = convert_pdf_to_images(file_path)
                        logger.info("Starting Pytesseract extraction for PDF")
                        pytesseract_text = extract_text_with_pytesseract(images)
                        logger.info("Starting EasyOCR extraction for PDF")
                        easyocr_text = extract_text_with_easyocr(images)
                        combined_text = pytesseract_text + "\n" + easyocr_text
                    except Exception as e:
                        logger.error(f"OCR extraction failed: {str(e)}", exc_info=True)
                        combined_text = pypdf_text + "\n" + langchain_pdf_text
        
        if len(combined_text.strip()) < 10:
            logger.error(f"Very little text extracted from file: {file_id}")
            results_cache[file_id] = {
                'status': 'error',
                'error': 'Could not extract sufficient text from document'
            }
            if result_queue:
                result_queue.put(results_cache[file_id])
            return
            
        results_cache[file_id]['status'] = 'extracting_insights'
        
        logger.info(f"Generating insights for {file_id} with {len(combined_text)} characters of text")
        insights = generate_comprehensive_insights(
            combined_text, 
            file_path, 
            None, 
            model_cache,
            timeout=180
        )
        
        if not insights or not isinstance(insights, dict):
            logger.warning(f"Insights generation returned invalid data for {file_id}")
            insights = {
                'summary': "Unable to generate detailed insights from this document.",
                'entities': {},
                'topics': [],
                'sentiment': 'NEUTRAL'
            }

        result = {
            'file_id': file_id,
            'text_length': len(combined_text),
            'extraction_method': 'optimized',
            'insights': insights.get('summary', "No summary available"),
            'entities': list(insights.get('entities', {}).keys())[:10],
            'topics': insights.get('topics', [])[:3],
            'sentiment': insights.get('sentiment', 'NEUTRAL'),
            'key_phrases': insights.get('key_phrases', [])[:7]
        }

        if 'transformer_summary' in insights and insights['transformer_summary']:
            result['detailed_summary'] = insights['transformer_summary']

        results_cache[file_id] = {
            'status': 'complete',
            'result': result
        }

        logger.info(f"Processing complete for {file_id}")
        if result_queue:
            result_queue.put(results_cache[file_id])

    except Exception as e:
        logger.error(f"Error processing {file_id}: {str(e)}", exc_info=True)
        results_cache[file_id] = {
            'status': 'error',
            'error': str(e)
        }
        if result_queue:
            result_queue.put(results_cache[file_id])
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {str(e)}")

@app.route('/api/analyze/<file_id>', methods=['GET'])
def analyze_file(file_id):
    if file_id in results_cache:
        cache_entry = results_cache[file_id]
        status = cache_entry.get('status')
        
        if status == 'complete':
            return jsonify(cache_entry['result']), 200
        elif status == 'error':
            return jsonify({'error': cache_entry.get('error', 'Unknown error'), 'file_id': file_id}), 500
        else:
            return jsonify({
                'status': 'processing', 
                'stage': status if status != 'processing' else 'extracting_text',
                'file_id': file_id
            }), 202
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found', 'file_id': file_id}), 404
    
    results_cache[file_id] = {'status': 'initializing'}
    
    # Use multiprocessing for async processing
    result_queue = Queue()
    process = Process(target=process_file_async, args=(file_id, file_path, result_queue))
    process.daemon = True
    process.start()
    
    # For synchronous processing (uncomment to use instead of multiprocessing):
    # process_file_async(file_id, file_path)
    # cache_entry = results_cache[file_id]
    # if cache_entry['status'] == 'complete':
    #     return jsonify(cache_entry['result']), 200
    # elif cache_entry['status'] == 'error':
    #     return jsonify({'error': cache_entry.get('error', 'Unknown error'), 'file_id': file_id}), 500
    # else:
    #     return jsonify({'status': 'processing', 'stage': cache_entry['status'], 'file_id': file_id}), 202
    
    return jsonify({'status': 'processing', 'file_id': file_id}), 202

@app.route('/api/status/<file_id>', methods=['GET'])
def check_status(file_id):
    if file_id in results_cache:
        status = results_cache[file_id].get('status')
        response = {'status': status, 'file_id': file_id}
        
        if status == 'error':
            response['error'] = results_cache[file_id].get('error', 'Unknown error')
        elif status == 'complete':
            response['result'] = results_cache[file_id].get('result')
            
        return jsonify(response), 200
    
    return jsonify({'status': 'not_found', 'file_id': file_id}), 404

@app.route('/api/upload-to-gcs', methods=['POST'])
def upload_models_to_gcs():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
            
        bucket_name = data.get('bucket_name')
        
        if not bucket_name:
            return jsonify({'error': 'Bucket name is required'}), 400
        
        uploaded_files = upload_to_gcs(
            app.config['MODELS_FOLDER'], 
            bucket_name
        )
        
        return jsonify({
            'message': 'Models uploaded to GCS successfully',
            'uploaded_files': uploaded_files
        }), 200
    except Exception as e:
        logger.error(f"GCS upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_cache():
    try:
        removed = 0
        current_size = len(results_cache) 
        
        if current_size > 150:
            keys_to_remove = list(results_cache.keys())[:-100]
            for key in keys_to_remove:
                del results_cache[key]
                removed += 1
        
        return jsonify({
            'message': f'Cache cleanup complete. Removed {removed} entries.',
            'remaining': len(results_cache) 
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_models()
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
