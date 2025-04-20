import os
import uuid
import json
import logging
from threading import Thread
from flask import Flask, request, jsonify
import pickle
from werkzeug.utils import secure_filename
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

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Results cache for async processing
results_cache = {}

# Load models at startup
model_cache = {}

# Initialize models at startup
with app.app_context():
    try:
        logger.info("Loading models...")
        model_cache = load_models(app.config['MODELS_FOLDER'])
        logger.info(f"Loaded {len(model_cache)} models successfully")
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
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Handle both PDF and image files
    if not (file.filename.lower().endswith('.pdf') or 
            file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
        return jsonify({'error': 'File must be a PDF or image (PNG, JPG)'}), 400
    
    # Create a unique filename
    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    logger.info(f"File uploaded: {unique_filename}")
    
    try:
        # Return file ID for the next step
        return jsonify({
            'message': 'File uploaded successfully',
            'file_id': unique_filename
        }), 200
    except Exception as e:
        os.remove(file_path)  # Clean up on error
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_file_async(file_id, file_path):
    """Process file asynchronously and store results in cache"""
    try:
        logger.info(f"Starting async processing for {file_id}")
        results_cache[file_id]['status'] = 'processing'
        
        # Check if file is image or PDF
        is_image = file_path.lower().endswith(('.png', '.jpg', '.jpeg'))
        
        # For images, we only need OCR methods
        if is_image:
            logger.info(f"Processing image file: {file_id}")
            images = [file_path]  # For images, just use the file directly
            pypdf_text = ""
            langchain_pdf_text = ""
            pytesseract_text = extract_text_with_pytesseract(images)
            easyocr_text = extract_text_with_easyocr(images)
            
            combined_text = pytesseract_text + "\n" + easyocr_text
        else:
            # For PDFs, use optimized extraction approach
            logger.info(f"Processing PDF file: {file_id}")
            
            # Try PyPDF first as it's fastest
            pypdf_text = extract_text_with_pyPDF(file_path)
            
            # If PyPDF gets sufficient text, skip other methods
            if len(pypdf_text.strip()) > 200:  # Higher threshold for meaningful text
                logger.info(f"Using PyPDF extraction for {file_id}")
                combined_text = pypdf_text
            else:
                # Fall back to langchain
                logger.info(f"PyPDF extraction insufficient, trying Langchain for {file_id}")
                langchain_pdf_text = extract_text_with_langchain_pdf(file_path)
                
                if len(langchain_pdf_text.strip()) > 200:
                    combined_text = langchain_pdf_text
                else:
                    # Only if both fail, use OCR (most time-consuming)
                    logger.info(f"Text extraction failed, using OCR for {file_id}")
                    images = convert_pdf_to_images(file_path)
                    pytesseract_text = extract_text_with_pytesseract(images)
                    easyocr_text = extract_text_with_easyocr(images)
                    combined_text = pytesseract_text + "\n" + easyocr_text
        
        # Generate insights with a timeout
        logger.info(f"Generating insights for {file_id}")
        insights = generate_comprehensive_insights(
            combined_text, 
            file_path, 
            None,  # Only pass images if needed
            model_cache,
            timeout=60  # Set a timeout for inference
        )
        
        # Prepare response
        result = {
            'file_id': file_id,
            'text_length': len(combined_text),
            'extraction_method': 'optimized',
            'insights': insights['summary'],
            'entities': list(insights['entities'].keys())[:10] if 'entities' in insights else [],
            'topics': insights['topics'][:3] if 'topics' in insights else [],
            'sentiment': insights.get('sentiment', 'NEUTRAL')
        }
        
        # Update cache with completed result
        results_cache[file_id] = {
            'status': 'complete',
            'result': result
        }
        logger.info(f"Processing complete for {file_id}")
        
    except Exception as e:
        logger.error(f"Error processing {file_id}: {str(e)}")
        results_cache[file_id] = {
            'status': 'error',
            'error': str(e)
        }
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {str(e)}")

@app.route('/api/analyze/<file_id>', methods=['GET'])
def analyze_file(file_id):
    # Check if file is being processed or is complete
    if file_id in results_cache:
        cache_entry = results_cache[file_id]
        status = cache_entry.get('status')
        
        if status == 'complete':
            # Processing is complete, return result
            return jsonify(cache_entry['result']), 200
        elif status == 'error':
            # Error occurred during processing
            return jsonify({'error': cache_entry.get('error', 'Unknown error')}), 500
        else:
            # Still processing
            return jsonify({'status': 'processing', 'file_id': file_id}), 202
    
    # File hasn't been submitted for processing yet
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Initialize cache entry and start processing
    results_cache[file_id] = {'status': 'initializing'}
    
    # Start async processing
    thread = Thread(target=process_file_async, args=(file_id, file_path))
    thread.daemon = True  # Make thread a daemon so it doesn't block app shutdown
    thread.start()
    
    return jsonify({'status': 'processing', 'file_id': file_id}), 202

@app.route('/api/status/<file_id>', methods=['GET'])
def check_status(file_id):
    if file_id in results_cache:
        status = results_cache[file_id].get('status')
        response = {'status': status, 'file_id': file_id}
        
        # Include error message if there was an error
        if status == 'error' and 'error' in results_cache[file_id]:
            response['error'] = results_cache[file_id]['error']
            
        return jsonify(response), 200
    
    return jsonify({'status': 'not_found', 'file_id': file_id}), 404

@app.route('/api/upload-to-gcs', methods=['POST'])
def upload_models_to_gcs():
    try:
        data = request.get_json()
        bucket_name = data.get('bucket_name')
        
        if not bucket_name:
            return jsonify({'error': 'Bucket name is required'}), 400
        
        # Upload models to GCS
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

# Periodically clean up old cache entries (could be implemented with a scheduled task)
@app.route('/api/cleanup', methods=['POST'])
def cleanup_cache():
    try:
        # Remove entries older than specified time
        removed = 0
        for key in list(results_cache.keys()):
            # Implement your cleanup logic here
            removed += 1
            del results_cache[key]
        
        return jsonify({
            'message': f'Cache cleanup complete. Removed {removed} entries.',
            'remaining': len(results_cache) 
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use this for local development
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
