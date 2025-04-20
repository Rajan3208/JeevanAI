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
            
            # Added try-except blocks to ensure robustness
            try:
                pytesseract_text = extract_text_with_pytesseract(images)
            except Exception as e:
                logger.warning(f"Pytesseract extraction failed: {str(e)}")
                pytesseract_text = ""
                
            try:
                easyocr_text = extract_text_with_easyocr(images)
            except Exception as e:
                logger.warning(f"EasyOCR extraction failed: {str(e)}")
                easyocr_text = ""
            
            combined_text = pytesseract_text + "\n" + easyocr_text
            
            # Check if we got any text
            if len(combined_text.strip()) < 50:
                logger.warning(f"Very little text extracted from image: {file_id}")
                # Keep processing but log the warning
        else:
            # For PDFs, use optimized extraction approach
            logger.info(f"Processing PDF file: {file_id}")
            
            # Try PyPDF first as it's fastest
            try:
                pypdf_text = extract_text_with_pyPDF(file_path)
            except Exception as e:
                logger.warning(f"PyPDF extraction failed: {str(e)}")
                pypdf_text = ""
            
            # If PyPDF gets sufficient text, skip other methods
            if len(pypdf_text.strip()) > 200:  # Higher threshold for meaningful text
                logger.info(f"Using PyPDF extraction for {file_id}")
                combined_text = pypdf_text
            else:
                # Fall back to langchain
                logger.info(f"PyPDF extraction insufficient, trying Langchain for {file_id}")
                try:
                    langchain_pdf_text = extract_text_with_langchain_pdf(file_path)
                except Exception as e:
                    logger.warning(f"Langchain extraction failed: {str(e)}")
                    langchain_pdf_text = ""
                
                if len(langchain_pdf_text.strip()) > 200:
                    combined_text = langchain_pdf_text
                else:
                    # Only if both fail, use OCR (most time-consuming)
                    logger.info(f"Text extraction failed, using OCR for {file_id}")
                    try:
                        images = convert_pdf_to_images(file_path)
                        pytesseract_text = extract_text_with_pytesseract(images)
                        easyocr_text = extract_text_with_easyocr(images)
                        combined_text = pytesseract_text + "\n" + easyocr_text
                    except Exception as e:
                        logger.error(f"OCR extraction failed: {str(e)}")
                        # Use whatever text we have so far
                        combined_text = pypdf_text + "\n" + langchain_pdf_text
        
        # Update status to show text extraction complete
        results_cache[file_id]['status'] = 'extracting_insights'
        
        # Generate insights with timeout
        logger.info(f"Generating insights for {file_id} with {len(combined_text)} characters of text")
        insights = generate_comprehensive_insights(
            combined_text, 
            file_path, 
            None,  # Only pass images if needed
            model_cache,
            timeout=180  # Timeout for inference (180 seconds)
        )
        
        # If insights generation failed, provide at least some basic information
        if not insights or not isinstance(insights, dict):
            logger.warning(f"Insights generation returned invalid data for {file_id}")
            insights = {
                'summary': "Unable to generate detailed insights from this document.",
                'entities': {},
                'topics': [],
                'sentiment': 'NEUTRAL'
            }
        
        # Prepare response
        result = {
            'file_id': file_id,
            'text_length': len(combined_text),
            'extraction_method': 'optimized',
            'insights': insights.get('summary', "No summary available"),
            'entities': list(insights.get('entities', {}).keys())[:10],
            'topics': insights.get('topics', [])[:3],
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
        # Clean up uploaded file - but only after processing is complete
        try:
            if os.path.exists(file_path):
                # For debugging large documents, optionally keep files temporarily
                # Uncomment next line to retain files for debugging
                # logger.info(f"Keeping file for debugging: {file_path}")
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
            # Still processing - include more detailed status
            return jsonify({
                'status': 'processing', 
                'stage': status if status != 'processing' else 'extracting_text',
                'file_id': file_id
            }), 202
    
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

# Periodically clean up old cache entries
@app.route('/api/cleanup', methods=['POST'])
def cleanup_cache():
    try:
        # Remove entries older than specified time
        removed = 0
        current_size = len(results_cache) 
        
        # Keep only the 100 most recent entries if we have more than 150
        if current_size > 150:
            # This is a simple approach - in production you'd track timestamps
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
    # Use this for local development
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
