import os
import uuid
import json
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

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Load models at startup
model_cache = {}

# Replace the @app.before_first_request with this approach
def load_all_models():
    global model_cache
    model_cache = load_models(app.config['MODELS_FOLDER'])

# Execute at startup time
with app.app_context():
    load_all_models()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': bool(model_cache)}), 200

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400
    
    # Create a unique filename
    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    try:
        # Return file ID for the next step
        return jsonify({
            'message': 'File uploaded successfully',
            'file_id': unique_filename
        }), 200
    except Exception as e:
        os.remove(file_path)  # Clean up on error
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/<file_id>', methods=['GET'])
def analyze_pdf(file_id):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Process PDF and extract texts
        images = convert_pdf_to_images(file_path)
        
        # Extract text using different methods
        pypdf_text = extract_text_with_pyPDF(file_path)
        langchain_pdf_text = extract_text_with_langchain_pdf(file_path)
        pytesseract_text = extract_text_with_pytesseract(images)
        easyocr_text = extract_text_with_easyocr(images)
        
        # Combine text from all methods
        combined_text = (pypdf_text + "\n" + 
                        langchain_pdf_text + "\n" + 
                        pytesseract_text + "\n" + 
                        easyocr_text)
        
        # Generate insights from the text
        insights = generate_comprehensive_insights(
            combined_text, 
            file_path, 
            images,
            model_cache
        )
        
        # Prepare response
        result = {
            'file_id': file_id,
            'text_length': len(combined_text),
            'extraction_methods': {
                'pypdf': len(pypdf_text) > 0,
                'langchain': len(langchain_pdf_text) > 0,
                'pytesseract': len(pytesseract_text) > 0,
                'easyocr': len(easyocr_text) > 0
            },
            'insights': insights['summary'],
            'entities': list(insights['entities'].keys())[:10] if 'entities' in insights else [],
            'topics': insights['topics'][:3] if 'topics' in insights else [],
            'sentiment': insights.get('sentiment', 'NEUTRAL')
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use this for local development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
