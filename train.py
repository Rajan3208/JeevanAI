import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.pdf_processor import convert_pdf_to_images
from utils.text_extractor import (
    extract_text_with_pyPDF,
    extract_text_with_langchain_pdf,
    extract_text_with_pytesseract,
    extract_text_with_easyocr
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_parallel(file_path, images=None):
    """Extract text using multiple methods in parallel"""
    is_pdf = file_path.lower().endswith('.pdf')
    
    with ThreadPoolExecutor() as executor:
        futures = {}
        
        # For PDFs, use all extraction methods
        if is_pdf:
            futures['pypdf'] = executor.submit(extract_text_with_pyPDF, file_path)
            futures['langchain'] = executor.submit(extract_text_with_langchain_pdf, file_path)
            
            # Get images if not provided
            if images is None:
                logger.info("Converting PDF to images...")
                images = convert_pdf_to_images(file_path)
        
        # For both PDFs and images, use OCR methods
        if images:
            futures['pytesseract'] = executor.submit(extract_text_with_pytesseract, images)
            futures['easyocr'] = executor.submit(extract_text_with_easyocr, images)
        
        # Get results
        results = {}
        for k, f in futures.items():
            try:
                results[k] = f.result()
                logger.info(f"Extraction method {k} extracted {len(results[k])} characters")
            except Exception as e:
                logger.error(f"Extraction method {k} failed: {e}")
                results[k] = ""
        
        return results

def train_topic_model(text, n_components=3):
    """Train a topic model on the provided text"""
    try:
        logger.info("Training topic model...")
        vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2, 
            stop_words='english', 
            max_features=5000
        )
        dtm = vectorizer.fit_transform([text])
        
        nmf_model = NMF(
            n_components=n_components, 
            random_state=42,
            max_iter=300
        )
        nmf_model.fit(dtm)
        
        return nmf_model, vectorizer
    except Exception as e:
        logger.error(f"Failed to train topic model: {e}")
        raise

def train_sentiment_model(input_dim=768, output_dim=2):
    """Train a sentiment analysis model with dummy data"""
    try:
        logger.info("Training sentiment model...")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )

        # More realistic dummy data
        X_dummy = np.random.normal(size=(300, input_dim))
        y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, output_dim, size=(300,)))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]

        model.fit(
            X_dummy, y_dummy,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        return model
    except Exception as e:
        logger.error(f"Failed to train sentiment model: {e}")
        raise

def train_models(file_path, output_dir='models'):
    """Train models from a PDF or image file"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing file: {file_path}")
    
    # Determine file type
    is_image = file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
    is_pdf = file_path.lower().endswith('.pdf')
    
    if not (is_image or is_pdf):
        raise ValueError("File must be a PDF or image (PNG, JPG, JPEG, TIFF, BMP)")
    
    # Extract text
    if is_image:
        logger.info("Processing image file...")
        images = [file_path]  # For images, use file path directly
        text_outputs = extract_text_parallel(file_path, images)
    else:
        logger.info("Processing PDF file...")
        # For PDFs, extract_text_parallel will handle converting to images
        text_outputs = extract_text_parallel(file_path)
    
    # Combine text from all extraction methods
    combined_text = "\n".join([text for text in text_outputs.values() if text])
    
    if len(combined_text.strip()) < 100:
        logger.warning("Very little text extracted. Models might not perform well.")

    # Train and save models
    model_paths = {}
    
    # Train topic model
    try:
        nmf_model, vectorizer = train_topic_model(combined_text)
        topic_model_path = os.path.join(output_dir, 'topic_model.pkl')
        with open(topic_model_path, 'wb') as f:
            pickle.dump((nmf_model, vectorizer), f)
        logger.info(f"Topic model saved to {topic_model_path}")
        model_paths['topic_model'] = topic_model_path
    except Exception as e:
        logger.error(f"Failed to save topic model: {e}")
        model_paths['topic_model'] = None

    # Train sentiment model
    try:
        sentiment_model = train_sentiment_model()
        sentiment_model_path = os.path.join(output_dir, 'sentiment_model.keras')
        sentiment_model.save(sentiment_model_path)
        logger.info(f"Sentiment model saved to {sentiment_model_path}")
        model_paths['sentiment_model'] = sentiment_model_path
    except Exception as e:
        logger.error(f"Failed to save sentiment model: {e}")
        model_paths['sentiment_model'] = None

    # Save extraction insights
    try:
        insights = {
            'file_type': 'image' if is_image else 'pdf',
            'combined_text_length': len(combined_text),
            'extraction_methods': {k: len(v) for k, v in text_outputs.items()}
        }
        insights_path = os.path.join(output_dir, 'document_insights.pkl')
        with open(insights_path, 'wb') as f:
            pickle.dump(insights, f)
        logger.info(f"Document insights saved to {insights_path}")
        model_paths['insights'] = insights_path
    except Exception as e:
        logger.error(f"Failed to save insights: {e}")
        model_paths['insights'] = None

    return model_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models on PDF or image data')
    parser.add_argument('--file_path', required=True, help='Path to the PDF or image file')
    parser.add_argument('--output_dir', default='models', help='Directory to save models')
    args = parser.parse_args()

    try:
        results = train_models(args.file_path, args.output_dir)
        logger.info(f"All models saved to {args.output_dir}")
        
        # Print summary of results
        logger.info("Training Results Summary:")
        for model_name, path in results.items():
            status = "SUCCESS" if path else "FAILED"
            logger.info(f"  - {model_name}: {status}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        exit(1)
