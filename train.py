import os
import argparse
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from utils.pdf_processor import convert_pdf_to_images
from utils.text_extractor import (
    extract_text_with_pyPDF,
    extract_text_with_langchain_pdf,
    extract_text_with_pytesseract,
    extract_text_with_easyocr
)
import spacy

def train_models(pdf_path, output_dir='models'):
    """Train and save the NLP and ML models based on PDF content"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing PDF: {pdf_path}")
    # Extract images from PDF
    images = convert_pdf_to_images(pdf_path)
    
    # Extract text using multiple methods
    pypdf_text = extract_text_with_pyPDF(pdf_path)
    langchain_pdf_text = extract_text_with_langchain_pdf(pdf_path)
    pytesseract_text = extract_text_with_pytesseract(images)
    easyocr_text = extract_text_with_easyocr(images)
    
    # Combine text from all extraction methods
    combined_text = (pypdf_text + "\n" + 
                    langchain_pdf_text + "\n" + 
                    pytesseract_text + "\n" + 
                    easyocr_text)
    
    print("Training topic model...")
    # Train Topic Model with scikit-learn
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    try:
        dtm = vectorizer.fit_transform([combined_text])
        num_topics = 3
        nmf_model = NMF(n_components=num_topics, random_state=42)
        nmf_model.fit(dtm)
        
        # Save topic model
        topic_model_path = os.path.join(output_dir, 'topic_model.pkl')
        with open(topic_model_path, 'wb') as f:
            pickle.dump((nmf_model, vectorizer), f)
        print(f"Topic model saved to {topic_model_path}")
    except Exception as e:
        print(f"Error training topic model: {e}")
    
    print("Training sentiment model...")
    # Create and save a simple sentiment model
    try:
        # Create a simple neural network for sentiment analysis
        model = Sequential([
            Dense(128, activation='relu', input_shape=(768,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Generate some dummy data for demonstration
        X_dummy = np.random.random((100, 768))
        y_dummy = np.random.randint(0, 2, size=(100, 2))
        
        # Train the model
        checkpoint = ModelCheckpoint(
            os.path.join(output_dir, 'sentiment_model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        model.fit(
            X_dummy, y_dummy,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=[checkpoint],
            verbose=1
        )
        
        # Save the model
        sentiment_model_path = os.path.join(output_dir, 'sentiment_model.keras')
        model.save(sentiment_model_path)
        print(f"Sentiment model saved to {sentiment_model_path}")
    except Exception as e:
        print(f"Error training sentiment model: {e}")
    
    print("Saving document insights...")
    # Save document insights
    insights = {
        'combined_text_length': len(combined_text),
        'extraction_methods': {
            'pypdf_length': len(pypdf_text),
            'langchain_length': len(langchain_pdf_text),
            'pytesseract_length': len(pytesseract_text),
            'easyocr_length': len(easyocr_text)
        }
    }
    
    insights_path = os.path.join(output_dir, 'document_insights.pkl')
    with open(insights_path, 'wb') as f:
        pickle.dump(insights, f)
    print(f"Document insights saved to {insights_path}")
    
    return {
        'topic_model': topic_model_path,
        'sentiment_model': sentiment_model_path,
        'insights': insights_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models on PDF data')
    parser.add_argument('--pdf_path', required=True, help='Path to the PDF file')
    parser.add_argument('--output_dir', default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Train models
    trained_models = train_models(args.pdf_path, args.output_dir)
    print(f"Training complete. Models saved in {args.output_dir}")
