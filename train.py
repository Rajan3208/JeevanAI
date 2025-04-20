import os
import argparse
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.pdf_processor import convert_pdf_to_images
from utils.text_extractor import (
    extract_text_with_pyPDF,
    extract_text_with_langchain_pdf,
    extract_text_with_pytesseract,
    extract_text_with_easyocr
)

def extract_text_parallel(pdf_path, images):
    with ThreadPoolExecutor() as executor:
        futures = {
            'pypdf': executor.submit(extract_text_with_pyPDF, pdf_path),
            'langchain': executor.submit(extract_text_with_langchain_pdf, pdf_path),
            'pytesseract': executor.submit(extract_text_with_pytesseract, images),
            'easyocr': executor.submit(extract_text_with_easyocr, images)
        }
        return {k: f.result() for k, f in futures.items()}

def train_models(pdf_path, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Processing PDF: {pdf_path}")
    
    images = convert_pdf_to_images(pdf_path)
    print(f"[INFO] Extracting text using multiple methods...")
    text_outputs = extract_text_parallel(pdf_path, images)
    
    combined_text = "\n".join(text_outputs.values())
    
    # Train topic model
    print("[INFO] Training topic model...")
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform([combined_text])
        nmf_model = NMF(n_components=3, random_state=42)
        nmf_model.fit(dtm)

        topic_model_path = os.path.join(output_dir, 'topic_model.pkl')
        with open(topic_model_path, 'wb') as f:
            pickle.dump((nmf_model, vectorizer), f)
        print(f"[SUCCESS] Topic model saved to {topic_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to train topic model: {e}")
        topic_model_path = None

    # Dummy sentiment model training
    print("[INFO] Training sentiment model...")
    try:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(768,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # More realistic dummy data
        X_dummy = np.random.normal(size=(200, 768))
        y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(200,)))

        checkpoint_path = os.path.join(output_dir, 'sentiment_model_checkpoint.h5')
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

        model.fit(
            X_dummy, y_dummy,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=[checkpoint],
            verbose=1
        )

        sentiment_model_path = os.path.join(output_dir, 'sentiment_model.keras')
        model.save(sentiment_model_path)
        print(f"[SUCCESS] Sentiment model saved to {sentiment_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to train sentiment model: {e}")
        sentiment_model_path = None

    # Save insights
    print("[INFO] Saving document insights...")
    insights = {
        'combined_text_length': len(combined_text),
        'extraction_methods': {k: len(v) for k, v in text_outputs.items()}
    }
    insights_path = os.path.join(output_dir, 'document_insights.pkl')
    with open(insights_path, 'wb') as f:
        pickle.dump(insights, f)
    print(f"[SUCCESS] Document insights saved to {insights_path}")

    # Optional: Add code here to upload model to Google Cloud if needed

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

    results = train_models(args.pdf_path, args.output_dir)
    print(f"[COMPLETE] All models saved to {args.output_dir}")
