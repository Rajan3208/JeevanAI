import os
import pickle
import tensorflow as tf
from google.cloud import storage

def load_models(models_dir):
    """
    Load all models from the models directory
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    
    try:
        # Load topic model
        topic_model_path = os.path.join(models_dir, 'topic_model.pkl')
        if os.path.exists(topic_model_path):
            with open(topic_model_path, 'rb') as f:
                models['topic_model'] = pickle.load(f)
                print(f"Loaded topic model from {topic_model_path}")
        
        # Load sentiment model
        sentiment_model_path = os.path.join(models_dir, 'sentiment_model.keras')
        if os.path.exists(sentiment_model_path):
            models['sentiment_model'] = tf.keras.models.load_model(sentiment_model_path)
            print(f"Loaded sentiment model from {sentiment_model_path}")
        
        # Load document insights
        insights_path = os.path.join(models_dir, 'document_insights.pkl')
        if os.path.exists(insights_path):
            with open(insights_path, 'rb') as f:
                models['document_insights'] = pickle.load(f)
                print(f"Loaded document insights from {insights_path}")
                
    except Exception as e:
        print(f"Error loading models: {e}")
    
    return models

def upload_to_gcs(models_dir, bucket_name, prefix='models'):
    """
    Upload models to Google Cloud Storage
    
    Args:
        models_dir: Directory containing model files
        bucket_name: GCS bucket name
        prefix: Prefix for the GCS object path
        
    Returns:
        list: List of uploaded file paths
    """
    try:
        # Create GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # List of files to upload
        model_files = [
            'topic_model.pkl',
            'sentiment_model.keras',
            'document_insights.pkl'
        ]
        
        uploaded_files = []
        
        for file_name in model_files:
            local_path = os.path.join(models_dir, file_name)
            if os.path.exists(local_path):
                # Create GCS path
                gcs_path = f"{prefix}/{file_name}"
                
                # Upload file
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                
                # Add to uploaded files
                uploaded_files.append({
                    'local_path': local_path,
                    'gcs_path': f"gs://{bucket_name}/{gcs_path}"
                })
        
        return uploaded_files
    
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return []

def download_from_gcs(bucket_name, models_dir, prefix='models'):
    """
    Download models from Google Cloud Storage
    
    Args:
        bucket_name: GCS bucket name
        models_dir: Directory to save downloaded models
        prefix: Prefix for the GCS object path
        
    Returns:
        list: List of downloaded file paths
    """
    try:
        # Create GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # List of files to download
        model_files = [
            'topic_model.pkl',
            'sentiment_model.keras',
            'document_insights.pkl'
        ]
        
        downloaded_files = []
        
        for file_name in model_files:
            # Create GCS path
            gcs_path = f"{prefix}/{file_name}"
            
            # Create local path
            local_path = os.path.join(models_dir, file_name)
            
            # Download file
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            
            # Add to downloaded files
            downloaded_files.append({
                'gcs_path': f"gs://{bucket_name}/{gcs_path}",
                'local_path': local_path
            })
        
        return downloaded_files
    
    except Exception as e:
        print(f"Error downloading from GCS: {e}")
        return []
