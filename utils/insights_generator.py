import spacy
import pickle
import numpy as np
import time
import signal
import re
import os
import logging
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)

# Initialize spaCy model with better error handling
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            import subprocess
            logger.info("Downloading spaCy model...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                         check=True, capture_output=True)
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {e}")
            # Create a minimal blank model as fallback
            return spacy.blank("en")

# Load the model on module import, but don't crash if it fails
try:
    nlp = load_spacy_model()
    logger.info("SpaCy model loaded successfully")
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    # Initialize a blank model as fallback
    nlp = spacy.blank("en")

# Timeout decorator
def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

def with_timeout(timeout_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the timeout handler
            original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm and restore original handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
            return result
        return wrapper
    return decorator

def extract_insights_with_spacy(text):
    """
    Extract insights using spaCy
    
    Args:
        text: Text to analyze
        
    Returns:
        tuple: (insights dict, summary string)
    """
    try:
        # Ensure we have text to analyze
        if not text or len(text.strip()) < 10:
            return {}, "Not enough text to analyze."
        
        # Truncate to avoid memory issues but ensure we have enough to analyze
        max_text_length = 50000  # Reduced from 100000 to avoid memory issues
        truncated_text = text[:max_text_length]
        
        # Process the text in chunks if it's very large
        doc = None
        if len(truncated_text) > 25000:
            # Process in chunks and combine results
            chunk_size = 10000
            chunks = [truncated_text[i:i+chunk_size] for i in range(0, len(truncated_text), chunk_size)]
            
            entities = {}
            noun_chunks = []
            sentences = []
            
            for chunk in chunks:
                chunk_doc = nlp(chunk)
                # Collect entities from each chunk
                for ent in chunk_doc.ents:
                    entities[ent.text] = ent.label_
                # Collect noun chunks from each chunk
                noun_chunks.extend([chunk.text for chunk in chunk_doc.noun_chunks])
                # Collect sentences from each chunk
                sentences.extend(list(chunk_doc.sents))
        else:
            # Process normally for smaller texts
            doc = nlp(truncated_text)
            entities = {ent.text: ent.label_ for ent in doc.ents}
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            sentences = list(doc.sents)
        
        # Only proceed with summary if we have sentences
        if sentences:
            # For chunked processing, we might not have doc.ents available, so we need a different approach
            if doc is None:
                # Just use the first few sentences as a summary for chunked processing
                summary = ' '.join([sent.text for sent in sentences[:3]])
            else:
                # Score sentences by the number of entities they contain
                important_sentences = sorted(
                    sentences,
                    key=lambda s: sum(1 for ent in doc.ents if ent.start >= s.start and ent.end <= s.end),
                    reverse=True
                )[:3]
                
                # Create summary
                summary = ' '.join([sent.text for sent in important_sentences])
                
                # Fall back to first few sentences if no entities were found
                if not summary:
                    summary = ' '.join([sent.text for sent in sentences[:3]])
        else:
            # No sentences found, use the first part of text
            summary = truncated_text[:200]
        
        # Collect key medical terms if present
        medical_terms = []
        if doc is not None:
            medical_entities = [ent.text for ent in doc.ents if ent.label_ in ("DISEASE", "CONDITION", "TREATMENT", "MEDICATION")]
            if medical_entities:
                medical_terms = medical_entities[:5]
        
        # Create insights dictionary
        insights = {
            'entities': entities,
            'key_phrases': noun_chunks[:10],
            'summary': summary,
            'medical_terms': medical_terms
        }
        
        # Create a readable summary
        entity_summary = "No named entities detected."
        if entities:
            entity_list = list(entities.keys())[:5]
            entity_summary = f"Document contains {len(entities)} named entities including {', '.join(entity_list)}."
        
        topic_summary = "No clear topics identified."
        if noun_chunks:
            topics = noun_chunks[:3]
            topic_summary = f"Key topics focus on {', '.join(topics)}."
            
        return insights, f"{entity_summary} {topic_summary}"
    
    except Exception as e:
        logger.error(f"Error extracting insights with spaCy: {e}")
        return {
            'entities': {},
            'key_phrases': [],
            'summary': "Basic text analysis couldn't be completed."
        }, "Could not extract insights with spaCy."

def extract_insights_with_sklearn(text, model_cache=None, num_topics=3):
    """
    Extract topics using scikit-learn
    
    Args:
        text: Text to analyze
        model_cache: Dictionary of cached models
        num_topics: Number of topics to extract
        
    Returns:
        tuple: (topics list, summary string, model filename)
    """
    try:
        # Check for minimum text length
        if len(text.split()) < 20:
            return [], "Text too short for topic extraction", None
        
        # Use cached model if available
        if model_cache and 'topic_model' in model_cache:
            try:
                nmf_model, vectorizer = model_cache['topic_model']
                # Handle case where the vocabulary might be different
                try:
                    dtm = vectorizer.transform([text])
                except:
                    # Fall back to creating a new model
                    logger.warning("Cached vectorizer vocabulary mismatch, creating new topic model")
                    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                    dtm = vectorizer.fit_transform([text])
                    nmf_model = NMF(n_components=num_topics, random_state=42)
                    nmf_model.fit(dtm)
            except Exception as e:
                logger.warning(f"Error using cached topic model: {e}, creating new model")
                # Create new model
                vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                dtm = vectorizer.fit_transform([text])
                
                # Check if we have any features
                if dtm.shape[1] == 0:
                    return [], "Not enough unique terms for topic extraction", None
                    
                nmf_model = NMF(n_components=min(num_topics, dtm.shape[1]), random_state=42)
                nmf_model.fit(dtm)
        else:
            # Create new model
            logger.info("No cached topic model found, creating new one")
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform([text])
            
            # Check if we have any features
            if dtm.shape[1] == 0:
                return [], "Not enough unique terms for topic extraction", None
                
            nmf_model = NMF(n_components=min(num_topics, dtm.shape[1]), random_state=42)
            nmf_model.fit(dtm)
            
            # Save model - with try/except to handle permission issues
            try:
                model_filename = os.path.join(os.getenv('MODELS_FOLDER', '.'), 'topic_model.pkl')
                with open(model_filename, 'wb') as f:
                    pickle.dump((nmf_model, vectorizer), f)
                
                if model_cache is not None:
                    model_cache['topic_model'] = (nmf_model, vectorizer)
            except Exception as e:
                logger.error(f"Error saving topic model: {e}")
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        if hasattr(nmf_model, 'components_'):
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)
        
        if topics:
            topic_insights = f"Main topics: {', '.join([' & '.join(topic[:3]) for topic in topics])}"
        else:
            topic_insights = "No clear topics could be identified in the text."
            
        return topics, topic_insights, 'topic_model.pkl'
        
    except Exception as e:
        logger.error(f"Error extracting insights with scikit-learn: {e}")
        return [], "Could not extract topics due to insufficient text data", None

def extract_insights_with_transformers(text, model_cache=None):
    """
    Extract insights using transformers
    
    Args:
        text: Text to analyze
        model_cache: Dictionary of cached models
        
    Returns:
        tuple: (insights dict, summary string, keras model path)
    """
    try:
        # Check if we have enough text to analyze
        if len(text.strip()) < 50:
            return {
                'sentiment': 'NEUTRAL',
                'summary': text[:100]
            }, "Text too short for detailed analysis.", None
        
        # Determine overall sentiment with simple rule-based approach if transformers not available
        # This is a fallback if the model isn't loaded correctly
        sentiment = 'NEUTRAL'
        
        # Try using cached models first
        if model_cache and 'sentiment_analyzer' in model_cache:
            try:
                logger.info("Using cached sentiment analyzer")
                sentiment_analyzer = model_cache['sentiment_analyzer']
                
                # Break text into manageable chunks for the model
                chunks = [text[i:i+512] for i in range(0, min(len(text), 5000), 512)]
                
                # Only analyze non-empty chunks
                valid_chunks = [chunk for chunk in chunks if chunk.strip()]
                
                if not valid_chunks:
                    return {
                        'sentiment': 'NEUTRAL', 
                        'summary': text[:100]
                    }, "Document appears neutral in tone.", None
                
                # Analyze sentiment of first few chunks
                sentiments = []
                for chunk in valid_chunks[:5]:  # Limit to first 5 chunks
                    try:
                        result = sentiment_analyzer(chunk)
                        if result:
                            sentiments.append(result[0])
                    except Exception as e:
                        logger.error(f"Error analyzing chunk: {e}")
                
                # Determine overall sentiment
                if sentiments:
                    # Get the most common sentiment label
                    sentiment_labels = [s['label'] for s in sentiments]
                    sentiment = max(set(sentiment_labels), key=sentiment_labels.count)
                    
                    # Get average confidence
                    avg_confidence = sum(s['score'] for s in sentiments) / len(sentiments)
            except Exception as e:
                logger.error(f"Error using cached sentiment analyzer: {e}")
                sentiment = 'NEUTRAL'
                avg_confidence = 0.5
        else:
            logger.warning("No sentiment analyzer in model cache, using simple rule-based approach")
            # Simple rule-based sentiment analysis as fallback
            positive_words = ['good', 'great', 'excellent', 'positive', 'well', 'healthy', 'normal']
            negative_words = ['bad', 'poor', 'negative', 'sick', 'ill', 'abnormal', 'cancer', 'disease', 'tumor']
            
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            
            if positive_count > negative_count * 1.5:
                sentiment = 'POSITIVE'
            elif negative_count > positive_count * 1.5:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            avg_confidence = 0.7  # Default confidence value
        
        # Only try to run summarizer if transformers models are available
        summary_text = ""
        if model_cache and 'summarizer' in model_cache:
            try:
                logger.info("Using cached summarizer")
                summarizer = model_cache['summarizer']
                if len(text) > 100:
                    # Limit input text to what the model can handle
                    summary = summarizer(text[:1024], max_length=100, min_length=30, do_sample=False)
                    summary_text = summary[0]['summary_text'] if summary else ""
                else:
                    summary_text = text
            except Exception as e:
                logger.error(f"Error in summarization: {e}")
                # Fallback to extracting first few sentences
                sentences = re.split(r'[.!?]+', text)
                summary_text = '. '.join(sentences[:3]) + '.'
        else:
            logger.warning("No summarizer in model cache, using simple extraction")
            # Simple extraction as fallback
            sentences = re.split(r'[.!?]+', text)
            summary_text = '. '.join([s.strip() for s in sentences[:3] if s.strip()]) + '.'
        
        sentiment_description = ""
        if sentiment == "POSITIVE":
            sentiment_description = "The document generally expresses positive sentiments."
        elif sentiment == "NEGATIVE":
            sentiment_description = "The document generally expresses negative sentiments."
        else:
            sentiment_description = "The document appears neutral in tone."
            
        return {
            'sentiment': sentiment,
            'sentiment_confidence': avg_confidence if 'avg_confidence' in locals() else 0.7,
            'summary': summary_text
        }, f"Document sentiment: {sentiment}. {sentiment_description} {summary_text[:100]}...", 'sentiment_model.keras'
            
    except Exception as e:
        logger.error(f"Error in transformer processing: {e}")
        return {
            'sentiment': 'NEUTRAL',
            'summary': text[:100]
        }, "Unable to determine document sentiment.", None

@with_timeout(120)  # Ensure this function never runs more than 120 seconds
def generate_comprehensive_insights(text, file_path, images=None, model_cache=None, timeout=120):
    """
    Generate comprehensive insights from text
    
    Args:
        text: Text to analyze
        file_path: Path to the PDF file
        images: List of images from the PDF
        model_cache: Dictionary of cached models
        timeout: Maximum time in seconds for insights generation (default: 120)
        
    Returns:
        dict: Dictionary of insights
    """
    logger.info(
