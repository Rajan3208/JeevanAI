import spacy
import pickle
import numpy as np
import time
import signal
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from transformers import pipeline

# Initialize spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Timeout decorator
def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

def with_timeout(timeout_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
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
        doc = nlp(text[:100000])  # Limit to avoid memory issues
        entities = {ent.text: ent.label_ for ent in doc.ents}
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        sentences = list(doc.sents)
        important_sentences = sorted(
            sentences,
            key=lambda s: sum(1 for ent in doc.ents if ent.start >= s.start and ent.end <= s.end),
            reverse=True
        )[:3]
        summary = ' '.join([sent.text for sent in important_sentences])

        insights = {
            'entities': entities,
            'key_phrases': noun_chunks[:10],
            'summary': summary
        }

        return insights, f"Document contains {len(entities)} named entities including {', '.join(list(entities.keys())[:5])}. Key topics focus on {', '.join(noun_chunks[:3])}."
    except Exception as e:
        print(f"Error extracting insights with spaCy: {e}")
        return {}, "Could not extract insights with spaCy."

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
        # Use cached model if available
        if model_cache and 'topic_model' in model_cache:
            nmf_model, vectorizer = model_cache['topic_model']
            dtm = vectorizer.transform([text])
        else:
            # Create new model
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform([text])
            nmf_model = NMF(n_components=num_topics, random_state=42)
            nmf_model.fit(dtm)
            
            # Save model
            model_filename = 'topic_model.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump((nmf_model, vectorizer), f)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
        
        topic_insights = f"Main topics: {', '.join([' & '.join(topic[:3]) for topic in topics])}"
        return topics, topic_insights, 'topic_model.pkl'
    except Exception as e:
        print(f"Error extracting insights with scikit-learn: {e}")
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
        sentiment_analyzer = pipeline('sentiment-analysis')
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        sentiments = [sentiment_analyzer(chunk) for chunk in chunks[:5] if chunk.strip()]  # Limit to first 5 chunks

        if sentiments:
            overall_sentiment = max(set([s[0]['label'] for s in sentiments if s]),
                                  key=[s[0]['label'] for s in sentiments if s].count)
            
            # Only run summarizer if text is long enough
            if len(text) > 100:
                summarizer = pipeline('summarization')
                summary = summarizer(text[:1024], max_length=100, min_length=30, do_sample=False)
                summary_text = summary[0]['summary_text']
            else:
                summary_text = text

            return {
                'sentiment': overall_sentiment,
                'summary': summary_text
            }, f"Document sentiment: {overall_sentiment}. {summary_text[:100]}...", 'sentiment_model.keras'
        else:
            return {'sentiment': 'NEUTRAL', 'summary': text[:100]}, "Document appears neutral in tone.", None
    except Exception as e:
        print(f"Error in transformer processing: {e}")
        return {'sentiment': 'UNKNOWN', 'summary': text[:100]}, "Unable to determine document sentiment.", None

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
    print(f"Generating insights with timeout of {timeout} seconds...")
    
    # Wrap the insights generation in timeout handling
    start_time = time.time()
    
    try:
        # Use separate timeouts for each method to ensure we get some results
        max_time_per_method = timeout / 3
        
        # First try spaCy (fastest)
        spacy_insights, spacy_summary = extract_insights_with_spacy(text)
        
        # Check remaining time
        elapsed = time.time() - start_time
        remaining_time = timeout - elapsed
        
        if remaining_time <= 0:
            return {
                'summary': spacy_summary.strip(),
                'entities': spacy_insights.get('entities', {}),
                'key_phrases': spacy_insights.get('key_phrases', []),
                'topics': [],
                'sentiment': 'UNKNOWN',
                'transformer_summary': ''
            }
        
        # Then try sklearn
        topics, topic_summary, _ = extract_insights_with_sklearn(text, model_cache)
        
        # Check remaining time again
        elapsed = time.time() - start_time
        remaining_time = timeout - elapsed
        
        if remaining_time <= 0:
            return {
                'summary': f"{spacy_summary}\n\n{topic_summary}".strip(),
                'entities': spacy_insights.get('entities', {}),
                'key_phrases': spacy_insights.get('key_phrases', []),
                'topics': topics,
                'sentiment': 'UNKNOWN',
                'transformer_summary': ''
            }
        
        # Finally try transformers (slowest)
        transformer_results, transformer_summary, _ = extract_insights_with_transformers(text, model_cache)
        
        insights_summary = f"""
        DOCUMENT INSIGHTS SUMMARY:

        {spacy_summary}

        {topic_summary}

        {transformer_summary}
        """

        detailed_insights = {
            'summary': insights_summary.strip(),
            'entities': spacy_insights.get('entities', {}),
            'key_phrases': spacy_insights.get('key_phrases', []),
            'topics': topics,
            'sentiment': transformer_results.get('sentiment', 'NEUTRAL'),
            'transformer_summary': transformer_results.get('summary', '')
        }

        return detailed_insights
        
    except TimeoutError:
        print(f"Insights generation timed out after {timeout} seconds")
        # Return whatever we've got so far
        return {
            'summary': "Analysis timed out. Partial results available.",
            'entities': spacy_insights.get('entities', {}) if 'spacy_insights' in locals() else {},
            'key_phrases': spacy_insights.get('key_phrases', []) if 'spacy_insights' in locals() else [],
            'topics': topics if 'topics' in locals() else [],
            'sentiment': 'UNKNOWN',
            'transformer_summary': ''
        }
    except Exception as e:
        print(f"Error generating comprehensive insights: {e}")
        return {
            'summary': f"Error generating insights: {str(e)}",
            'entities': {},
            'key_phrases': [],
            'topics': [],
            'sentiment': 'UNKNOWN',
            'transformer_summary': ''
        }
