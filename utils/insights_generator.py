import spacy
import pickle
import numpy as np
import time
import signal
import re
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
        # Ensure we have text to analyze
        if not text or len(text.strip()) < 10:
            return {}, "Not enough text to analyze."
        
        # Truncate to avoid memory issues but ensure we have enough to analyze
        truncated_text = text[:100000]
        doc = nlp(truncated_text)
        
        # Extract entities
        entities = {ent.text: ent.label_ for ent in doc.ents}
        
        # Get noun chunks for key phrases
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        # Generate a smarter summary using entity density
        sentences = list(doc.sents)
        
        # Only proceed with summary if we have sentences
        if sentences:
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
        print(f"Error extracting insights with spaCy: {e}")
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
            nmf_model, vectorizer = model_cache['topic_model']
            # Handle case where the vocabulary might be different
            try:
                dtm = vectorizer.transform([text])
            except:
                # Fall back to creating a new model
                vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                dtm = vectorizer.fit_transform([text])
                nmf_model = NMF(n_components=num_topics, random_state=42)
                nmf_model.fit(dtm)
        else:
            # Create new model
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform([text])
            
            # Check if we have any features
            if dtm.shape[1] == 0:
                return [], "Not enough unique terms for topic extraction", None
                
            nmf_model = NMF(n_components=min(num_topics, dtm.shape[1]), random_state=42)
            nmf_model.fit(dtm)
            
            # Save model
            model_filename = 'topic_model.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump((nmf_model, vectorizer), f)
        
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
        # Check if we have enough text to analyze
        if len(text.strip()) < 50:
            return {
                'sentiment': 'NEUTRAL',
                'summary': text[:100]
            }, "Text too short for detailed analysis.", None
        
        # Load sentiment analyzer
        sentiment_analyzer = pipeline('sentiment-analysis')
        
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
                print(f"Error analyzing chunk: {e}")
        
        # Determine overall sentiment
        if sentiments:
            # Get the most common sentiment label
            sentiment_labels = [s['label'] for s in sentiments]
            overall_sentiment = max(set(sentiment_labels), key=sentiment_labels.count)
            
            # Get average confidence
            avg_confidence = sum(s['score'] for s in sentiments) / len(sentiments)
            
            # Only run summarizer if text is long enough
            summary_text = ""
            if len(text) > 100:
                try:
                    summarizer = pipeline('summarization')
                    # Limit input text to what the model can handle
                    summary = summarizer(text[:1024], max_length=100, min_length=30, do_sample=False)
                    summary_text = summary[0]['summary_text'] if summary else ""
                except Exception as e:
                    print(f"Error in summarization: {e}")
                    # Fallback to extracting first few sentences
                    sentences = re.split(r'[.!?]+', text)
                    summary_text = '. '.join(sentences[:3]) + '.'
            else:
                summary_text = text
            
            sentiment_description = ""
            if overall_sentiment == "POSITIVE":
                sentiment_description = "The document generally expresses positive sentiments."
            elif overall_sentiment == "NEGATIVE":
                sentiment_description = "The document generally expresses negative sentiments."
            else:
                sentiment_description = "The document appears neutral in tone."
                
            return {
                'sentiment': overall_sentiment,
                'sentiment_confidence': avg_confidence,
                'summary': summary_text
            }, f"Document sentiment: {overall_sentiment} (confidence: {avg_confidence:.2f}). {sentiment_description} {summary_text[:100]}...", 'sentiment_model.keras'
        else:
            return {
                'sentiment': 'NEUTRAL',
                'summary': text[:100]
            }, "Document appears neutral in tone.", None
            
    except Exception as e:
        print(f"Error in transformer processing: {e}")
        return {
            'sentiment': 'UNKNOWN',
            'summary': text[:100]
        }, "Unable to determine document sentiment.", None

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
    
    # Initialize results with defaults
    spacy_insights = {}
    spacy_summary = "No text analysis available."
    topics = []
    topic_summary = "No topic information available."
    transformer_results = {'sentiment': 'NEUTRAL', 'summary': ''}
    transformer_summary = "No sentiment analysis available."
    
    # Validate input text
    if not text or len(text.strip()) < 10:
        print("Warning: Input text is too short for analysis")
        insights_summary = "The document doesn't contain enough text for analysis."
        return {
            'summary': insights_summary,
            'entities': {},
            'key_phrases': [],
            'topics': [],
            'sentiment': 'NEUTRAL',
            'transformer_summary': ''
        }
    
    # Track processing time
    start_time = time.time()
    
    try:
        # Process with spaCy first (fastest)
        spacy_insights, spacy_summary = extract_insights_with_spacy(text)
        
        # Check for timeout
        elapsed = time.time() - start_time
        remaining_time = timeout - elapsed
        print(f"spaCy processing completed in {elapsed:.2f}s, remaining time: {remaining_time:.2f}s")
        
        if remaining_time <= 0:
            print("Timeout reached after spaCy processing")
            insights_summary = f"Partial analysis completed due to timeout.\n{spacy_summary}"
            return {
                'summary': insights_summary.strip(),
                'entities': spacy_insights.get('entities', {}),
                'key_phrases': spacy_insights.get('key_phrases', []),
                'topics': [],
                'sentiment': 'NEUTRAL',
                'transformer_summary': ''
            }
        
        # Process with sklearn for topics
        topics, topic_summary, _ = extract_insights_with_sklearn(text, model_cache)
        
        # Check for timeout again
        elapsed = time.time() - start_time
        remaining_time = timeout - elapsed
        print(f"sklearn processing completed in {elapsed:.2f}s, remaining time: {remaining_time:.2f}s")
        
        if remaining_time <= 0:
            print("Timeout reached after sklearn processing")
            insights_summary = f"""
            DOCUMENT INSIGHTS SUMMARY:
            
            {spacy_summary}
            
            {topic_summary}
            
            Document appears neutral in tone.
            """
            return {
                'summary': insights_summary.strip(),
                'entities': spacy_insights.get('entities', {}),
                'key_phrases': spacy_insights.get('key_phrases', []),
                'topics': topics,
                'sentiment': 'NEUTRAL',
                'transformer_summary': ''
            }
        
        # Process with transformers for sentiment and summary
        transformer_results, transformer_summary, _ = extract_insights_with_transformers(text, model_cache)
        
        # Combine all insights into a comprehensive summary
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
        insights_summary = f"Analysis partially completed due to processing timeout."
        
        if 'spacy_summary' in locals() and spacy_summary != "No text analysis available.":
            insights_summary += f"\n\n{spacy_summary}"
            
        if 'topic_summary' in locals() and topic_summary != "No topic information available.":
            insights_summary += f"\n\n{topic_summary}"
            
        return {
            'summary': insights_summary,
            'entities': spacy_insights.get('entities', {}) if 'spacy_insights' in locals() else {},
            'key_phrases': spacy_insights.get('key_phrases', []) if 'spacy_insights' in locals() else [],
            'topics': topics if 'topics' in locals() else [],
            'sentiment': transformer_results.get('sentiment', 'NEUTRAL') if 'transformer_results' in locals() else 'NEUTRAL',
            'transformer_summary': transformer_results.get('summary', '') if 'transformer_results' in locals() else ''
        }
    except Exception as e:
        print(f"Error generating comprehensive insights: {e}")
        return {
            'summary': f"Error during insights generation: {str(e)}",
            'entities': {},
            'key_phrases': [],
            'topics': [],
            'sentiment': 'NEUTRAL',
            'transformer_summary': ''
        }
