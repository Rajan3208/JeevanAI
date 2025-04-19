import os
import pickle
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from transformers import pipeline
import tensorflow as tf
from keras.models import load_model
import concurrent.futures

# Load models only once
nlp = None
topic_model = None
vectorizer = None
sentiment_model = None
summarizer = None
sentiment_analyzer = None

def load_models():
    global nlp, sentiment_model
    
    # Load spaCy
    try:
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")
    except:
        try:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
    
    # Load your pre-trained sentiment model
    try:
        if sentiment_model is None and os.path.exists('models/sentiment_model.keras'):
            sentiment_model = load_model('models/sentiment_model.keras')
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        
    # Load any insights from your pickle file
    try:
        if os.path.exists('models/document_insights.pkl'):
            with open('models/document_insights.pkl', 'rb') as f:
                pre_trained_insights = pickle.load(f)
                # Use these insights in your analysis process
    except Exception as e:
        print(f"Error loading document insights: {e}")

def extract_insights_with_spacy(text):
    """Extract insights using spaCy"""
    if not nlp:
        return {}, "spaCy model not loaded."
    
    # Process only up to 100,000 characters to avoid memory issues
    text = text[:100000]
    doc = nlp(text)
    
    entities = {ent.text: ent.label_ for ent in doc.ents}
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    
    # Get important sentences based on entity density
    sentences = list(doc.sents)
    if sentences:
        important_sentences = sorted(
            sentences,
            key=lambda s: sum(1 for ent in doc.ents if ent.start >= s.start and ent.end <= s.end),
            reverse=True
        )[:3]
        summary = ' '.join([sent.text for sent in important_sentences])
    else:
        summary = "No complete sentences found."

    insights = {
        'entities': entities,
        'key_phrases': noun_chunks[:10],
        'summary': summary
    }

    return insights, f"Document contains {len(entities)} named entities including {', '.join(list(entities.keys())[:5]) if entities else 'none'}. Key topics focus on {', '.join(noun_chunks[:3]) if noun_chunks else 'unknown topics'}."

def extract_insights_with_sklearn(text, num_topics=3):
    """Extract topics using sklearn NMF"""
    if not topic_model or not vectorizer:
        return [], "Topic model not loaded.", None
    
    try:
        # Process with pre-trained topic model
        dtm = vectorizer.transform([text])
        topic_values = topic_model.transform(dtm)[0]
        
        # Get top topics
        topics = []
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(topic_model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
        
        topic_insights = f"Main topics: {', '.join([' & '.join(topic[:3]) for topic in topics])}"
        return topics, topic_insights
    except Exception as e:
        print(f"Error in topic extraction: {e}")
        return [], "Could not extract topics due to insufficient text data"

def extract_insights_with_transformers(text):
    """Extract insights using transformers"""
    if not sentiment_analyzer or not summarizer:
        return {}, "Transformer models not loaded.", None
    
    try:
        # Process text in chunks to avoid memory issues
        chunks = [text[i:i+512] for i in range(0, min(len(text), 5000), 512)]
        sentiments = []
        
        # Process in smaller batches
        for i in range(0, len(chunks), 5):
            batch = chunks[i:i+5]
            batch_sentiments = [sentiment_analyzer(chunk) for chunk in batch if chunk.strip()]
            sentiments.extend(batch_sentiments)
        
        if sentiments:
            sentiment_labels = [s[0]['label'] for s in sentiments if s]
            if sentiment_labels:
                overall_sentiment = max(set(sentiment_labels), key=sentiment_labels.count)
            else:
                overall_sentiment = "NEUTRAL"
                
            # Summarize only if text is long enough
            if len(text) > 100:
                summary = summarizer(text[:1024], max_length=100, min_length=30, do_sample=False)
                summary_text = summary[0]['summary_text']
            else:
                summary_text = text
                
            return {
                'sentiment': overall_sentiment,
                'summary': summary_text
            }, f"Document sentiment: {overall_sentiment}. {summary_text[:100]}..."
        else:
            return {'sentiment': 'NEUTRAL', 'summary': text[:100]}, "Document appears neutral in tone."
    except Exception as e:
        print(f"Error in transformer processing: {e}")
        return {'sentiment': 'UNKNOWN', 'summary': text[:100]}, "Unable to determine document sentiment."

def generate_insights(text, fast_mode=True):
    """Generate comprehensive insights from text"""
    # Load models
    model_status = load_models()
    print(f"Model status: {model_status}")
    
    results = {}
    summary_parts = []
    
    # Use concurrent processing for faster insights
    with concurrent.futures.ThreadPoolExecutor() as executor:
        spacy_future = executor.submit(extract_insights_with_spacy, text)
        topic_future = executor.submit(extract_insights_with_sklearn, text)
        
        # Only run transformer-based analysis if not in fast mode
        if not fast_mode:
            transformer_future = executor.submit(extract_insights_with_transformers, text)
        
        # Get spaCy results
        try:
            spacy_insights, spacy_summary = spacy_future.result()
            results['spacy'] = spacy_insights
            summary_parts.append(spacy_summary)
        except Exception as e:
            print(f"Error in spaCy processing: {e}")
        
        # Get topic modeling results
        try:
            topics, topic_summary = topic_future.result()
            results['topics'] = topics
            summary_parts.append(topic_summary)
        except Exception as e:
            print(f"Error in topic modeling: {e}")
        
        # Get transformer results if not in fast mode
        if not fast_mode:
            try:
                transformer_results, transformer_summary = transformer_future.result()
                results['transformer'] = transformer_results
                summary_parts.append(transformer_summary)
            except Exception as e:
                print(f"Error in transformer processing: {e}")
    
    # Compile summary
    insights_summary = "\n".join(summary_parts)
    
    return results, insights_summary