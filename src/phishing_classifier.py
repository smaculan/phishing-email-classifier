#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import logging

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Text preprocessing
def preprocess_text(text):
    """Preprocess email text: tokenize, remove stopwords, stem."""
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(t) for t in tokens)

# Training function
def train_model(data_path, model_path, vectorizer_path):
    """Train the phishing email classifier."""
    df = pd.read_csv(data_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        logging.error("CSV must have 'text' and 'label' columns.")
        return False
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    logging.info(f"Model saved to {model_path}, Vectorizer saved to {vectorizer_path}")
    return True

# Prediction function
def predict_email(text, model_path, vectorizer_path):
    """Classify a new email as phishing or legitimate."""
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        logging.error("Model or vectorizer file not found.")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)[0]
    label = "Phishing" if prediction == 1 else "Legitimate"
    logging.info(f"Email classified as: {label}")

def main():
    parser = argparse.ArgumentParser(description="Phishing Email Classifier")
    parser.add_argument('--train', help="Path to training CSV file")
    parser.add_argument('--predict', help="Text of email to classify")
    parser.add_argument('--model', default='model.pkl', help="Path to save/load model")
    parser.add_argument('--vectorizer', default='vectorizer.pkl', help="Path to save/load vectorizer")
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.train, args.model, args.vectorizer)
    elif args.predict:
        predict_email(args.predict, args.model, args.vectorizer)
    else:
        logging.error("Specify either --train or --predict.")

if __name__ == "__main__":
    main()
