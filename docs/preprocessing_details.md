# Preprocessing Details for Phishing Email Classifier

This document outlines the text preprocessing steps and model choice.

## Preprocessing Steps
1. **Tokenization**: Split text into individual words using NLTK's `word_tokenize`.
2. **Lowercasing**: Convert all text to lowercase for consistency.
3. **Stop Word Removal**: Remove common words (e.g., "the", "is") using NLTK's stopwords list.
4. **Stemming**: Reduce words to their root form (e.g., "running" → "run") using Porter Stemmer.
5. **Vectorization**: Convert text to numerical features using TF-IDF with a max of 5000 features.

## Model Choice
- **Logistic Regression**: Chosen for its simplicity, interpretability, and effectiveness in binary text classification.
- TF-IDF vectorization captures word importance, making it suitable for phishing detection.

These steps reduce noise and focus on meaningful content, improving classification accuracy.
