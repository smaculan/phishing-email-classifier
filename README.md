# Phishing Email Classifier

## Description
This project classifies emails as phishing or legitimate using NLP and Logistic Regression. It preprocesses text and trains a model to detect phishing attempts.

## Prerequisites
- Python 3.8+

## Installation
1. **Clone the Repository**: Extract this ZIP file.
2. **Install Dependencies**: Run `pip install -r requirements.txt`.
3. **Download NLTK Data**: The script downloads the required NLTK data on the first run.

## Usage
### Training Mode
Train the model with a dataset:
```bash
python src/phishing_classifier.py --train data/emails.csv --model model.pkl --vectorizer vectorizer.pkl
