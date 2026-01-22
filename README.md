Fake News Detection using NLP
Overview

This repository contains the implementation of a Fake News Detection system as part of a Bachelor's thesis in Computer Science. The project explores two approaches for stance detection in news articles:

Lexical-Based Stance Detection with Logistic Regression

TF-IDF Vectorization with Logistic Regression

The main goal is to classify news articles into one of four stance categories:

Agree – headline and article convey the same message

Disagree – headline and article present opposing views

Discuss – article touches on the same topic without clear agreement/disagreement

Unrelated – article talks about a completely different topic from the headline

The system is evaluated on metrics such as accuracy, precision, recall, and F1-score to compare the effectiveness of these approaches.

This project is based on the Fake News Challenge dataset from Kaggle



Features

Data Preprocessing: Lowercasing, stopword removal, stemming, tokenization, removal of non-alphanumeric characters, and whitespace standardization.

Lexical Feature Extraction: Using word frequency and bigrams based on stance-specific dictionaries.

TF-IDF Feature Extraction: Using TfidfVectorizer to capture the importance of words across the corpus.

Machine Learning: Logistic Regression classifier for stance detection.

Model Validation: 5-fold cross-validation and train-validation-test split (60:20:20).

Performance Metrics: Accuracy, precision, recall, F1-score, and confusion matrices.

Results

Lexical-Based Model:

Accuracy: 77.77%

Precision: 75.20%

Recall: 77.77%

F1-score: 73.08%

Final thesis draft

TF-IDF Model:

Accuracy: 81.06%

Precision: 79.73%

Recall: 81.06%

F1-score: 77.49%

Final thesis draft

TF-IDF model outperforms the lexical-based model, demonstrating the importance of feature weighting in text classification.

Installation

Clone this repository:

git clone <REPO_URL>
cd <REPO_FOLDER>


Install required packages (recommended using venv or conda):

pip install -r requirements.txt


Requirements include:

Python >=3.8

pandas

numpy

scikit-learn

nltk

matplotlib

seaborn

Download the dataset from Fake News Challenge Kaggle Dataset
 and place train-bodies.csv and train-stances.csv in the data/ folder.

Usage

Preprocessing:

from preprocess import preprocess_data
df = preprocess_data('data/train-bodies.csv', 'data/train-stances.csv')


Training Lexical-Based Model:

from lexical_model import train_lexical_model
train_lexical_model(df)


Training TF-IDF Model:

from tfidf_model import train_tfidf_model
train_tfidf_model(df)


Evaluation & Visualization:

from evaluation import evaluate_model
evaluate_model(model, X_test, y_test)
