#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the datasets
bodies_path = r'C:\Users\Vishnu\OneDrive\Desktop\ml\train_bodies.csv'
stances_path = r'C:\Users\Vishnu\OneDrive\Desktop\ml\train_stances.csv'
bodies_df = pd.read_csv(bodies_path)
stances_df = pd.read_csv(stances_path)

# Merge the datasets
merged_df = stances_df.merge(bodies_df, on='Body ID', how='left')

# Preprocess the text
def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'^[a-z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

merged_df['Headline'] = merged_df['Headline'].apply(preprocess)
merged_df['articleBody'] = merged_df['articleBody'].apply(preprocess)

# Extract lexical features for the first model
def lexical_features(text):
    words = text.split()
    stance_words = {'agree': ['agree', 'accept', 'support'],
                    'disagree': ['disagree', 'reject', 'oppose'],
                    'discuss': ['discuss', 'debate', 'argue'],
                    'unrelated': ['unrelated', 'irrelevant']}
    
    features = {}

    # Count occurrence of stance-related words
    for stance, stance_list in stance_words.items():
        count = sum(word in stance_list for word in words)
        features[stance] = count
    
    # Count individual words
    word_counts = {word: words.count(word) for word in words}
    features.update(word_counts)
    
    # Count bigrams
    n = 2  # bigrams
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        features[ngram] = features.get(ngram, 0) + 1
    
    return features

merged_df['LexicalFeatures'] = merged_df['articleBody'].apply(lexical_features)

# Split the data into training, validation, and testing sets in one step
X = merged_df[['Headline', 'articleBody']]
y = merged_df['Stance']

# Perform a single split to create training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the k-fold cross-validation strategy
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics
lexical_accuracies = []
lexical_precisions = []
lexical_recalls = []
lexical_f1_scores = []

tfidf_accuracies = []
tfidf_precisions = []
tfidf_recalls = []
tfidf_f1_scores = []

# Define the DictVectorizer and SelectKBest for lexical features
vec = DictVectorizer()
selector = SelectKBest(score_func=chi2, k=1000)
logistic_model = LogisticRegression(max_iter=1000)

# Define the TfidfVectorizer for TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
logistic_model_tfidf = LogisticRegression(max_iter=1000)

# Perform k-fold cross-validation for both models
for train_index, val_index in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Lexical features
    X_train_sparse = vec.fit_transform(X_train_fold['articleBody'].apply(lexical_features))
    X_val_sparse = vec.transform(X_val_fold['articleBody'].apply(lexical_features))
    X_train_selected = selector.fit_transform(X_train_sparse, y_train_fold)
    X_val_selected = selector.transform(X_val_sparse)

    # Train and validate lexical model
    logistic_model.fit(X_train_selected, y_train_fold)
    y_val_pred = logistic_model.predict(X_val_selected)
    lexical_accuracies.append(accuracy_score(y_val_fold, y_val_pred))
    lexical_precisions.append(precision_score(y_val_fold, y_val_pred, average='weighted'))
    lexical_recalls.append(recall_score(y_val_fold, y_val_pred, average='weighted'))
    lexical_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='weighted'))

    # TF-IDF features
    X_train_vec_tfidf = tfidf_vectorizer.fit_transform(X_train_fold['Headline'] + ' ' + X_train_fold['articleBody'])
    X_val_vec_tfidf = tfidf_vectorizer.transform(X_val_fold['Headline'] + ' ' + X_val_fold['articleBody'])

    # Train and validate TF-IDF model
    logistic_model_tfidf.fit(X_train_vec_tfidf, y_train_fold)
    y_val_pred_tfidf = logistic_model_tfidf.predict(X_val_vec_tfidf)
    tfidf_accuracies.append(accuracy_score(y_val_fold, y_val_pred_tfidf))
    tfidf_precisions.append(precision_score(y_val_fold, y_val_pred_tfidf, average='weighted'))
    tfidf_recalls.append(recall_score(y_val_fold, y_val_pred_tfidf, average='weighted'))
    tfidf_f1_scores.append(f1_score(y_val_fold, y_val_pred_tfidf, average='weighted'))

# Calculate average performance metrics for lexical features
avg_lexical_accuracy = np.mean(lexical_accuracies)
avg_lexical_precision = np.mean(lexical_precisions)
avg_lexical_recall = np.mean(lexical_recalls)
avg_lexical_f1_score = np.mean(lexical_f1_scores)

print("Cross-Validation Performance with Lexical Features:")
print(f'Accuracy: {avg_lexical_accuracy}')
print(f'Precision: {avg_lexical_precision}')
print(f'Recall: {avg_lexical_recall}')
print(f'F1-score: {avg_lexical_f1_score}')

# Calculate average performance metrics for TF-IDF features
avg_tfidf_accuracy = np.mean(tfidf_accuracies)
avg_tfidf_precision = np.mean(tfidf_precisions)
avg_tfidf_recall = np.mean(tfidf_recalls)
avg_tfidf_f1_score = np.mean(tfidf_f1_scores)

print("Cross-Validation Performance with TF-IDF:")
print(f'Accuracy: {avg_tfidf_accuracy}')
print(f'Precision: {avg_tfidf_precision}')
print(f'Recall: {avg_tfidf_recall}')
print(f'F1-score: {avg_tfidf_f1_score}')

# Final evaluation on the test set

# Lexical features
X_train_sparse = vec.fit_transform(X_train['articleBody'].apply(lexical_features))
X_test_sparse = vec.transform(X_test['articleBody'].apply(lexical_features))
X_train_selected = selector.fit_transform(X_train_sparse, y_train)
X_test_selected = selector.transform(X_test_sparse)

logistic_model.fit(X_train_selected, y_train)
y_test_pred = logistic_model.predict(X_test_selected)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print("Test Performance with Lexical Features:")
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1-score: {test_f1}')
print(f'Classification Report:\n{classification_report(y_test, y_test_pred)}')

# TF-IDF features
X_train_vec_tfidf = tfidf_vectorizer.fit_transform(X_train['Headline'] + ' ' + X_train['articleBody'])
X_test_vec_tfidf = tfidf_vectorizer.transform(X_test['Headline'] + ' ' + X_test['articleBody'])

logistic_model_tfidf.fit(X_train_vec_tfidf, y_train)
y_test_pred_tfidf = logistic_model_tfidf.predict(X_test_vec_tfidf)

test_accuracy_tfidf = accuracy_score(y_test, y_test_pred_tfidf)
test_precision_tfidf = precision_score(y_test, y_test_pred_tfidf, average='weighted')
test_recall_tfidf = recall_score(y_test, y_test_pred_tfidf, average='weighted')
test_f1_tfidf = f1_score(y_test, y_test_pred_tfidf, average='weighted')

print("Test Performance with TF-IDF:")
print(f'Accuracy: {test_accuracy_tfidf}')
print(f'Precision: {test_precision_tfidf}')
print(f'Recall: {test_recall_tfidf}')
print(f'F1-score: {test_f1_tfidf}')
print(f'Classification Report:\n{classification_report(y_test, y_test_pred_tfidf)}')


# In[2]:


import matplotlib.pyplot as plt

# Data for Lexical Features-based model
metrics_lexical = {
    'Accuracy': 0.7777888944472237,
    'Precision': 0.7520886102720722,
    'Recall': 0.7777888944472237,
    'F1-score': 0.7308067433124839
}

# Data for TF-IDF-based model
metrics_tfidf = {
    'Accuracy': 0.8106053026513257,
    'Precision': 0.7973015229275842,
    'Recall': 0.8106053026513257,
    'F1-score': 0.7749455437297513
}

# Function to plot bar graph
def plot_metrics(metrics, title, color, bar_width=0.5):
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=color, alpha=0.7, width=bar_width)
    
    plt.ylim(0, 1)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
    plt.show()

# Plot for Lexical Features-based model
plot_metrics(metrics_lexical, 'Test Performance with Lexical Features', 'lightblue', bar_width=0.4)

# Plot for TF-IDF-based model
plot_metrics(metrics_tfidf, 'Test Performance with TF-IDF', 'lightgreen', bar_width=0.4)


# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# Data for Lexical Features-based model
metrics_lexical = {
    'Accuracy': 0.7777888944472237,
    'Precision': 0.7520886102720722,
    'Recall': 0.7777888944472237,
    'F1-score': 0.7308067433124839
}

# Data for TF-IDF-based model
metrics_tfidf = {
    'Accuracy': 0.8106053026513257,
    'Precision': 0.7973015229275842,
    'Recall': 0.8106053026513257,
    'F1-score': 0.7749455437297513
}

# Function to plot bar graph with lines
def plot_metrics_comparison(metrics_lexical, metrics_tfidf, title):
    categories = list(metrics_lexical.keys())
    values_lexical = list(metrics_lexical.values())
    values_tfidf = list(metrics_tfidf.values())
    bar_width = 0.35  # Width of each bar
    
    plt.figure(figsize=(12, 6))
    
    # Plot Lexical Features-based model
    plt.bar(np.arange(len(categories)), values_lexical, color='lightblue', alpha=0.7, width=bar_width, label='Lexical Features')
    
    # Plot TF-IDF-based model
    plt.bar(np.arange(len(categories)) + bar_width, values_tfidf, color='lightgreen', alpha=0.7, width=bar_width, label='TF-IDF')
    
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(np.arange(len(categories)) + bar_width / 2, categories)  # Set x-axis tick positions and labels
    plt.legend(loc='lower right')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
    plt.show()

# Plot the comparison
plot_metrics_comparison(metrics_lexical, metrics_tfidf, 'Comparison of Test Performance Metrics')


# In[4]:


import pandas as pd
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix

# Assuming y_test, y_test_pred, y_test_pred_tfidf are already defined

# Generate confusion matrices
conf_matrix_lexical = confusion_matrix(y_test, y_test_pred)
conf_matrix_tfidf = confusion_matrix(y_test, y_test_pred_tfidf)

# Convert confusion matrices to DataFrames
df_lexical = pd.DataFrame(conf_matrix_lexical, index=["agree", "disagree", "discuss", "unrelated"], columns=["agree", "disagree", "discuss", "unrelated"])
df_tfidf = pd.DataFrame(conf_matrix_tfidf, index=["agree", "disagree", "discuss", "unrelated"], columns=["agree", "disagree", "discuss", "unrelated"])

# Define HTML table style
table_style = """
<style>
table {
    font-size: 18px;
    width: 50%;
    margin: 25px 0;
    border-collapse: collapse;
}
th, td {
    padding: 12px 15px;
    text-align: center;
}
th {
    background-color: #f2f2f2;
    font-weight: bold;
}
</style>
"""

# Create HTML code
html_lexical = df_lexical.to_html(classes='dataframe', border=0)
html_tfidf = df_tfidf.to_html(classes='dataframe', border=0)

# Combine the style and the tables
html_combined = table_style + "<h2>Confusion Matrix for Logistic Regression with Lexical Features:</h2>" + html_lexical
html_combined += table_style + "<h2>Confusion Matrix for Logistic Regression with TF-IDF:</h2>" + html_tfidf

# Display in Jupyter Notebook
display(HTML(html_combined))

# To save the HTML output to a file
with open("confusion_matrix_output.html", "w") as f:
    f.write(html_combined)


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix heatmap
def plot_confusion_matrix_heatmap(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['agree', 'disagree', 'discuss', 'unrelated'], yticklabels=['agree', 'disagree', 'discuss', 'unrelated'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrix heatmap for Lexical Features-based model
plot_confusion_matrix_heatmap(y_test, y_test_pred, 'Confusion Matrix for Lexical Features-based Model')

# Plot confusion matrix heatmap for TF-IDF-based model
plot_confusion_matrix_heatmap(y_test, y_test_pred_tfidf, 'Confusion Matrix for TF-IDF-based Model')


# In[ ]:




