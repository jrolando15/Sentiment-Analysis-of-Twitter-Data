# Sentiment-Analysis-of-Twitter-Data

# Project Description
This project implements a sentiment analysis model using various machine learning techniques to classify tweets from the Kaggle Sentiment140 dataset. The dataset consists of 1.6 million tweets labeled as either positive or negative. The goal is to preprocess the text data, extract features, and train a model to accurately predict the sentiment of new tweets.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Feature Extraction](#feature-extraction)
- [Model training](#model-training)
- [Model Evaluation](#model-evaluation)
- [License](#license)

# Installation
To run this project, you need to have Python installed along with the following libraries:
pandas
numpy
nltk
scikit-learn
lightgbm
matplotlib
seaborn
You can install the required libraries using the following command:
```bash
pip install pandas numpy nltk scikit-learn lightgbm matplotlib seaborn
```

# Usage
1. Clone the repository
```bash
git clone https://github.com/your_username/Sentiment-Analysis-Twitter.git
cd Sentiment-Analysis-Twitter
```

2. Run the script
```python
Sasm.ipynb
```

# Project Structure
```bash
Sentiment-Analysis-Twitter/
├── Sasm.ipynb              # Main script with the code
├── training.1600000.processed.noemoticon.csv  # Dataset file
├── README.md                          # Project README file
```
# Data Preprocessing
The dataset is loaded using pandas and cleaned by removing URLs, mentions, hashtags, numbers, and punctuation. The text is then converted to lowercase, tokenized, and stopwords are removed. Finally, stemming is applied to reduce words to their root form.

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df.columns = ["target", 'ids', 'data', 'flag', 'user', 'text']

# Function to clean the text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['tokens'] = df['clean_text'].apply(word_tokenize)

stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

stemmer = PorterStemmer()
df['processed_text'] = df['tokens'].apply(lambda x: " ".join(x))
```

# Feature Extraction
The TF-IDF vectorizer is used to convert the processed text data into numerical features.
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_text'])
y = df['target'].apply(lambda x: 1 if x == 4 else 0)
```

# Model Training 
The data is split into training and testing sets. A LightGBM model is trained on the training data.
```bash
from sklearn.model_selection import train_test_split
import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.05,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'is_unbalance': True,
    'random_state': 42
}

model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=20000,
    valid_sets=[lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_test, label=y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)
```

# Model Evaluation
The model is evaluated using accuracy, precision, recall, and F1-score. A confusion matrix is also plotted.
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred_binary))

conf_matrix = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
# License
This README template includes all the pertinent information about your project, such as installation instructions, usage, project structure, data processing, model training, model evaluation, and details about the web application. It also includes sections for contributing and licensing, which are important for open-source projects.
