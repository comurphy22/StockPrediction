"""
Financial Sentiment Classifier using FinancialPhraseBank
Trains a simple but effective sentiment model on expert-labeled financial text
Then applies it to our news headlines instead of VADER
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import re

print("="*70)
print("TRAINING FINANCIAL SENTIMENT CLASSIFIER")
print("="*70)
print()

# Load FinancialPhraseBank data
print("[1/5] Loading FinancialPhraseBank dataset...")
df = pd.read_csv('data/SentimentAnalysis/all-data.csv', 
                 names=['sentiment', 'text'], 
                 encoding='latin-1')

print(f"[OK] Loaded {len(df):,} labeled sentences")
print()
print("Sentiment Distribution:")
print(df['sentiment'].value_counts())
print()

# Simple text preprocessing
def preprocess_text(text):
    """Basic text cleaning for financial text"""
    # Lowercase
    text = text.lower()
    # Remove special characters but keep numbers (important for financial text)
    text = re.sub(r'[^a-z0-9\s%.]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

print("[2/5] Preprocessing text...")
df['text_clean'] = df['text'].apply(preprocess_text)
print("[OK] Text preprocessed")
print()

# Encode labels: positive=1, neutral=0, negative=-1
label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['label'] = df['sentiment'].map(label_map)

# Split data
print("[3/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

print(f"[OK] Train: {len(X_train)} | Test: {len(X_test)}")
print()

# Create TF-IDF features
print("[4/5] Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,      # Top 5000 words
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=2,               # Must appear in at least 2 documents
    max_df=0.95,            # Remove very common words
    strip_accents='unicode',
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"[OK] Feature matrix shape: {X_train_tfidf.shape}")
print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,} terms")
print()

# Train classifier
print("[5/5] Training Logistic Regression classifier...")
print()

classifier = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    solver='lbfgs',
    multi_class='multinomial'
)

# Cross-validation
print("Running 5-fold cross-validation...")
cv_scores = cross_val_score(classifier, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
print()

# Train on full training set
classifier.fit(X_train_tfidf, y_train)
print("[OK] Model trained")
print()

# Evaluate on test set
y_pred = classifier.predict(X_test_tfidf)
test_acc = (y_pred == y_test).mean()

print("="*70)
print("MODEL PERFORMANCE")
print("="*70)
print()
print(f"Test Accuracy: {test_acc:.3f}")
print()

# Classification report
label_names = ['negative (-1)', 'neutral (0)', 'positive (1)']
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_names))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
print(pd.DataFrame(cm, 
                   index=['True Neg', 'True Neu', 'True Pos'],
                   columns=['Pred Neg', 'Pred Neu', 'Pred Pos']))
print()

# Show most informative features
print("="*70)
print("MOST INFORMATIVE FEATURES")
print("="*70)
print()

feature_names = vectorizer.get_feature_names_out()

# Get coefficients for each class
for class_idx, class_name in [(-1, 'NEGATIVE'), (0, 'NEUTRAL'), (1, 'POSITIVE')]:
    # Find the index in classifier classes
    coef_idx = list(classifier.classes_).index(class_idx)
    coefs = classifier.coef_[coef_idx]
    
    # Top 10 features
    top_indices = coefs.argsort()[-10:][::-1]
    top_features = [(feature_names[i], coefs[i]) for i in top_indices]
    
    print(f"{class_name} indicators:")
    for feat, coef in top_features:
        print(f"  {feat:20s} {coef:+.3f}")
    print()

# Save model
print("="*70)
print("SAVING MODEL")
print("="*70)
print()

model_path = 'models/financial_sentiment_classifier.pkl'
vectorizer_path = 'models/financial_sentiment_vectorizer.pkl'

import os
os.makedirs('models', exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)
print(f"[OK] Saved classifier: {model_path}")

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"[OK] Saved vectorizer: {vectorizer_path}")

print()
print("="*70)
print("NEXT STEP: Use this model to score our news headlines!")
print("="*70)
print()
print("Example usage:")
print("""
import pickle

# Load model
with open('models/financial_sentiment_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/financial_sentiment_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Score a headline
headline = "Apple stock surges on strong earnings report"
headline_tfidf = vectorizer.transform([headline.lower()])
sentiment_score = classifier.predict(headline_tfidf)[0]  # Returns -1, 0, or 1

# Get probabilities for confidence
probs = classifier.predict_proba(headline_tfidf)[0]
# probs = [prob_negative, prob_neutral, prob_positive]
""")

print()
print("[OK] Training complete!")
