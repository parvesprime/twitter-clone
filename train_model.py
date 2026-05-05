import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("sentiment_data.csv")

X = data["text"]
y = data["label"]

# TF-IDF with n-grams
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)

# Train classifier
model = LogisticRegression()
model.fit(X_tfidf, y)

# Save model & vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model trained and saved successfully.")
