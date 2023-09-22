import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import streamlit as st
import joblib

# Cek apakah data stopwords sudah diunduh, jika belum, unduh
try:
    nltk.data.find('corpora/stopwords.zip')
    print("Lagi dicari stopwordsnya")
except LookupError:
    nltk.download('stopwords')
    print("Download bruh")

data = pd.read_csv("labeled_data.csv")

data["labels"] = data["class"].map({0: "Hate Speech Detected", 1: "Offensive Language", 2: "No Hate and Offensive"})

data = data[["tweet", "labels"]]

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.pipeline import Pipeline
import string
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join([word for word in word_tokenize(text) if word.lower() not in stopword])
    return text

def pos_tag_text(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    pos_tags = [tag for word, tag in tags]
    return " ".join(pos_tags)

data["cleaned_tweet"] = data["tweet"].apply(clean)
data["pos_tags"] = data["cleaned_tweet"].apply(pos_tag_text)

x = np.array(data["pos_tags"])
y = np.array(data["labels"])

# Optimize the vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(x)

# Reduce data size for faster iterations
X_train, _, y_train, _ = train_test_split(X, y, train_size=0.1, random_state=42)

# Muat model dari file jika ada, jika tidak latih model baru
try:
    clf = joblib.load('naive_bayes_model.pkl')
    print("Model loaded from file.")
except FileNotFoundError:
    print("Model not found. Training new model...")

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Simpan model ke file
    joblib.dump(clf, 'naive_bayes_model.pkl')
    print("New model trained and saved to file.")

# Define the test data
X_test, y_test = tfidf_vectorizer.transform(data["pos_tags"]).toarray(), y

# Limit the number of samples for testing
X_test, y_test = X_test[:1000], y_test[:1000]

# Evaluasi model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

offensive_hate_words = ["fucking", "dick", "hate"]

# Antarmuka Streamlit
def hate_speech_detection():
    st.title("[Hate Speech Detection System]")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        cleaned_sample = clean(sample)
        pos_tagged_sample = pos_tag_text(cleaned_sample)

        if any(word in cleaned_sample for word in offensive_hate_words):
            automatic_label = "Hate Speech Detected"
        else:
            data = tfidf_vectorizer.transform([pos_tagged_sample]).toarray()
            a = clf.predict(data)
            automatic_label = a[0]

        st.title("Automatic Classification: " + automatic_label)

hate_speech_detection()
