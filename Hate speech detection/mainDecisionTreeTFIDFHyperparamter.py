import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
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
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Muat model dari file jika ada, jika tidak latih model baru
try:
    clf = joblib.load('decision_tree_model.pkl')
    print("Model loaded from file.")
except FileNotFoundError:
    print("Model not found. Training new model...")

    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [None, 10, 20, 30],
                  'min_samples_split': [2, 5, 10]}

    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    clf.fit(X_train, y_train)

    # Simpan model ke file
    joblib.dump(clf, 'decision_tree_model.pkl')
    print("New model trained and saved to file.")

# Evaluasi model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

offensive_hate_words = ["fucking", "kontol", "dick head", "hateword2"]


# Antarmuka Streamlit
def hate_speech_detection():
    st.title("[Hate Speech Detection System]")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        cleaned_sample = clean(sample)  # Membersihkan teks
        
        if any(word in cleaned_sample for word in offensive_hate_words):
            automatic_label = "Hate Speech Detected"
        else:
            data = tfidf_vectorizer.transform([cleaned_sample]).toarray()
            a = clf.predict(data)
            automatic_label = a[0]
        
        st.title("Automatic Classification: " + automatic_label)

hate_speech_detection()