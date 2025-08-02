import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("chat_moderation_dataset.csv")

y = (df['label'] == "poor_sportsmanship").astype(int)

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['message'])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "chat_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("✅ Model trained and saved!")
