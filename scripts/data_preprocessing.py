import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords (Run once)
nltk.download("stopwords")

# Load dataset (Adjust path if necessary)
df = pd.read_csv("data.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(word for word in text.split() if word not in stopwords.words("english"))  # Remove stopwords
    return text

# Apply cleaning
df["Cleaned Text"] = df["Email Text"].apply(clean_text)

# TF-IDF Vectorization with trigrams
vectorizer = TfidfVectorizer(ngram_range=(1,3))  # ✅ Includes unigrams, bigrams, and trigrams
X = vectorizer.fit_transform(df["Cleaned Text"])

# Save processed data
pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).to_csv("processed_data.csv", index=False)

print("✅ Preprocessing complete! Saved as 'processed_data.csv'.")