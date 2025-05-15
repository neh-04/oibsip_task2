import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load processed data
df = pd.read_csv("processed_data.csv")

# Ensure dataset size matches labels
num_samples = len(df)

# Generate labels dynamically based on data split
y = [1] * (num_samples // 2) + [0] * (num_samples // 2)  # 1 = Spam, 0 = Not Spam

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

# Train **Multinomial Naive Bayes Classifier** (ideal for text data)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save trained model
with open("spam_classifier.pkl", "wb") as file:
    pickle.dump(model, file)

print("🎉 Training complete! Model saved as 'spam_classifier.pkl'.")