import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open("spam_classifier.pkl", "rb") as file:
    model = pickle.load(file)

# Sample email data (Replace this with actual testing data)
sample_emails = ["Congratulations! You won a lottery. Click the link to claim.", 
                 "Hi John, can we schedule a meeting for tomorrow?",
                 "Limited-time offer! Get 50% off on all purchases."]

# Load processed data features (Ensuring column compatibility)
df = pd.read_csv("processed_data.csv")
feature_names = df.columns

# Convert sample email text into features (Dummy example, refine if needed)
X_sample = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)  # Placeholder

# Predict spam or not spam
predictions = model.predict(X_sample)

# Display results
for email, pred in zip(sample_emails, predictions):
    label = "Spam" if pred == 1 else "Not Spam"
    print(f"📩 Email: {email} → Prediction: {label}")