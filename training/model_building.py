import pandas as pd
import numpy as np
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load dataset
df = pd.read_csv("data/jobs_with_demand.csv")
print("Dataset loaded:", df.shape)

# STEP 2: Feature Engineering
df["skills"] = df["skills"].str.lower()

# TF-IDF for skills
tfidf = TfidfVectorizer(max_features=300)
X_skills = tfidf.fit_transform(df["skills"])

# Encode categorical columns
le_location = LabelEncoder()
le_industry = LabelEncoder()

df["location_enc"] = le_location.fit_transform(df["location"])
df["industry_enc"] = le_industry.fit_transform(df["industry"])

# Combine all features
X = np.hstack((
    X_skills.toarray(),
    df[["experience", "salary_lpa", "location_enc", "industry_enc"]].values
))

# Target variable
y = df["demand_label"]

# STEP 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# STEP 4: Model Building
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# STEP 5: Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# STEP 6: Save model & encoders
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/job_demand_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(le_location, "models/location_encoder.pkl")
joblib.dump(le_industry, "models/industry_encoder.pkl")

print("Model and encoders saved successfully!")