import pandas as pd

# Load dataset
df = pd.read_csv("data/jobs_1000_plus.csv")

print("Initial Shape:", df.shape)
print(df.head())
#Check Missing Values
print(df.isnull().sum())
#Standardize Text
df["skills"] = df["skills"].str.lower()
df["job_title"] = df["job_title"].str.lower()

# Save Cleaned Data
df.to_csv("data/jobs_cleaned.csv", index=False)
print("Data cleaning completed. Saved to data/jobs_cleaned.csv")