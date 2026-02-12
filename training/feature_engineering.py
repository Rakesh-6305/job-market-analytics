import pandas as pd

# STEP 1: Load dataset
df = pd.read_csv("data/jobs_1000_plus.csv")

print("Dataset loaded")
print("Shape:", df.shape)

# STEP 2: Create Demand Score
df["demand_score"] = df.groupby("job_title")["job_title"].transform("count")

# STEP 3: Create Demand Label
def label_demand(x):
    if x > 110:
        return "High"
    elif x > 100:
        return "Medium"
    else:
        return "Low"

df["demand_label"] = df["demand_score"].apply(label_demand)

print(df[["job_title", "demand_score", "demand_label"]].head())

# STEP 4: Save updated dataset
df.to_csv("data/jobs_with_demand.csv", index=False)

print("Feature engineering completed and file saved!")