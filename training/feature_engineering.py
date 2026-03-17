import pandas as pd

# STEP 1: Load dataset
df = pd.read_csv("data/jobs_1000_plus.csv")

print("Dataset loaded")
print("Shape:", df.shape)

# STEP 2: Create Demand Score
df["demand_score"] = df.groupby("job_title")["job_title"].transform("count")

# STEP 3: Create Demand Label using percentile-based thresholds
# This ensures a balanced distribution across High/Medium/Low
df["demand_label"] = pd.qcut(
    df["demand_score"],
    q=3,
    labels=["Low", "Medium", "High"]
)

print("\n--- Demand Label Distribution ---")
print(df["demand_label"].value_counts())
print()
print(df[["job_title", "demand_score", "demand_label"]].drop_duplicates("job_title").sort_values("demand_score"))

# STEP 4: Simulate Remote Work (Since original data lacks 'Remote' keywords)
import numpy as np
np.random.seed(42) # For consistent simulation
mask = np.random.rand(len(df)) < 0.35
df.loc[mask, "job_title"] = df.loc[mask, "job_title"] + " (Remote)"

# STEP 5: Save updated dataset
df.to_csv("data/jobs_with_demand.csv", index=False)

print("\nFeature engineering completed and file saved!")