import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = "data/jobs_with_demand.csv"
PLOT_DIR = "static/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# ===============================
#Demand Level Distribution
# ===============================
plt.figure(figsize=(6,6))
df["demand_label"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Demand Level Distribution")
plt.ylabel("")
plt.savefig(f"{PLOT_DIR}/demand_label.png")
plt.close()

# ===============================
#Jobs by Location
# ===============================
plt.figure(figsize=(8,5))
df["location"].value_counts().plot(kind="bar")
plt.title("Jobs by Location")
plt.xlabel("Location")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/location_jobs.png")
plt.close()

# ===============================
#Jobs by Industry
# ===============================
plt.figure(figsize=(8,5))
df["industry"].value_counts().plot(kind="bar")
plt.title("Jobs by Industry")
plt.xlabel("Industry")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/industry_jobs.png")
plt.close()

# ===============================
#Top Job Roles
# ===============================
plt.figure(figsize=(8,5))
df["job_title"].value_counts().head(10).plot(kind="bar")
plt.title("Top Job Roles")
plt.xlabel("Job Role")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/job_roles.png")
plt.close()

# ===============================
#Top Skills
# ===============================
skills_series = df["skills"].str.split(",").explode().str.strip()

plt.figure(figsize=(8,5))
skills_series.value_counts().head(10).plot(kind="bar")
plt.title("Top Skills in Job Market")
plt.xlabel("Skill")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/skills.png")
plt.close()

# ===============================
#Salary vs Demand
# ===============================
plt.figure(figsize=(6,5))
df.boxplot(column="salary_lpa", by="demand_label")
plt.title("Salary vs Demand Level")
plt.suptitle("")
plt.xlabel("Demand Level")
plt.ylabel("Salary (LPA)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/salary_vs_demand.png")
plt.close()

print("ALL 6 DASHBOARD GRAPHS GENERATED SUCCESSFULLY")
