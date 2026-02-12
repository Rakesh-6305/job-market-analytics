import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import Counter


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'jobs_with_demand.csv')
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'plots')

os.makedirs(PLOT_DIR, exist_ok=True)


def generate_plots():
    df = pd.read_csv(DATA_PATH)

    # Demand Level Distribution
    plt.figure(figsize=(6, 6))
    df["demand_label"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=140)
    plt.title("Demand Level Distribution")
    plt.ylabel("")
    plt.savefig(os.path.join(PLOT_DIR, "demand_label.png"))
    plt.close()

    # Jobs by Location
    plt.figure(figsize=(8, 5))
    df["location"].value_counts().plot(kind="bar")
    plt.title("Jobs by Location")
    plt.xlabel("Location")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "location_jobs.png"))
    plt.close()

    # Jobs by Industry
    plt.figure(figsize=(8, 5))
    df["industry"].value_counts().plot(kind="bar")
    plt.title("Jobs by Industry")
    plt.xlabel("Industry")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "industry_jobs.png"))
    plt.close()

    # Top Job Roles
    plt.figure(figsize=(8, 5))
    df["job_title"].value_counts().head(10).plot(kind="bar")
    plt.title("Top Job Roles")
    plt.xlabel("Job Role")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "job_roles.png"))
    plt.close()

    # Top Skills
    skills_series = df["skills"].str.split(",").explode().str.strip()
    plt.figure(figsize=(8, 5))
    skills_series.value_counts().head(10).plot(kind="bar")
    plt.title("Top Skills in Job Market")
    plt.xlabel("Skill")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "skills.png"))
    plt.close()

    # Salary vs Demand
    plt.figure(figsize=(6, 5))
    df.boxplot(column="salary_lpa", by="demand_label")
    plt.title("Salary vs Demand Level")
    plt.suptitle("")
    plt.xlabel("Demand Level")
    plt.ylabel("Salary (LPA)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "salary_vs_demand.png"))
    plt.close()

    # Summary stats to return
    total_jobs = len(df)
    remote_pct = int((df['job_title'].str.contains('remote', case=False, na=False).sum() / max(total_jobs,1)) * 100)
    top_skills = skills_series.value_counts().head(5).to_dict()
    demand_counts = df['demand_label'].value_counts().to_dict()
    industry_counts = df['industry'].value_counts().head(5).to_dict()

    summary = {
        'total_jobs': int(total_jobs),
        'remote_pct': int(remote_pct),
        'top_skills': top_skills,
        'demand_counts': demand_counts,
        'industry_counts': industry_counts
    }

    return summary


if __name__ == '__main__':
    print('Generating plots...')
    print(generate_plots())
