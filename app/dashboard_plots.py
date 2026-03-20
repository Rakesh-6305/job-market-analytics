import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import Counter


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'jobs_with_demand.csv')
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'plots')

# PLOT_DIR is handled by matplotlib logic


import io
import base64

def generate_plots(role_filter=None, min_salary=None):
    df = pd.read_csv(DATA_PATH)

    # Apply filters
    if role_filter and role_filter != 'All':
        df = df[df['job_title'] == role_filter]
    
    if min_salary:
        try:
            df = df[df['salary_lpa'] >= int(min_salary)]
        except:
            pass

    if df.empty:
        return {'total_jobs': 0, 'remote_pct': 0, 'key_insights': [], 'plots': {}}

    plots = {}

    def get_base64_plot():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64

    # Demand Level Distribution
    plt.figure(figsize=(6, 6))
    df["demand_label"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=140, colors=['#10b981', '#f59e0b', '#ef4444'])
    plt.title("Demand Level Distribution")
    plt.ylabel("")
    plots['demand_label'] = get_base64_plot()

    # Jobs by Location
    plt.figure(figsize=(10, 6))
    df["location"].value_counts().head(10).plot(kind="bar", color='#6366f1')
    plt.title("Top Hiring Locations")
    plt.xlabel("Location")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plots['location_jobs'] = get_base64_plot()

    # Jobs by Industry
    plt.figure(figsize=(10, 6))
    df["industry"].value_counts().head(10).plot(kind="bar", color='#10b981')
    plt.title("Jobs by Industry")
    plt.xlabel("Industry")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plots['industry_jobs'] = get_base64_plot()

    # Top Job Roles
    plt.figure(figsize=(10, 6))
    df["job_title"].value_counts().head(10).plot(kind="barh", color='#f59e0b').invert_yaxis()
    plt.title("Top Job Roles")
    plt.xlabel("Count")
    plt.ylabel("Job Role")
    plt.tight_layout()
    plots['job_roles'] = get_base64_plot()

    # Top Skills
    skills_series = df["skills"].str.split(",").explode().str.strip().str.lower()
    plt.figure(figsize=(10, 6))
    skills_series.value_counts().head(12).plot(kind="bar", color='#8b5cf6')
    plt.title("Top In-Demand Skills")
    plt.xlabel("Skill")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plots['skills'] = get_base64_plot()

    # Salary Distribution
    plt.figure(figsize=(10, 6))
    df['salary_lpa'].hist(bins=20, color='#14b8a6', edgecolor='white')
    plt.title("Salary Distribution (LPA)")
    plt.xlabel("Salary (LPA)")
    plt.ylabel("Number of Jobs")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plots['salary_dist'] = get_base64_plot()

    # Key Insights
    key_insights = []
    
    # 1. Most common role
    top_role = df['job_title'].value_counts().idxmax()
    key_insights.append(f"<b>Top Role:</b> {top_role} is the most frequent job title in this segment.")
    
    # 2. Avg Salary
    avg_sal = round(df['salary_lpa'].mean(), 1)
    key_insights.append(f"<b>Avg Salary:</b> ₹{avg_sal} LPA is the average compensation offered.")
    
    # 3. Remote percentage
    remote_jobs = (df['job_title'].str.contains('remote', case=False, na=False) | 
                   df['location'].str.contains('remote', case=False, na=False)).sum()
    remote_pct = int((remote_jobs / len(df)) * 100)
    key_insights.append(f"<b>Remote Work:</b> Approximately {remote_pct}% of roles mention remote flexibility.")
    
    # 4. Top Skill
    top_skill = skills_series.value_counts().idxmax()
    key_insights.append(f"<b>Key Skill:</b> <u>{top_skill.title()}</u> is the most sought-after skill.")
    
    # 5. Highest industry
    top_ind = df['industry'].value_counts().idxmax()
    key_insights.append(f"<b>Hot Sector:</b> {top_ind} is leading the hiring activity.")

    # Summary
    summary = {
        'total_jobs': len(df),
        'remote_pct': remote_pct,
        'avg_salary': avg_sal,
        'key_insights': key_insights,
        'roles': sorted(df['job_title'].unique().tolist()),
        'top_skills': skills_series.value_counts().head(5).to_dict(),
        'plots': plots
    }

    return summary


if __name__ == '__main__':
    print('Generating plots...')
    print(generate_plots())
