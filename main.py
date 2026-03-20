import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import numpy as np
import pandas as pd
import time
import os
import csv
from app.dashboard_plots import generate_plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'app', 'templates'), 
            static_folder=os.path.join(BASE_DIR, 'app', 'static'))
app.secret_key = 'job-market-analytics-2024'

# Load models with robust path resolution
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_model(filename):
    """Load a joblib model file from the models directory."""
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"[OK] Loading {filename} from {path}")
        return joblib.load(path)
    # Fallback for environments where path might be relative
    fallback = os.path.join('models', filename)
    if os.path.exists(fallback):
        return joblib.load(fallback)
    raise FileNotFoundError(f"Model file not found: {filename}")

# Initialize models to None to avoid NameErrors if loading fails
model = tfidf = le_loc = le_ind = None

try:
    model = load_model("job_demand_model.pkl")
    tfidf = load_model("tfidf.pkl")
    le_loc = load_model("location_encoder.pkl")
    le_ind = load_model("industry_encoder.pkl")
    print("[OK] All models loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")

def get_market_insights():
    """Compute key market insights from the dataset for the home page."""
    try:
        csv_path = os.path.join(DATA_DIR, 'jobs_with_demand.csv')
        df = pd.read_csv(csv_path)

        # Top skills
        all_skills = df['skills'].str.split(',').explode().str.strip().str.lower()
        top_skills = all_skills.value_counts().head(6).to_dict()

        # Average salary
        avg_salary = round(df['salary_lpa'].mean(), 1)
        max_salary = int(df['salary_lpa'].max())
        min_salary = int(df['salary_lpa'].min())

        # Top locations
        top_locations = df['location'].value_counts().head(5).to_dict()

        # Top industries
        top_industries = df['industry'].value_counts().head(5).to_dict()

        # Top roles
        top_roles = df['job_title'].value_counts().head(5).to_dict()

        # Total jobs
        total_jobs = len(df)

        # Demand distribution
        demand_dist = df['demand_label'].value_counts().to_dict()

        return {
            'top_skills': top_skills,
            'avg_salary': avg_salary,
            'max_salary': max_salary,
            'min_salary': min_salary,
            'top_locations': top_locations,
            'top_industries': top_industries,
            'top_roles': top_roles,
            'total_jobs': total_jobs,
            'demand_dist': demand_dist,
        }
    except Exception as e:
        print(f"[WARNING] Could not compute insights: {e}")
        return {
            'top_skills': {'python': 450, 'sql': 380, 'ml': 320},
            'avg_salary': 13.5,
            'max_salary': 25,
            'min_salary': 3,
            'top_locations': {'Bangalore': 200, 'Delhi': 180},
            'top_industries': {'IT': 300},
            'top_roles': {'ML Engineer': 138},
            'total_jobs': 1200,
            'demand_dist': {'High': 400, 'Medium': 400, 'Low': 400},
        }

def get_smart_suggestions(skills_str, demand, experience, salary):
    """Generate smart suggestions based on user input + model prediction."""
    skills_list = [s.strip().lower() for s in skills_str.split(',') if s.strip()]

    role_map = {
        'python': ['Data Scientist', 'ML Engineer', 'Software Engineer', 'AI Engineer'],
        'ml': ['ML Engineer', 'Data Scientist', 'AI Engineer'],
        'ai': ['AI Engineer', 'ML Engineer', 'Data Scientist'],
        'sql': ['Data Analyst', 'Business Analyst', 'Data Scientist', 'Software Engineer'],
        'java': ['Software Engineer'],
        'javascript': ['Web Developer'],
        'html': ['Web Developer'],
        'css': ['Web Developer'],
        'react': ['Web Developer'],
        'tensorflow': ['ML Engineer', 'AI Engineer'],
        'deep learning': ['ML Engineer', 'AI Engineer'],
        'aws': ['Cloud Engineer', 'DevOps Engineer'],
        'docker': ['DevOps Engineer', 'Cloud Engineer'],
        'kubernetes': ['DevOps Engineer', 'Cloud Engineer'],
        'linux': ['DevOps Engineer', 'Cyber Security Analyst', 'Cloud Engineer'],
        'security': ['Cyber Security Analyst'],
        'networking': ['Cyber Security Analyst'],
        'excel': ['Business Analyst', 'Data Analyst'],
        'power bi': ['Data Analyst', 'Business Analyst'],
        'communication': ['Business Analyst'],
        'oops': ['Software Engineer'],
        'azure': ['Cloud Engineer'],
        'nlp': ['AI Engineer', 'ML Engineer'],
        'statistics': ['Data Scientist'],
    }

    role_scores = {}
    for skill in skills_list:
        for role in role_map.get(skill, []):
            role_scores[role] = role_scores.get(role, 0) + 1
    recommended_roles = sorted(role_scores.items(), key=lambda x: -x[1])[:3]
    recommended_roles = [r[0] for r in recommended_roles] if recommended_roles else ['Software Engineer']

    base = 4
    exp_bonus = experience * 1.8
    demand_bonus = {'High': 6, 'Medium': 3, 'Low': 0}.get(demand, 2)
    skill_bonus = len(skills_list) * 0.8
    low_sal = max(3, round(base + exp_bonus + demand_bonus - 2))
    high_sal = round(base + exp_bonus + demand_bonus + skill_bonus + 4)
    salary_range = f"{low_sal} - {high_sal} LPA"

    high_demand_skills = ['python', 'sql', 'ml', 'deep learning', 'aws', 'docker']
    missing_skills = [s for s in high_demand_skills if s not in skills_list][:4]

    tips = []
    if len(skills_list) <= 2:
        tips.append("Add more complementary skills to increase your marketability")
    if 'python' not in skills_list:
        tips.append("Python is the #1 in-demand skill — consider learning it")
    if experience < 2:
        tips.append("Build projects and internships to compensate for limited experience")
    if demand == "High":
        tips.append("Your skills are in high demand — leverage this for better offers!")
    elif demand == "Low":
        tips.append("Consider upskilling in trending technologies like AI/ML/Cloud")

    return {
        'recommended_roles': recommended_roles,
        'salary_range': salary_range,
        'missing_skills': missing_skills,
        'tips': tips[:3],
    }

market_insights = get_market_insights()

@app.route("/")
def home():
    l_classes = le_loc.classes_ if le_loc is not None else []
    i_classes = le_ind.classes_ if le_ind is not None else []
    
    return render_template(
        "index.html",
        locations=l_classes,
        industries=i_classes,
        insights=market_insights
    )

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})

@app.route("/predict", methods=["POST"])
def predict():
    skills = request.form["skills"].lower()
    location = request.form["location"]
    industry = request.form["industry"]
    experience = int(request.form["experience"])
    salary = int(request.form["salary"])

    experience = min(max(experience, 0), 20)
    salary = min(max(salary, 2), 50)

    if any(m is None for m in [model, tfidf, le_loc, le_ind]):
        flash("System is currently recovering. Please try again in a few moments.")
        return redirect(url_for('home'))

    skill_vec = tfidf.transform([skills]).toarray()
    loc_enc = le_loc.transform([location])[0]
    ind_enc = le_ind.transform([industry])[0]

    X = np.hstack((skill_vec, [[experience, salary, loc_enc, ind_enc]]))

    confidence = None
    try:
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        prob_map = {cls: float(probs[i]) for i, cls in enumerate(classes)}

        if prob_map.get("High", 0) > 0.6:
            demand = "High"
        elif prob_map.get("Medium", 0) > 0.4:
            demand = "Medium"
        else:
            demand = classes[int(np.argmax(probs))]

        num_skills = len([s.strip() for s in skills.split(',') if s.strip()])
        if num_skills <= 1 and demand == "High":
            demand = "Medium"

        confidence = prob_map.get(demand, float(max(probs)))
    except Exception:
        demand = model.predict(X)[0]
        confidence = None

    score = 0
    for s in ["python","ml","ai","sql","deep learning"]:
        if s in skills: score += 10
    if demand == "High": score += 30
    elif demand == "Medium": score += 20
    else: score += 10
    score = min(score, 100)

    required = ["python","ml","sql","deep learning"]
    missing = [s for s in required if s not in skills]
    remote_pct = 35
    suggestions = get_smart_suggestions(skills, demand, experience, salary)

    return render_template(
        "result.html",
        demand=demand, score=score, missing=missing,
        remote_pct=remote_pct, confidence=round(confidence*100, 1) if confidence is not None else None,
        suggestions=suggestions, input_skills=skills, input_location=location,
        input_industry=industry, input_experience=experience, input_salary=salary
    )

@app.route("/dashboard")
def dashboard():
    role_filter = request.args.get('role', 'All')
    min_salary = request.args.get('min_salary', '')
    try:
        summary = generate_plots(role_filter=role_filter, min_salary=min_salary)
        if 'roles' not in summary:
            summary['roles'] = market_insights.get('top_roles', {}).keys()
    except Exception as e:
        print(f"[ERROR] Dashboard error: {e}")
        summary = {'total_jobs': 0, 'remote_pct': 0, 'key_insights': ["Error loading data."], 'roles': []}
    return render_template("dashboard.html", summary=summary, current_role=role_filter, current_salary=min_salary)

TRADES_PATH = os.path.join(BASE_DIR, 'data', 'trades.csv')
# Only create directory if it doesn't exist to avoid issues on read-only FS
if not os.path.exists(os.path.dirname(TRADES_PATH)):
    try:
        os.makedirs(os.path.dirname(TRADES_PATH), exist_ok=True)
    except OSError:
        pass

@app.route('/trade', methods=['GET', 'POST'])
def trade():
    if request.method == 'POST':
        action, instrument, quantity = request.form.get('action'), request.form.get('instrument'), request.form.get('quantity')
        try: quantity = int(quantity)
        except: flash('Quantity must be an integer'); return redirect(url_for('trade'))
        from datetime import datetime
        ts = datetime.utcnow().isoformat()
        with open(TRADES_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([ts, action, instrument, quantity, request.form.get('notes', '')])
        return render_template('trade_result.html', action=action, instrument=instrument, quantity=quantity, timestamp=ts)
    return render_template('trade.html')

if __name__ == "__main__":
    app.run(debug=True)
