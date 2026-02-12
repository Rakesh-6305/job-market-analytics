from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import time
import os
import csv

# Import generate_plots robustly so the app can be started either with
# `python -m app.app` (package) or `python app/app.py` (script).
try:
    from app.dashboard_plots import generate_plots
except Exception:
    # When running as a script, the `app` package path may not be set.
    # Add the app directory to sys.path and try a local import.
    import sys
    app_dir = os.path.dirname(__file__)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    try:
        from dashboard_plots import generate_plots
    except Exception:
        # As a last resort, attempt package import again after adding parent dir
        parent = os.path.abspath(os.path.join(app_dir, '..'))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from app.dashboard_plots import generate_plots

app = Flask(__name__)

model = joblib.load("models/job_demand_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")
le_loc = joblib.load("models/location_encoder.pkl")
le_ind = joblib.load("models/industry_encoder.pkl")

@app.route("/")
def home():
    return render_template(
        "index.html",
        locations=le_loc.classes_,
        industries=le_ind.classes_
    )

@app.route("/predict", methods=["POST"])
def predict():

    # 🔄 Fake Loading (5 sec)
    time.sleep(5)

    skills = request.form["skills"].lower()
    location = request.form["location"]
    industry = request.form["industry"]
    experience = int(request.form["experience"])
    salary = int(request.form["salary"])

    # Backend safety
    experience = min(max(experience, 0), 20)
    salary = min(max(salary, 2), 50)

    # ML prediction
    skill_vec = tfidf.transform([skills]).toarray()
    loc_enc = le_loc.transform([location])[0]
    ind_enc = le_ind.transform([industry])[0]

    X = np.hstack((skill_vec, [[experience, salary, loc_enc, ind_enc]]))
    demand = model.predict(X)[0]
    # get confidence/probability for predicted class
    confidence = None
    try:
        probs = model.predict_proba(X)
        # find index of predicted class
        class_index = list(model.classes_).index(demand)
        confidence = float(probs[0][class_index])
    except Exception:
        confidence = None

    # Opportunity score
    score = 0
    for s in ["python","ml","ai","sql","deep learning"]:
        if s in skills:
            score += 10

    if demand == "High":
        score += 30
    elif demand == "Medium":
        score += 20
    else:
        score += 10

    score = min(score, 100)

    # Skill gap
    required = ["python","ml","sql","deep learning"]
    missing = [s for s in required if s not in skills]

    # Default Remote percentage (placeholder — replace with real calc if needed)
    remote_pct = 35

    return render_template(
        "result.html",
        demand=demand,
        score=score,
        missing=missing,
        remote_pct=remote_pct,
        confidence=round(confidence*100, 1) if confidence is not None else None
    )

@app.route("/dashboard")
def dashboard():
    # Regenerate dashboard plots and get summary stats
    try:
        summary = generate_plots()
        remote_pct = summary.get('remote_pct', 35)
    except Exception:
        remote_pct = 35
        summary = {}

    return render_template("dashboard.html", remote_pct=remote_pct, summary=summary)


# Simple market trade simulator: records mock trades to data/trades.csv
TRADES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trades.csv')
os.makedirs(os.path.dirname(TRADES_PATH), exist_ok=True)
if not os.path.exists(TRADES_PATH):
    with open(TRADES_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'action', 'instrument', 'quantity', 'notes'])


@app.route('/trade', methods=['GET', 'POST'])
def trade():
    if request.method == 'POST':
        action = request.form.get('action')
        instrument = request.form.get('instrument')
        quantity = request.form.get('quantity')
        notes = request.form.get('notes', '')

        # basic validation
        try:
            quantity = int(quantity)
        except Exception:
            flash('Quantity must be an integer', 'error')
            return redirect(url_for('trade'))

        from datetime import datetime
        ts = datetime.utcnow().isoformat()
        with open(TRADES_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([ts, action, instrument, quantity, notes])

        return render_template('trade_result.html', action=action, instrument=instrument, quantity=quantity, timestamp=ts)

    # GET -> render trade form
    return render_template('trade.html')

if __name__ == "__main__":
    app.run(debug=True)
