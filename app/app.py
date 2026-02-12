from flask import Flask, render_template, request
import joblib
import numpy as np
import time

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
        remote_pct=remote_pct
    )

@app.route("/dashboard")
def dashboard():
    # Dashboard-level metric (placeholder)
    remote_pct = 35
    return render_template("dashboard.html", remote_pct=remote_pct)

if __name__ == "__main__":
    app.run(debug=True)
