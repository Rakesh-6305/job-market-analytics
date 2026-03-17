# 📊 Job Market Analytics AI

A premium, data-driven web application that leverages Machine Learning to predict job demand and provide intelligent career insights. This project uses a Random Forest classifier trained on 1,000+ job postings to analyze the market and help job seekers optimize their career paths.

![Home Page](app/static/bg.jpg) *(Example context)*

## 🚀 Key Features

### 1. **Intelligent Prediction Engine**
- **Smart Demand Analysis:** Predicts if your skillset is in **High**, **Medium**, or **Low** demand using a Random Forest model.
- **Confidence Scoring:** Shows the model's certainty level for every prediction.
- **Opportunity Score:** A weighted 0-100% score representing your overall marketability.

### 2. **Career Guidance & Suggestions**
- **Recommended Roles:** Suggests the best job titles based on your current skills.
- **Salary Range Prediction:** Estimates your potential salary range in LPA (Lakhs Per Annum).
- **Skill Gap Analysis:** Identifies missing high-demand skills (e.g., Python, AWS, ML) and provides upskilling tips.

### 3. **Market Insights Dashboard**
- **Dynamic Filtering:** Sift through data by **Job Role** and **Salary Range**.
- **Visual Analytics:** Interactive charts for Demand Distribution, Salary Histograms, Top Skills, and Hiring Locations.
- **AI-Driven Insights:** Automaticaly highlighting hot sectors, top-paying roles, and remote work trends.

### 4. **Modern UI/UX**
- **Premium Design:** Glassmorphism, smooth animations, and a dark/light mode toggle.
- **Responsive Layout:** Fully optimized for mobile, tablet, and desktop viewing.

---

## 🛠️ Tech Stack

- **Backend:** Python (Flask)
- **Machine Learning:** Scikit-Learn (Random Forest, TF-IDF Vectorization)
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Frontend:** HTML5, CSS3, JavaScript (Three.js for background particles), Bootstrap 5

---

## 📁 Project Structure

```text
├── app/
│   ├── static/             # Static assets (images, CSS, plots)
│   ├── templates/          # HTML templates (Bootstrap 5 & Vanilla CSS)
│   ├── app.py              # Main Flask application logic
│   └── dashboard_plots.py  # Plotting and insights engine
├── data/
│   ├── jobs_1000_plus.csv   # Raw dataset
│   └── jobs_with_demand.csv # Processed dataset with engineered labels
├── models/
│   ├── job_demand_model.pkl # Trained Random Forest model
│   ├── tfidf.pkl            # TF-IDF Vectorizer for skills
│   └── ...                  # Label Encoders
├── training/
│   ├── data_cleaning.py     # Preprocessing scripts
│   ├── feature_engineering.py # Percentile-based demand labeling
│   └── model_building.py    # Training and evaluation logic
├── requirements.txt        # Python dependencies
└── README.md               # User documentation
```

---

## ⚙️ Setup & Installation

### 1. Prerequisite
Ensure you have **Python 3.8+** installed.

### 2. Installation
```bash
# Clone the repository (if applicable)
# Navigate to project directory
cd final

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
```bash
python -m app.app
```
Access the app at `http://127.0.0.1:5000`

---

## 📈 Model Performance
- **Accuracy:** ~97% on balanced test set.
- **Features:** Skills (TF-IDF), Location, Industry, Experience, Salary.
- **Labels:** Percentile-based splitting (Lo/Med/Hi) ensuring unbiased training.

---

## 👤 Author
Developed as a high-performance Data Science & ML Showcase project. 🚀
