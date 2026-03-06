# 🛡️ Phishing Detection – Hands-on 1 Training

This repository contains materials for a hands-on training on phishing detection using machine learning and explainable AI. Participants will explore EDA, train multiple models, evaluate their performance, and use both Streamlit and Flask apps for deployment and interpretability.

---

## 📁 Folder Structure

```
phishing-detection-hands-on/
├── data/                     # Input dataset (phishing.csv)
├── models/                   # Trained model files (pkl)
├── metrics/                  # Classification reports & confusion matrices in JSON
├── notebooks/                # EDA and model comparison notebooks
│   ├── Phishing_EDA.ipynb
│   └── Compare_Models.ipynb
├── templates/                # HTML form used by Flask app
│   └── form.html
├── app.py                   # Flask web app
├── streamlit_app.py         # Streamlit dashboard
├── requirements.txt         # Required Python packages
├── train_and_evaluate_all_models.py  # Training script
├── README.md
└── .devcontainer/           # Codespaces configuration
    ├── devcontainer.json
    └── Dockerfile
```

---

## 📊 Contents

### 1. **EDA Notebook**
- `notebooks/Phishing_EDA.ipynb`
- Feature exploration, visualizations, and detailed interpretation.

### 2. **Model Comparison**
- `notebooks/Compare_Models.ipynb`
- Side-by-side metrics and visual comparison of three models: Logistic Regression, Random Forest, and XGBoost.

### 3. **Model Training**
- `train_and_evaluate_all_models.py`
- Trains all models and stores results in `models/` and `metrics/` folders.

### 4. **Apps**
- `streamlit_app.py`: Streamlit UI with SHAP explanations
- `app.py`: Flask form-based app for prediction

---

## 🚀 Running in GitHub Codespaces

Codespaces automatically sets up the Python environment via `.devcontainer/`. Once the Codespace starts:

```bash
pip install -r requirements.txt
```

> Alternatively, use the `Rebuild Container` option if dependencies were not automatically installed.

### 💡 Tips:
- Use the VS Code interface to open and run the notebooks.
- All files and folders are structured for easy exploration.

---

## 🔍 Running Apps

### 🔴 Streamlit
```bash
streamlit run streamlit_app.py
```

### 🌐 Flask
```bash
python app.py
```

Both apps will allow participants to experiment with real predictions, SHAP visualizations, and natural language explanations.

---

## ⚙️ Dev Container Setup (Codespaces)

`.devcontainer/devcontainer.json`
```json
{
  "name": "phishing-detection-env",
  "image": "mcr.microsoft.com/devcontainers/python:3.10",
  "features": {},
  "postCreateCommand": "pip install -r requirements.txt",
  "customizations": {
    "vscode": {
      "settings": {},
      "extensions": ["ms-python.python"]
    }
  }
}
```

`.devcontainer/Dockerfile`
```Dockerfile
# No custom image needed; using base Python container
```

---

## ✨ Credits

This training setup was developed to help learners understand the full ML pipeline for phishing detection and build interpretable AI applications.
