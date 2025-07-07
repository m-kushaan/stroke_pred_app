# Stroke Prediction Web App 🚑

This project is a machine learning-powered web application designed to predict the likelihood of stroke based on healthcare data. It includes data preprocessing, exploratory data analysis (EDA), model training using the K-Nearest Neighbors (KNN) algorithm, and an interactive Streamlit-based user interface.

---

## 📋 Features
- **Stroke Risk Prediction** using optimized KNN model (achieved 97% accuracy).
- Multi-page Streamlit app:
  - **Home Page:** Dataset summary, project description, and insights.
  - **Visualization Page:** Interactive plots and EDA insights.
  - **Prediction Page:** Real-time stroke risk prediction based on user input.
- Model evaluation using **cross-validation** and **confusion matrix**.
- Fully deployed and accessible web app.

---

## 🔧 Technologies Used
- **Python**
- **Pandas & NumPy** – Data Manipulation
- **Matplotlib & Seaborn & Plotly** – Data Visualization
- **scikit-learn** – Machine Learning Model
- **Streamlit** – Web App Development
- **Jupyter Notebook** – Data Exploration & Modeling

---

## 🚀 Project Structure
```
|── Home.py                  # Home Page
├── pages/
│   ├── Visualization.py     # EDA & Visualizations
│   └── Prediction.py        # Stroke Prediction
│
├── knn\_model.sav           # Trained KNN Model
│
├── healthcare-dataset-stroke-data.csv  # Dataset
├── requirements.txt         # Python Dependencies
├── README.md                # Project Description
└── stroke_prediction_project # Jupyter Notebook (EDA + Modeling)

````

---

## 📊 Dataset Information
The dataset includes:
- Age, Gender, Hypertension, Heart Disease, Marital Status
- Work Type, Residence Type, Average Glucose Level, BMI
- Smoking Status, Stroke Outcome (Target Variable)

---

## ⚙️ How to Run Locally
1. Clone the repository:
```bash
git clone <repo-url>
cd stroke-prediction-webapp
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app/Home.py
```

---

## 📈 Model Performance

* **Algorithm:** K-Nearest Neighbors (KNN)
* **Cross-validation:** 5-fold
* **Accuracy:** 97%
* Evaluation Metrics: Confusion Matrix, Precision, Recall, F1-Score.

---

## 💡 Key Learnings

* End-to-End ML Model Deployment.
* Data Preprocessing & EDA.
* Hyperparameter Tuning & Model Validation.
* Web App Development using Streamlit.

---

## 📌 License

This project is open-sourced for educational purposes.

---

## 🙋‍♀️ Developed by:

**Kushaan Mahajan**

Feel free to fork this repo and customize it for your learning projects!


