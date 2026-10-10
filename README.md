# 🚢 Titanic Survival Predictor

![Python](https://img.shields.io/badge/Python-1E3A8A?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-2563EB?logo=scikitlearn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-1E40AF?logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-1D4ED8?logo=plotly&logoColor=white)

A Streamlit app that predicts whether a Titanic passenger would have survived, using a Support Vector Machine. It also includes an interactive exploratory data analysis of the passenger data.

## What it does

- **Predict:** enter a passenger's class, sex, age, siblings/spouses aboard, parents/children aboard, fare and port of embarkation in the sidebar, then click **Predict Survival**. The result is *Survived* or *Did Not Survive*, plus the engineered feature values that were sent to the model.
- **Validate:** inputs are checked against valid ranges before prediction, and errors are listed clearly
- **Explore:** summary cards (total passengers, survived, perished, survival rate) and interactive Plotly charts in four tabs: **Class & Gender**, **Age**, **Fare** and **Family**

## Dataset

`titanic3.csv` is the extended Titanic passenger dataset, with 1,309 passengers and columns such as `pclass`, `survived`, `sex`, `age`, `sibsp`, `parch`, `fare` and `embarked`.

For the charts, the app drops unused columns (name, ticket, cabin, boat, body, home.dest), fills missing ages with the median and missing ports with the most common one.

## Approach / model

- **Features (9):** Pclass, Sex (male = 1), Age, SibSp, Parch, Fare, Embarked (S = 0, C = 1, Q = 2), FamilySize (SibSp + Parch + 1) and IsAlone
- **Scaling:** `StandardScaler` (saved as `scaler.pkl`), fitted during training
- **Model:** scikit-learn `SVC` with an RBF kernel (saved as `titanic_model.pkl`)
- The app's header states the model was trained on 1,047 passengers

## How to run

```bash
git clone https://github.com/JoshuaXcel/titanic-model.git
cd titanic-model
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` pins `scikit-learn==1.6.1`, the version the model was saved with.

## Project structure

```
app.py             Streamlit app: prediction, validation and EDA charts
titanic3.csv       Titanic passenger dataset
titanic_model.pkl  Trained SVM (RBF kernel)
scaler.pkl         Fitted StandardScaler
requirements.txt   Python dependencies
```

## Tech used

Python · scikit-learn (SVM, StandardScaler) · Streamlit · pandas · NumPy · Plotly · joblib

## Author

**Joshua Mudana (Excel)**, AI & Machine Learning Engineer

[Portfolio](https://joshuaxcel.github.io/joshuamudana-portfolio/) · [GitHub](https://github.com/JoshuaXcel) · [X](https://x.com/JoshuaXcell) · mudanajoshua@gmail.com
