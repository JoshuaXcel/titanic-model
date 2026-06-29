import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#  Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }

    .main { background-color: #0d1117; }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        color: #e8d5b7;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #8b9ab0;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2a40 100%);
        border: 1px solid #2d3a52;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b9ab0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #e8d5b7;
    }
    .metric-value.survived { color: #4ade80; }
    .metric-value.perished { color: #f87171; }

    .predict-survived {
        background: linear-gradient(135deg, #052e16, #064e28);
        border: 1px solid #4ade80;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .predict-perished {
        background: linear-gradient(135deg, #2d0a0a, #450e0e);
        border: 1px solid #f87171;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .predict-label {
        font-size: 0.85rem;
        color: #8b9ab0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .predict-result {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        margin-top: 0.3rem;
    }
    .survived-text { color: #4ade80; }
    .perished-text { color: #f87171; }

    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #e8d5b7;
        border-bottom: 1px solid #2d3a52;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .sidebar-note {
        font-size: 0.78rem;
        color: #8b9ab0;
        background: #1a2035;
        border-left: 3px solid #3b5bdb;
        padding: 0.6rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin-top: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2a40;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #c9d1e0 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    .validation-error {
        background: #2d0a0a;
        border: 1px solid #f87171;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        color: #f87171;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


#Loading model & scaler
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model  = load_model()
scaler = load_scaler()


#Load & cache Titanic reference data
@st.cache_data
def load_titanic_data():
    df = pd.read_csv("titanic3.csv")
    #Normalising the data
    col_map = {}
    for c in df.columns:
        if c.lower() == "sibsp":   col_map[c] = "SibSp"
        elif c.lower() == "parch": col_map[c] = "Parch"
        else:                      col_map[c] = c.capitalize()
    df = df.rename(columns=col_map)
    #dropping unwanted columns
    for col in ["Cabin", "Boat", "Body", "Home.dest", "Ticket", "Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.dropna(subset=["Survived", "Pclass", "Sex", "Age", "Fare"])
    return df

df = load_titanic_data()

#Feature engineering to match training
#9 features: Pclass, Sex(male=1), Age, SibSp, Parch, Fare,
#              Embarked_Q, Embarked_S, FamilySize

FEATURE_NAMES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                 "Embarked_Q", "Embarked_S", "FamilySize"]

#Validation rules
VALID_PCLASS    = [1, 2, 3]
AGE_MIN, AGE_MAX = 0.17, 80.0
SIBSP_MIN, SIBSP_MAX = 0, 8
PARCH_MIN, PARCH_MAX = 0, 6
FARE_MIN, FARE_MAX   = 0.0, 512.33
VALID_EMBARKED  = ["C", "Q", "S"]
VALID_SEX       = ["male", "female"]


def validate_input(pclass, sex, age, sibsp, parch, fare, embarked):
    errors = []
    if pclass not in VALID_PCLASS:
        errors.append(f"Pclass must be 1, 2, or 3. Got {pclass}.")
    if sex not in VALID_SEX:
        errors.append(f"Sex must be 'male' or 'female'. Got '{sex}'.")
    if not (AGE_MIN <= age <= AGE_MAX):
        errors.append(f"Age must be between {AGE_MIN} and {AGE_MAX}. Got {age}.")
    if not (SIBSP_MIN <= sibsp <= SIBSP_MAX):
        errors.append(f"SibSp must be between {SIBSP_MIN} and {SIBSP_MAX}. Got {sibsp}.")
    if not (PARCH_MIN <= parch <= PARCH_MAX):
        errors.append(f"Parch must be between {PARCH_MIN} and {PARCH_MAX}. Got {parch}.")
    if not (FARE_MIN <= fare <= FARE_MAX):
        errors.append(f"Fare must be between ${FARE_MIN:.2f} and ${FARE_MAX:.2f}. Got {fare}.")
    if embarked not in VALID_EMBARKED:
        errors.append(f"Embarked must be C, Q, or S. Got '{embarked}'.")
    return errors


def engineer_and_scale(pclass, sex, age, sibsp, parch, fare, embarked):
    sex_enc     = 1 if sex == "male" else 0
    embarked_q  = 1 if embarked == "Q" else 0
    embarked_s  = 1 if embarked == "S" else 0
    family_size = sibsp + parch + 1

    raw = np.array([[pclass, sex_enc, age, sibsp, parch,
                     fare, embarked_q, embarked_s, family_size]])
    #using the real scaler fitted during training
    scaled = scaler.transform(raw)
    return scaled


#SIDEBAR for Passenger Input
with st.sidebar:
    st.markdown("<div class='hero-title' style='font-size:1.6rem'>🚢 Passenger Details</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Enter passenger's information to predict.</div>", unsafe_allow_html=True)
    st.markdown("---")

    pclass = st.selectbox(
        "Passenger Class (Pclass)",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1st — Upper", 2: "2nd — Middle", 3: "3rd — Lower"}[x],
    )

    sex = st.selectbox("Sex", options=["male", "female"])

    age = st.slider(
        f"Age (valid: {AGE_MIN}–{AGE_MAX})",
        min_value=float(AGE_MIN),
        max_value=float(AGE_MAX),
        value=28.0,
        step=0.5,
    )

    sibsp = st.number_input(
        f"Siblings / Spouses aboard (SibSp, {SIBSP_MIN}–{SIBSP_MAX})",
        min_value=SIBSP_MIN,
        max_value=SIBSP_MAX,
        value=0,
        step=1,
    )

    parch = st.number_input(
        f"Parents / Children aboard (Parch, {PARCH_MIN}–{PARCH_MAX})",
        min_value=PARCH_MIN,
        max_value=PARCH_MAX,
        value=0,
        step=1,
    )

    fare = st.number_input(
        f"Ticket Fare in USD ({FARE_MIN}–{FARE_MAX:.2f})",
        min_value=float(FARE_MIN),
        max_value=float(FARE_MAX),
        value=32.20,
        step=0.50,
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["C", "Q", "S"],
        format_func=lambda x: {"C": "Cherbourg (C)", "Q": "Queenstown (Q)", "S": "Southampton (S)"}[x],
    )

    st.markdown("<div class='sidebar-note'>All inputs are validated against the original training data ranges before prediction.</div>", unsafe_allow_html=True)
    st.markdown("")
    predict_btn = st.button("Predict Survival", use_container_width=True, type="primary")

#MAIN — Hero + prediction output
st.markdown("<div class='hero-title'>Titanic Survival Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>SVM · RBF Kernel · StandardScaled · Trained on 1,047 passengers</div>", unsafe_allow_html=True)

#Top KPI row 
total     = len(df)
survived  = df["Survived"].sum()
perished  = total - survived
surv_rate = round(survived / total * 100, 1)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Total Passengers</div>
        <div class='metric-value'>{total:,}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Survived</div>
        <div class='metric-value survived'>{survived:,}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Perished</div>
        <div class='metric-value perished'>{perished:,}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Survival Rate</div>
        <div class='metric-value'>{surv_rate}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#Prediction block
if predict_btn:
    errors = validate_input(pclass, sex, age, sibsp, parch, fare, embarked)

    if errors:
        st.markdown("<div class='section-header'>⚠️ Validation Errors</div>", unsafe_allow_html=True)
        for e in errors:
            st.markdown(f"<div class='validation-error'>✖ {e}</div>", unsafe_allow_html=True)
        st.info("Fix the inputs above and try again. Values must match the training data distribution.")
    else:
        X_scaled = engineer_and_scale(pclass, sex, age, sibsp, parch, fare, embarked)
        pred = model.predict(X_scaled)[0]
        family_size = sibsp + parch + 1

        if pred == 1:
            st.markdown("""<div class='predict-survived'>
                <div class='predict-label'>Prediction</div>
                <div class='predict-result survived-text'>✓ Survived</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='predict-perished'>
                <div class='predict-label'>Prediction</div>
                <div class='predict-result perished-text'>✕ Did Not Survive</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        #Show the engineered + scaled feature vector
        with st.expander("🔍 See how your input was processed"):
            sex_enc    = 1 if sex == "male" else 0
            embarked_q = 1 if embarked == "Q" else 0
            embarked_s = 1 if embarked == "S" else 0
            raw_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Raw value": [pclass, sex_enc, age, sibsp, parch, fare,
                              embarked_q, embarked_s, family_size],
                "Scaled value": X_scaled[0].round(4),
            })
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

st.markdown("---")

#  EDA Section
st.markdown("<div class='section-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

DARK_BG   = "#0d1117"
CARD_BG   = "#1a2035"
GREEN     = "#4ade80"
RED       = "#d62728"
GOLD      = "#f5f3f0"
BLUE      = "#60a5fa"
TEXT      = "#c9d1e0"

chart_layout = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font_color=TEXT,
    font_family="Inter",
    margin=dict(t=40, b=30, l=10, r=10),
)

tab1, tab2, tab3, tab4 = st.tabs(["CLASS & GENDER", "AGE", "FARE", "FAMILY"])


#Tab 1: Class & Gender
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        surv_class = df.groupby(["Pclass", "Survived"]).size().reset_index(name="Count")
        surv_class["Status"] = surv_class["Survived"].map({0: "Perished", 1: "Survived"})
        surv_class["Class"] = surv_class["Pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})

        fig = px.bar(surv_class, x="Class", y="Count", color="Status",
                     barmode="group",
                     color_discrete_map={"Survived": "#2ca02c", "Perished": "#d62728"},
                     title="Survival by Passenger Class")
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        surv_sex = df.groupby(["Sex", "Survived"]).size().reset_index(name="Count")
        surv_sex["Status"] = surv_sex["Survived"].map({0: "Perished", 1: "Survived"})

        fig = px.bar(surv_sex, x="Sex", y="Count", color="Status",
                     barmode="group",
                     color_discrete_map={"Survived": "#2ca02c", "Perished": "#d62728"},
                     title="Survival by Gender")
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    #Heatmap: class × sex survival rate
    pivot = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack() * 100
    pivot.index = ["1st Class", "2nd Class", "3rd Class"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, "#d62728"], [0.5, "#f97316"], [1, "#2ca02c"]],
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        showscale=True,
    ))
    fig.update_layout(title="Survival Rate % by Class & Gender", **chart_layout)
    st.plotly_chart(fig, use_container_width=True)


#Tab 2: Age 
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(df, x="Age", color="Survived",
                           color_discrete_map={0: "#d62728", 1: "#2ca02c"},
                           nbins=30, barmode="overlay", opacity=0.75,
                           title="Age Distribution by Survival",
                           labels={"Survived": "Status"})
        fig.for_each_trace(lambda t: t.update(name="Survived" if t.name == "1" else "Perished"))
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.box(df, x="Pclass", y="Age",
                     color="Survived",
                     color_discrete_map={0: "#d62728", 1: "#2ca02c"},
                     title="Age Spread by Class & Survival",
                     labels={"Pclass": "Class", "Survived": "Status"})
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    #Age buckets
    df["AgeBucket"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                              labels=["Child (0–12)", "Teen (13–18)",
                                      "Adult (19–35)", "Middle (36–60)", "Senior (60+)"])
    surv_age = df.groupby("AgeBucket", observed=True)["Survived"].mean().reset_index()
    surv_age.columns = ["Age Group", "Survival Rate"]
    surv_age["Survival Rate"] = (surv_age["Survival Rate"] * 100).round(1)

    fig = px.bar(surv_age, x="Age Group", y="Survival Rate",
                 color="Survival Rate",
                 color_continuous_scale=[[0, "#d62728"], [0.5, "#f5f3f0"], [1, "#2ca02c"]],
                 title="Survival Rate by Age Group (%)")
    fig.update_layout(**chart_layout, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


#Tab 3: Fare
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.box(df, x="Survived", y="Fare",
                     color="Survived",
                     color_discrete_map={0: "#d62728", 1: "#2ca02c"},
                     title="Fare Distribution by Survival",
                     labels={"Survived": "Status"})
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Perished", "Survived"])
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        df["FareBucket"] = pd.cut(df["Fare"],
                                   bins=[0, 10, 30, 100, 513],
                                   labels=["Budget (<$10)", "Economy ($10–30)",
                                           "Premium ($30–100)", "Luxury ($100+)"])
        surv_fare = df.groupby("FareBucket", observed=True)["Survived"].mean().reset_index()
        surv_fare.columns = ["Fare Tier", "Survival Rate"]
        surv_fare["Survival Rate"] = (surv_fare["Survival Rate"] * 100).round(1)

        fig = px.bar(surv_fare, x="Fare Tier", y="Survival Rate",
                     color="Survival Rate",
                     color_continuous_scale=[[0, "#d62728"], [0.5, "#f5f3f0"], [1, "#2ca02c"]],
                     title="Survival Rate by Fare Tier (%)")
        fig.update_layout(**chart_layout, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df, x="Age", y="Fare", color="Survived",
                     color_discrete_map={0: "#d62728", 1: "#2ca02c"},
                     opacity=0.65, size_max=8,
                     title="Age vs. Fare — Coloured by Survival",
                     labels={"Survived": "Status"})
    fig.for_each_trace(lambda t: t.update(name="Survived" if t.name == "1" else "Perished"))
    fig.update_layout(**chart_layout)
    st.plotly_chart(fig, use_container_width=True)


#Tab 4: Family
with tab4:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    col_a, col_b = st.columns(2)

    with col_a:
        surv_fam = df.groupby("FamilySize")["Survived"].mean().reset_index()
        surv_fam.columns = ["Family Size", "Survival Rate"]
        surv_fam["Survival Rate"] = (surv_fam["Survival Rate"] * 100).round(1)

        fig = px.line(surv_fam, x="Family Size", y="Survival Rate",
                      markers=True,
                      title="Survival Rate by Family Size (%)")
        fig.update_traces(line_color=BLUE, marker_color=GOLD)
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        df["TravelType"] = df["FamilySize"].apply(
            lambda x: "Solo" if x == 1 else ("Small Family (2–4)" if x <= 4 else "Large Family (5+)")
        )
        surv_type = df.groupby("TravelType")["Survived"].mean().reset_index()
        surv_type.columns = ["Travel Type", "Survival Rate"]
        surv_type["Survival Rate"] = (surv_type["Survival Rate"] * 100).round(1)

        fig = px.pie(surv_type, values="Survival Rate", names="Travel Type",
                     title="Average Survival by Travel Type",
                     color_discrete_sequence=[BLUE, GREEN, GOLD])
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    surv_embark = df.groupby(["Embarked", "Survived"]).size().reset_index(name="Count")
    surv_embark["Status"] = surv_embark["Survived"].map({0: "Perished", 1: "Survived"})
    surv_embark["Port"] = surv_embark["Embarked"].map(
        {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    )

    fig = px.bar(surv_embark, x="Port", y="Count", color="Status",
                 barmode="group",
                 color_discrete_map={"Survived": "#2ca02c", "Perished": "#d62728"},
                 title="Survival by Port of Embarkation")
    fig.update_layout(**chart_layout)
    st.plotly_chart(fig, use_container_width=True)

#Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.8rem'>"
    "Model: SVM (RBF kernel) · Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize · "
    "Preprocessing: StandardScaler"
    "</div>",
    unsafe_allow_html=True,
)