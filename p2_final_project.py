import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv('social_media_usage (2).csv')

def clean_sm(x):
    return np.where(x == 1, 1, 0)

s['sm_li'] = clean_sm(s["web1h"])

ss = s[['sm_li', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()

ss['income'] = ss['income'].apply(lambda x: x if x <= 9 else np.nan)
ss['education'] = ss['educ2'].apply(lambda x: x if x <= 8 else np.nan)
ss['age'] = ss['age'].apply(lambda x: x if x <= 97 else np.nan)

ss['parent'] = ss['par'].apply(lambda x: 1 if x == 1 else 0)
ss['married'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)
ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)

ss = ss.drop(columns=['par', 'marital', 'gender', 'educ2']).dropna()

print("Summary Statistics:")
sumstats = ss.describe()
cormatrix =ss.corr()
sumstats, cormatrix

y = ss["sm_li"]
X = ss[['age', 'education','income', 'parent', 'married', 'female']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify=y,
                                                   test_size=.2,
                                                   random_state=3125)

log_reg = LogisticRegression(class_weight= 'balanced')
log_reg.fit(X_train, y_train)


st.markdown("Hello! My name is Giovanna. Welcome to my app!")

st.markdown("Please provide your personal details below to predict your usage of LinkedIn:")

st.subheader("Are you a LinkedIn user?")
sm_li = st.selectbox(
    "Select an option:",
    options=[
        (1, "Yes, I use LinkedIn"),
        (0, "No, I do not use LinkedIn")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your approximate household income?")
income = st.selectbox(
    "Select income range:",
    options=[
        (1, "Less than $10,000"),
        (2, "$10,000 - $19,999"),
        (3, "$20,000 - $29,999"),
        (4, "$30,000 - $39,999"),
        (5, "$40,000 - $49,999"),
        (6, "$50,000 - $74,999"),
        (7, "$75,000 - $99,999"),
        (8, "$100,000 - $149,999"),
        (9, "$150,000 or more"),
        (10, "Don't Know"),
        (11, "Refused")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your highest level of education?")
educ2 = st.selectbox(
    "Select your highest level of education:",
    options=[
        (1, "Middle school or less"),
        (2, "High school incomplete"),
        (3, "High school graduate/GED"),
        (4, "Some college, no degree"),
        (5, "Associate degree"),
        (6, "Bachelor's degree"),
        (7, "Some postgraduate schooling, no degree"),
        (8, "Postgraduate or professional degree")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("Are you the parent of a child under 18 years of age?")
parent = st.radio(
    "Select your parental status:",
    options=[
        (1, "Yes"),
        (0, "No")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your marital status?")
married = st.radio(
    "Select your marital status:",
    options=[
        (1, "Married"),
        (0, "Not Married")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your gender?")
female = st.radio(
    "Select your gender:",
    options=[
        (1, "Female"),
        (0, "Male")
    ],
    format_func=lambda x: x[1]
)[0]

age = st.number_input("Enter your age:", min_value=1, max_value=98, step=1)

def sent_app(user_data):

    user_data = pd.DataFrame([user_data], columns=['income', 'educ2', 'parent','married', 'female', 'age'])

    probability = logreg.predict_proba(user_data)[0][1]
    classification = logreg.predict(user_data)[0]
    
    st.subheader("Results:")
    st.write(f"**Classification:** You are {'a LinkedIn user' if classification == 1 else 'not a LinkedIn user'}.")
    st.write(f"**Probability:** There is a {probability * 100:.2f}% chance that you are a LinkedIn user.")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100, 
        title={'text': f"LinkedIn User Probability: {probability * 100:.2f}%"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "pink"},
                {"range": [30, 70], "color": "purple"},
                {"range": [70, 100], "color": "blue"},],
            "bar": {"color": "gold"} }))
    return st.plotly_chart(fig)

if st.button("Submit"):
    user_data = {
        'income': income,
        'educ2': educ2,
        'parent': parent,
        'married': married,
        'female': female,
        'age': age
    }
    sent_app(user_data)