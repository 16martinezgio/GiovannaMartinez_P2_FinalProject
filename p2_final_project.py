import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go

import pandas as pd
import numpy as np
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

s = pd.read_csv('social_media_usage.csv')


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

import streamlit as st

st.markdown("Hello! My name is Giovanna. Welcome to my app!")
st.markdown("### Welcome to the LinkedIn Usage Prediction App!")
st.markdown("#### Let's find out how likely you are to be a LinkedIn user based on your profile.")

st.markdown("Please provide your personal details below to predict your usage of LinkedIn:")

st.markdown("### Let's get to know you a little better!")
st.subheader("Are you on LinkedIn? Let's find out!")
sm_li = st.radio("Are you a LinkedIn user?", options=[(1, "Yes! I'm LinkedIn professional!"), (0, "Nope, not yet!")])

income = st.selectbox(
    "What's your income range? (No need to stress about it, just select your range!)",
    options=[(1, "Less than $10,000"), (2, "$10,000 - $19,999"), (3, "$20,000 - $29,999"), 
             (4, "$30,000 - $39,999"), (5, "$40,000 - $49,999"), (6, "$50,000 - $74,999"), 
             (7, "$75,000 - $99,999"), (8, "$100,000 - $149,999"), (9, "$150,000 or more")],
    format_func=lambda x: x[1]
)

educ2 = st.selectbox(
    "What’s your highest level of education?",
    options=[(1, "Middle school or less"), (2, "High school incomplete"), (3, "High school graduate/GED"), 
             (4, "Some college, no degree"), (5, "Associate degree"), (6, "Bachelor's degree"), 
             (7, "Some postgraduate schooling"), (8, "Postgraduate or professional degree")],
    format_func=lambda x: x[1]
)

parent = st.radio("Do you have any children under 18?", options=[(1, "Yes, I’m a parent!"), (0, "Nope, no kids here.")])
married = st.radio("Are you married?", options=[(1, "Yes, happily married!"), (0, "No, I'm single")])
female = st.radio("Are you a woman?", options=[(1, "Yes, I’m a lady!"), (0, "No, I’m a guy")])

age = st.number_input("How old are you? Don’t worry, we won't judge!", min_value=18, max_value=99, step=1)


def sent_app(user_data):
    user_data = pd.DataFrame([user_data], columns=['income', 'educ2', 'parent','married', 'female', 'age'])

    probability = log_reg.predict_proba(user_data)[0][1]
    classification = log_reg.predict(user_data)[0]

    st.markdown("### Here’s the big reveal! 🥁")
    st.write(f"**Classification:** Based on your profile, you are {'a LinkedIn user' if classification == 1 else 'not a LinkedIn user'}.")
    st.write(f"**Probability:** There’s a {probability * 100:.2f}% chance that you are a LinkedIn user!")

    if classification == 1:
        st.markdown("Wow, you are **LinkedIn Ready**! 📈")
    else:
        st.markdown("Looks like you might not be a LinkedIn pro...yet! 🚀")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100, 
        title={'text': f"LinkedIn User Probability: {probability * 100:.2f}%"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "lightgray"},
                {"range": [30, 70], "color": "lightblue"},
                {"range": [70, 100], "color": "blue"}],
            "bar": {"color": "gold"} }))
    
    st.plotly_chart(fig)
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100, 
    title={'text': f"LinkedIn User Probability: {probability * 100:.2f}%"},
    number={"suffix": "%"},
    gauge={
        "axis": {"range": [0, 100]},
        "steps": [
            {"range": [0, 30], "color": "lightgray"},
            {"range": [30, 70], "color": "lightblue"},
            {"range": [70, 100], "color": "blue"}],
        "bar": {"color": "gold"} }))
    
st.plotly_chart(fig)
<<<<<<< HEAD

st.markdown("### Thank you for completing the form! 🎉")
st.write(f"Based on your profile, here's what we found:")

if classification == 1:
    st.markdown("Great job! You're a LinkedIn user. Time to network and grow your professional empire! 🌐💼")
else:
    st.markdown("Hmm, it seems like you're not yet a LinkedIn user. Maybe now's a good time to join and explore new career opportunities? 🚀📈")

if age < 18:
    st.error("You must be at least 18 years old to use this app. Please try again later! 👶❌")
elif income > 9:
    st.error("Please select a valid income range. Don't worry, we’re not here to judge your finances! 💰")
=======

st.markdown("### Thank you for completing the form! 🎉")
st.write(f"Based on your profile, here's what we found:")

if classification == 1:
    st.markdown("Great job! You're a LinkedIn user. Time to network and grow your professional empire! 🌐💼")
else:
    st.markdown("Hmm, it seems like you're not yet a LinkedIn user. Maybe now's a good time to join and explore new career opportunities? 🚀📈")

if age < 18:
    st.error("You must be at least 18 years old to use this app. Please try again later! 👶❌")
elif income > 9:
    st.error("Please select a valid income range. Don't worry, we’re not here to judge your finances! 💰")
