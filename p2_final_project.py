import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Load the CSV file
s = pd.read_csv('social_media_usage.csv')

# Data cleaning function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Clean and preprocess the dataset
s['sm_li'] = clean_sm(s["web1h"])
ss = s[['sm_li', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
ss['income'] = ss['income'].apply(lambda x: x if x <= 9 else np.nan)
ss['education'] = ss['educ2'].apply(lambda x: x if x <= 8 else np.nan)
ss['age'] = ss['age'].apply(lambda x: x if x <= 97 else np.nan)
ss['parent'] = ss['par'].apply(lambda x: 1 if x == 1 else 0)
ss['married'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)
ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)
ss = ss.drop(columns=['par', 'marital', 'gender', 'educ2']).dropna()

# Split the data
y = ss["sm_li"]
X = ss[['age', 'education', 'income', 'parent', 'married', 'female']]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=3125)

# Train the model
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)

# Streamlit App
st.markdown("## 🌟 Hello! Welcome to the LinkedIn Usage Prediction App! 🌟")
st.markdown("### 🤔 Ever wondered if you're LinkedIn-ready? Let's find out together! 😄")

# Collect user inputs
income = st.selectbox(
    "💰 What's your income range?",
    options=[(1, "Less than $10,000"), (2, "$10,000 - $19,999"), (3, "$20,000 - $29,999"),
             (4, "$30,000 - $39,999"), (5, "$40,000 - $49,999"), (6, "$50,000 - $74,999"),
             (7, "$75,000 - $99,999"), (8, "$100,000 - $149,999"), (9, "$150,000 or more")],
    format_func=lambda x: x[1]
)
educ2 = st.selectbox(
    "🎓 What’s your highest level of education?",
    options=[(1, "Middle school or less"), (2, "High school incomplete"), (3, "High school graduate/GED"),
             (4, "Some college, no degree"), (5, "Associate degree"), (6, "Bachelor's degree"),
             (7, "Some postgraduate schooling"), (8, "Postgraduate or professional degree")],
    format_func=lambda x: x[1]
)
parent = st.radio("👶 Do you have any children under 18?", options=[(1, "Yes, I’m a proud parent!"), (0, "Nope, not yet!")])
married = st.radio("💍 Are you married?", options=[(1, "Yes, happily!"), (0, "No, single and loving it!")])
female = st.radio("💃 Are you a woman?", options=[(1, "Yes!"), (0, "Nope!")])
age = st.number_input("🎂 How old are you?", min_value=18, max_value=99, step=1)

# Prediction function
def predict_usage(user_data):
    user_data = pd.DataFrame([user_data], columns=['income', 'education', 'parent', 'married', 'female', 'age'])
    probability = log_reg.predict_proba(user_data)[0][1]
    classification = log_reg.predict(user_data)[0]

    st.markdown("### 🎉 Prediction Results 🎉")
    st.write(f"**🤖 Classification:** {'LinkedIn user! 🏆' if classification == 1 else 'Not a LinkedIn user. 🤷‍♂️'}")
    st.write(f"**📊 Probability:** {probability * 100:.2f}% chance of being a LinkedIn user.")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': f"📈 LinkedIn User Probability: {probability * 100:.2f}%"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "lightgray"},
                {"range": [30, 70], "color": "lightblue"},
                {"range": [70, 100], "color": "blue"}],
            "bar": {"color": "gold"}}))

    st.plotly_chart(fig)

# Process inputs and make predictions
if st.button("✨ Predict My LinkedIn Future! ✨"):
    user_data = [income[0], educ2[0], parent, married, female, age]
    predict_usage(user_data)

st.markdown("---")
st.markdown("💡 **Pro Tip:** Stay connected, keep networking, and watch your LinkedIn stats soar! 🚀")
