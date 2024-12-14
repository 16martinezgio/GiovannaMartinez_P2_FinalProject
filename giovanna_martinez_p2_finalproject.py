# A dynamic and fun Streamlit app to predict LinkedIn users 🎉

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add a sidebar for navigation
st.sidebar.markdown("## 🛠️ App Settings")
theme = st.sidebar.radio(
    "Choose your theme:",
    ["Default", "Dark Mode 🌙", "Light Mode ☀️"],
    index=0
)
if theme == "Dark Mode 🌙":
    st.markdown('<style>body{background-color: #1e1e1e; color: white;}</style>', unsafe_allow_html=True)
elif theme == "Light Mode ☀️":
    st.markdown('<style>body{background-color: #ffffff; color: black;}</style>', unsafe_allow_html=True)

# App title with style
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔮 Predicting LinkedIn Users</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Find out if you're part of the LinkedIn crowd or missing out on the networking fun!</p>", unsafe_allow_html=True)

# Load and clean data
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

s["sm_li"] = clean_sm(s["web1h"])

s['sm_li'] = clean_sm(s["web1h"])

ss = s[['sm_li', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()

ss['income'] = ss['income'].apply(lambda x: x if x <= 9 else np.nan)
ss['education'] = ss['educ2'].apply(lambda x: x if x <= 8 else np.nan)
ss['age'] = ss['age'].apply(lambda x: x if x <= 97 else np.nan)

ss['parent'] = ss['par'].apply(lambda x: 1 if x == 1 else 0)
ss['married'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)
ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)

ss = ss.drop(columns=['par', 'marital', 'gender', 'educ2']).dropna()

# Train the model
y = ss["sm_li"]
X = ss[['age', 'education', 'income', 'parent', 'married', 'female']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=3125
)

lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)

# Interactive inputs
st.markdown("### 🎯 Provide Your Information")

# Income selection
incomeoption = st.selectbox(
    '💵 What is your income range?',
    ('Less than $10,000', '$10,000-20,000', '$20,000-30,000', '$30,000-40,000', 
     '$40,000-50,000', '$50,000-75,000', '$75,000-100,000', 
     '$100,000-150,000', '$150,000+'),
    placeholder="Select income range..."
)
if incomeoption == 'Less than $10,000':
    incomeoption = 1
elif incomeoption == '$10,000-20,000':
    incomeoption = 2
elif incomeoption == '$20,000-30,000':
    incomeoption = 3
elif incomeoption == '$30,000-40,000':
    incomeoption = 4
elif incomeoption == '$40,000-50,000':
    incomeoption = 5
elif incomeoption == '$50,000-75,000':
    incomeoption = 6
elif incomeoption == '$75,000-100,000':
    incomeoption = 7
elif incomeoption == '$100,000-150,000':
    incomeoption = 8
else:
    incomeoption = 9

# Education selection
eductoption = st.selectbox(
    '🎓 What is your highest education level?',
    ('Less than high school', 'High school incomplete', 'High school graduate', 
     'Some college, no degree', '2-year degree (Associates)', 
     '4-year degree (Bachelors)', 'Some postgrad, no degree', 
     'Postgrad complete (Masters/Doctorate)'),
    placeholder="Select highest education level..."
)
if eductoption == 'Less than high school':
    eductoption = 1
elif eductoption == 'High school incomplete':
    eductoption = 2
elif eductoption == 'High school graduate':
    eductoption = 3
elif eductoption == 'Some college, no degree':
    eductoption = 4
elif eductoption == '2-year degree (Associates)':
    eductoption = 5
elif eductoption == '4-year degree (Bachelors)':
    eductoption = 6
elif eductoption == 'Some postgrad, no degree':
    eductoption = 7
else:
    eductoption = 8

# Parent status
parentoption = st.radio("👶 Are you a parent?", ["Yes 👪", "Nope ❌"])
if parentoption == "Yes 👪":
    parentoption = 1
else:
    parentoption = 0

# Marital status
marriedoption = st.radio("💍 Are you married?", ["Yes 💞", "Nope 💔"])
if marriedoption == "Yes 💞":
    marriedoption = 1
else:
    marriedoption = 0

# Gender
genderoption = st.radio("👩 Are you a female?", ["Yes 🙋‍♀️", "Nope 🙋‍♂️"])
if genderoption == "Yes 🙋‍♀️":
    genderoption = 1
else:
    genderoption = 0

# Age input with slider for better UX
age = st.slider(
    '🎂 Enter your age:',
    min_value=18, max_value=99, value=30, step=1
)
st.write(f'✨ ***You are {age} years young!***')

# Prepare new data for prediction
newdata = pd.DataFrame({
    "age": [age],
    "education": [eductoption],
    "income": [incomeoption],
    "parent": [parentoption],
    "married": [marriedoption],
    "female": [genderoption]
})
newdata['prediction_linkedin_user'] = lr.predict(newdata)

# Show the user input data for transparency
st.markdown("### 📊 Your Input Data:")
st.dataframe(newdata)

# Prediction button with animated balloons
if st.button('🔮 Click to Predict!'):
    prediction = newdata['prediction_linkedin_user'].iloc[0]
    if prediction == 1:
        st.balloons()
        st.success('🎉 Woohoo! You are a LinkedIn user! Time to network! 🤝')
    else:
        st.warning('🤔 Looks like you are not a LinkedIn user! Maybe it’s time to join? 🚀')

# Footer message
st.sidebar.markdown("### 🚀 Built with Streamlit for Fun and Insights!")
