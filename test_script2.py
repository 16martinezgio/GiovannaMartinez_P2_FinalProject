# A playful Streamlit app to predict LinkedIn users 🎉

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title with some flair!
st.markdown("# 🔮 Predicting LinkedIn Users")

# Load and clean data
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

s["sm_li"] = clean_sm(s["web1h"])

ss = pd.DataFrame({
    "sm_li": s["sm_li"],
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
}).dropna()

# Train the model
y = ss["sm_li"]
X = ss[['age', 'education', 'income', 'parent', 'married', 'female']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=3125
)

lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)

# Fun interaction for income
incomeoption = st.selectbox(
    '💵 What is your income range?',
    ('Less than $10,000', '$10,000-20,000', '$20,000-30,000', '$30,000-40,000', 
     '$40,000-50,000', '$50,000-75,000', '$75,000-100,000', 
     '$100,000-150,000', '$150,000+'),
    placeholder="Select income range..."
)
st.write('✨ ***You selected:***', incomeoption)

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

# Fun interaction for education
eductoption = st.selectbox(
    '🎓 What is your highest education level?',
    ('Less than high school', 'High school incomplete', 'High school graduate', 
     'Some college, no degree', '2-year degree (Associates)', 
     '4-year degree (Bachelors)', 'Some postgrad, no degree', 
     'Postgrad complete (Masters/Doctorate)'),
    placeholder="Select highest education level..."
)
st.write('🎉 ***You selected:***', eductoption)

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

# Parent, marriage, and gender inputs with fun captions
parentoption = st.radio("👶 Are you a parent?", ["Yes 👪", "Nope ❌"])
if parentoption == "Yes 👪":
    parentoption = 1
    st.write('✨ ***You are a proud parent!***')
else:
    parentoption = 0
    st.write('🍼 ***Not a parent? More me-time for you!***')

marriedoption = st.radio("💍 Are you married?", ["Yes 💞", "Nope 💔"])
if marriedoption == "Yes 💞":
    marriedoption = 1
    st.write('🎉 ***Happily married!***')
else:
    marriedoption = 0
    st.write('💔 ***Living the single life!***')

genderoption = st.radio("👩 Are you a female?", ["Yes 🙋‍♀️", "Nope 🙋‍♂️"])
if genderoption == "Yes 🙋‍♀️":
    genderoption = 1
    st.write('✨ ***You identified as female.***')
else:
    genderoption = 0
    st.write('✨ ***You identified as male.***')

# Fun age input
age = st.number_input(
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

# Prediction button with animation
if st.button('🔮 Click to Predict!'):
    prediction = newdata['prediction_linkedin_user'].iloc[0]
    if prediction == 1:
        st.success('🎉 Woohoo! You are a LinkedIn user! Time to network! 🤝')
    else:
        st.warning('🤔 Looks like you are not a LinkedIn user! Maybe it’s time to join? 🚀')
