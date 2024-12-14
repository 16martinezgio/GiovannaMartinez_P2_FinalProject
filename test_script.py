#testing if the script will work if more simple

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.markdown('# Predicting LinkedIn Users')

s= pd.read_csv("social_media_usage.csv")
def clean_sm(x):
    return np.where(x== 1, 1, 0)

sm_li= clean_sm(s["web1h"])
s["sm_li"] = sm_li


ss= pd.DataFrame({
    "sm_li":np.where(s["sm_li"]==1,1,0),
    "income": np.where(s["income"]> 9, np.nan, s["income"]),
    "education":np.where(s["educ2"]>8, np.nan, s["educ2"]),
    "parent": np.where(s["par"]==1, 1, 0),
    "married":np.where(s["marital"]==1, 1, 0),
    "female":np.where(s["gender"]==2, 1, 0),
    "age":np.where(s["age"]>98, np.nan, s["age"])
})


ss= ss.dropna()

y = ss["sm_li"]
X = ss[['age', 'education','income', 'parent', 'married', 'female']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify=y,
                                                   test_size=.2,
                                                   random_state=3125)


lr= LogisticRegression(class_weight= 'balanced')

lr.fit(X_train, y_train)

# incomeoptions = ["Less than $10,000", "10 to under $20,000", "20 to under $30,000",
# "30 to under $40,000", "40 to under $50,000", "50 to under $75,000",
# "75 to under $100,000", "100 to under $150,000", "more than $150,000"]

# income_mapping = {opt: idx + 1 for idx, opt in enumerate(incomeoptions)}

# inc = st.selectbox("Select Income Level", options=incomeoptions)

# income = income_mapping.get(inc, 0)



# eductoptions = ["Less than high school", "High school incomplete", "High school graduate",
# "Some college, no degree", "Two-year associate degree from a college or university",
# "Four-year college or university degree/Bachelor’s degree",
# "Some postgraduate or professional schooling, no postgraduate degree",
# "Postgraduate or professional degree, including master’s, doctorate, medical or law degree"]

# education_mapping = {opt: idx + 1 for idx, opt in enumerate(eductoptions)}

# educ = st.selectbox("Select Education Level", options=eductoptions)

# education = education_mapping.get(educ, 0)

incomeoption = st.selectbox(
    'What is your income range?',
    ('Less than $10,000', '$10,000-20,000', '$20,000-30,000', '$30,000-40,000', '$40,000-50,000', '$50,000-75,000', '$75,000-100,000', '$100,000-150,000', '$150,000+'), index=None, placeholder="Select income range...")

st.write('***You selected:***', incomeoption)

if incomeoption == 'Less than $10,000':
    incomeoption = 1
elif incomeoption == '$10-20,000':
    incomeoption = 2
elif incomeoption == '$20-30,000':
    incomeoption = 3
elif incomeoption == '$30-40,000':
    incomeoption = 4
elif incomeoption == '$40-50,000':
    incomeoption = 5
elif incomeoption == '$50-75,000':
    incomeoption = 6
elif incomeoption == '$75-100,000':
    incomeoption = 7
elif incomeoption == '$100-150,000':
    incomeoption = 8
else:
    incomeoption = 9

eductoption = st.selectbox(
    'What is your highest education level achieved?',
    ('less than high school', 'high school incomplete', 'high school graduate', 'some college, no degree', '2 year degree (Associates)', '4 year degree (Bachelors)', 'Some post grad, no post grad degree', 'post grad complete including Masters and Doctorate'), index=None, placeholder="Select highest education...")

st.write('***You selected:***', eductoption)

if eductoption == 'less than high school':
    eductoption = 1
elif eductoption == 'high school incomplete':
    eductoption = 2
elif eductoption == 'high school graduate':
    eductoption = 3
elif eductoption == 'some college, no degree':
    eductoption = 4
elif eductoption == '2 year degree (Associates)':
    eductoption = 5
elif eductoption == '4 year degree (Bachelors)':
    eductoption = 6
elif eductoption == 'Some post grad, no post grad degree':
    eductoption = 7
else:
    eductoption = 8

parentoption = st.radio(
    "Are you a parent?",
    ["Yes", "No"],
    captions = ["I am a proud parent", "Thank gawd No"], index=None)
if parentoption =="Yes":
    st.write('***You are a parent***')
else:
    st.write("***You are not a parent***")

if parentoption== "Yes":
    parentoption= 1
else:
    parentoption= 0

marriedoption = st.radio(
    "Are you married?",
    ["Yes", "No"], index=None
    )
if marriedoption =="Yes":
    st.write('***I am married***')
else:
    st.write("***I am not married***")

if marriedoption== "Yes":
    marriedoption= 1
else:
    marriedoption= 0

genderoption = st.radio(
    "Are you a female?",
    ["Yes", "No"], index=None
    )
if genderoption =="Yes":
    st.write('***I am a female***')
else:
    st.write("***I am not a female***")

if genderoption== "Yes":
    genderoption= 1
else:
    genderoption= 0

age = st.number_input(
    'Please update your age...', min_value=18, max_value=99, value=30, step= 1, format=None, help='Age is automatically calculated, please update as applicable')
st.write('***You are*** ', age, "***years old***")


newdata= pd.DataFrame({
    "age":[age],
    "education":[eductoption],
    "income":[incomeoption],
    "parent": [parentoption],
    "married": [marriedoption],
    "female": [genderoption]    
})
newdata['prediction_linkedin_user']=lr.predict(newdata)


if st.button('Click to Predict!!!', key='prediction_linkedin_user', type='primary'):
    prediction = newdata['prediction_linkedin_user']
    if any(prediction == 1):
        st.write('***You are a LinkedIn User***')
    else:
        st.write('***You are NOT a LinkedIn User***')
    
# if newdata['prediction_linkedin_user'].any()==1:
#     st.write('You are a LinkedIn User')
# else:
#     st.write('You are NOT a Linkedin User')

# if newdata['prediction_linkedin_user'].any()==1:
#     st.write('You are a LinkedIn User')
# else:
#     st.write('You are NOT a Linkedin User')

# st.markdown ('## Click to Predict')

# if st.button('Click to Predict!!!', type='primary'):
#     st.write('This is where I will print off my prediction')
