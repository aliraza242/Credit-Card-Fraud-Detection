import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Read Data
credit_card_df = pd.read_csv("creditcard_data.csv")

# Separate the Data
valid = credit_card_df[credit_card_df['Class'] == 0]
fraud = credit_card_df[credit_card_df['Class'] == 1]

valid_sample = valid.sample(n=492, random_state=2)
credit_card_df = pd.concat([valid_sample, fraud], axis=0)

# Split the Data into training & test sets
X = credit_card_df.drop('Class', axis=1)
y = credit_card_df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train LogisticRegression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy of training Data
y_pred2 = model.predict(X_train)
accuracy_score(y_train, y_pred2)

# Accuracy score of test data
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)

# Web App
st.title("Credit Card Fraud Detection")
input_df = st.text_input("Enter All Required Features Values (comma-separated)")
input_df_splitted = input_df.split(',')

submit = st.button("Submit")

if submit:
    if len(input_df_splitted) != X.shape[1]:
        st.error(f"Invalid input! Please enter exactly {X.shape[1]} values separated by commas.")
    else:
        try:
            features = np.asarray(input_df_splitted, dtype=np.float64)
            prediction = model.predict(features.reshape(1,-1))

            if prediction[0] == 0:
                st.write("Valid Transaction")
            else:
                st.write("Fraud Transaction")
        except ValueError:
            st.error("Invalid input format! Please enter only numerical values.")