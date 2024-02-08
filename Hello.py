import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
         
This app predicts the **Palmer Penguin** species!
         
Data obtained from [palmerpenguins library] (https://github.com//allisonhorst.palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type = ["csv"])

# If there is an uploaded file with the features read it
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
# If there is no uploaded file with the features do as follows
else:
    def user_input_features():
        # Variables with the inputs
        island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen")) # Dropdown
        sex = st.sidebar.selectbox("Sex", ("male", "female")) # Dropdown
        bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.1, 59.6, 43.9) # Slider with ranges and default value
        bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5, 17.2) # Slider with ranges and default value
        flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 201.0) # Slider with ranges and default value
        body_mass_g = st.sidebar.slider("Body Mass (g)", 2700.0,6300.0,4207.0) # Slider with ranges and default value
        # Dictionary with the input data of the previous variables
        data = {"island": island,
                "sex": sex,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex
                }
        # Create and return the data frame
        features = pd.DataFrame(data, index = [0])
        return features
    
    input_df = user_input_features()

# Combines user input features with the entire penguins dataset
# This will be useful for the incoding phase
penguins_raw = pd.read_csv("penguins_cleaned.csv")
penguins = penguins_raw.drop(columns = ["species"])
df = pd.concat([input_df, penguins], axis = 0)

# Get dummies for the categorical columns
encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    # Concatenate the dummy column to the dataset
    df = pd.concat([df, dummy], axis = 1)
    # Drop the column with the categorical values
    del df[col]

# Select only the first row (the user input data)
df = df[:1]

# Display the user input features
# If a file with input parameters is uploaded
if uploaded_file is None:
    st.write(df)
# If no file with input parameters is uploaded, display the default values
else:
    st.write("Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)")
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open("penguins_clf.pkl", "rb")) # rb = read binary

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader("Prediction")
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader("Prediciton Probability")
st.write(prediction_proba)