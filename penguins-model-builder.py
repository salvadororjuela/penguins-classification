"""
This application builds the classification model once and the resulting .pickle file
is read in the classification application, so that when the parameters are changed
the classification model is used from the file and not created every time.
"""

import pandas as pd

penguins = pd.read_csv("penguins_cleaned.csv")
print(type(penguins))

# Create a copy of the dataframe into the df variable
df = penguins.copy()
# target will be the parameter to classify the species
target = "species"
# This two columns will be the input paramters to determine the target
encode = ["sex", "island"]

for col in encode:
    # Get the dummies for the encode variables columns (sex, island)
    dummy = pd.get_dummies(df[col], prefix = col)
    # Concatenates the new columns to the df dataset
    df = pd.concat([df, dummy], axis = 1)
    # Deletes the column with the categorical values
    del df[col]

# Dictionary of the species
target_mapper = {"Adelie": 0, "Chinstrap":1, "Gentoo": 2}

# Function to encode the target species
def target_encode(val):
    return target_mapper[val]

# Apply the target_encode function in order to perform the encoding
df["species"] = df["species"].apply(target_encode)

# Separating X and Y, where X are the independant and Y is the dependant variable
X = df.drop("species", axis = 1)
Y = df["species"]

# Build a random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
# Train the model
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open("penguins_clf.pkl", "wb")) # "wb" = write binary