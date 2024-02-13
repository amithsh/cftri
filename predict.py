import pandas as pd
import numpy as np  # Add this line
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Step 1: Load the data
data = pd.read_csv('/home/amith/Desktop/dataset.csv')

# Step 2: Prepare the data
X = data[['Average Chl a', 'Average Chl b', 'Average Chl a+b', 'SD Chl a', 'SD Chl b', 'SD Chl a+b']]

# Assuming there is no 'condition' column, let's create one with random values for demonstration
conditions = ['Excellent', 'Good', 'Fair', 'Poor']
data['condition'] = np.random.choice(conditions, size=len(data))

y = data['condition']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose a machine learning algorithm (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Step 7: Streamlit application
st.title('Broccoli Condition Predictor')

chl_a = st.number_input('Enter Average Chl a')
chl_b = st.number_input('Enter Average Chl b')
chl_a_b = st.number_input('Enter Average Chl a+b')
sd_chl_a = st.number_input('Enter SD Chl a')
sd_chl_b = st.number_input('Enter SD Chl b')
sd_chl_a_b = st.number_input('Enter SD Chl a+b')

input_data = [[chl_a, chl_b, chl_a_b, sd_chl_a, sd_chl_b, sd_chl_a_b]]
prediction = model.predict(input_data)

st.write(f'Predicted condition: {prediction}')
