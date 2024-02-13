import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  # Add KNN classifier
from sklearn.svm import SVC  # Add SVM classifier
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image  # Import the Image class

# Define page layout
st.set_page_config(layout="wide")

# Define logo and CFTRI heading
logo = Image.open("logo.png")
col1, col2 = st.columns([1, 3])
col1.image(logo, width=300)
col2.title('Central Food Technological Research Institution')
col2.markdown(
    """
    <style>
        .stApp {padding-top: 20px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation bar
menu = ["Home", "About", "Sections"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.title("Home Page")
    # Add content for home page here
elif choice == "About":
    st.title("About CFTRI")
    about_text = """
    The Central Food Technological Research Institute (CFTRI) is an Indian food research institute and laboratory headquartered in Mysore, India. It is a constituent laboratory of the Council of Scientific and Industrial Research.
    
    The institute has nearly two hundred scientists, technologists, and engineers, and over a hundred technicians, skilled workers, and support staff. There are sixteen research and development departments, including laboratories focusing on food engineering, food biotechnology, microbiology, grain sciences, sensory science, biochemistry, molecular nutrition, and food safety.
    
    The institute has developed over 300 products, processes, and equipment designs, and most of these technologies have been released to over 4000 licensees for commercial application. The institute develops technologies to increase efficiency and reduce post-harvest losses, add convenience, increase export, find new sources of food products, integrate human resources in food industries, reduce costs, and modernize. It holds several patents and has published findings in reputed journals.
    """
    st.markdown(about_text)
elif choice == "Sections":
    st.title("Sections")
    # Add content for sections page here

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

# Classifier selection
classifier = st.sidebar.radio("Select Classifier", ('Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine'))

# Step 4: Choose a machine learning algorithm
if classifier == 'Random Forest':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif classifier == 'K-Nearest Neighbors':
    model = KNeighborsClassifier(n_neighbors=5)
elif classifier == 'Support Vector Machine':
    model = SVC(kernel='linear')

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy: {accuracy}')

# Step 7: Streamlit application
st.title('Broccoli Condition Predictor')

chl_a = st.number_input('Enter Average Chl a')
chl_b = st.number_input('Enter Average Chl b')
chl_a_b = st.number_input('Enter Average Chl a+b')
sd_chl_a = st.number_input('Enter SD Chl a')
sd_chl_b = st.number_input('Enter SD Chl b')
sd_chl_a_b = st.number_input('Enter SD Chl a+b')

if st.button('Predict'):
    input_data = [[chl_a, chl_b, chl_a_b, sd_chl_a, sd_chl_b, sd_chl_a_b]]
    prediction = model.predict(input_data)
    st.write(f'Predicted condition: {prediction}')
    st.write(f'Classifier used: {classifier}')
