import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_scaled, y)

# Streamlit UI
st.title("🌸 Iris Flower Classification")
st.write("Predict Iris flower species using Logistic Regression")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()))
sepal_width  = st.slider("Sepal Width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()))
petal_length = st.slider("Petal Length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()))
petal_width  = st.slider("Petal Width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()))

# Predict button
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"The predicted species is: **{iris.target_names[prediction]}** 🌸")
