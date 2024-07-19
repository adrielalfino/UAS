# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load data
def load_data():
    data = pd.read_csv('dataset.csv')
    return data

st.title('Aplikasi Klasifikasi dengan Streamlit')

# Input data
uploaded_file = st.file_uploader("Unggah file CSV untuk prediksi", type=["csv"])
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.write(input_data)

    # Load model
    model = load_model()

    # Prediksi
    predictions = model.predict(input_data)
    st.write("Prediksi:")
    st.write(predictions)

else:
    st.write("Unggah file CSV untuk melanjutkan")

# Menampilkan hasil akurasi model
data = load_data()
model, accuracy = train_model(data)
st.write(f"Akurasi model: {accuracy}")
