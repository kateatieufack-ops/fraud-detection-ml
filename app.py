import streamlit as st
import joblib  # <-- changer pickle par joblib
import pandas as pd

# Charger le modèle
model = joblib.load("fraud_model.pkl")  # <-- charger le modèle avec joblib

st.title("Détection de Fraude à la Carte de Crédit")
st.write("Téléversez un fichier CSV pour détecter les fraudes :")

uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(data.head())

    predictions = model.predict(data)
    data["Fraude"] = predictions
    st.write("Résultats de la détection :")
    st.dataframe(data)