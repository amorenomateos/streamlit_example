import streamlit as st
import pickle
import numpy as np

# Cargar el modelo previamente entrenado
with open("modelo.pkl", "rb") as file:
    modelo = pickle.load(file)

# Título de la aplicación
st.title("Despliegue de Modelo de Machine Learning con Streamlit")

# Descripción
st.write("Este modelo predice la clase de una flor del dataset Iris basado en cuatro características.")

# Entradas del usuario
sepal_length = st.number_input("Longitud del sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Ancho del sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Longitud del pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Ancho del pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0)

# Crear un array con los valores de entrada
inputs = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Botón para predecir
if st.button("Predecir"):
    prediccion = modelo.predict(inputs)
    clase = ["Setosa", "Versicolor", "Virginica"]
    st.write(f"La clase predicha es: {clase[prediccion[0]]}")
