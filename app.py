import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predicción de Riesgo Emocional", layout="wide")

st.title("🧠 Predicción de Riesgo Emocional en Pacientes Oncológicos")

# === Cargar modelo y scaler ===
@st.cache_resource
def cargar_modelo_y_scaler():
    modelo = joblib.load("Outputs/models_emo/cox_ph.pkl")
    scaler = joblib.load("Outputs/data_emo/scaler.pkl")
    return modelo, scaler

modelo, scaler = cargar_modelo_y_scaler()

# === Cargar todos los encoders ===
@st.cache_resource
def cargar_encoders():
    carpeta = "Outputs/encoder_emo"
    encoders = {}
    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.endswith(".pkl"):
            clave = nombre_archivo.replace("encoder_emo_", "").replace(".pkl", "")
            encoders[clave] = joblib.load(os.path.join(carpeta, nombre_archivo))
    return encoders

encoders = cargar_encoders()

# === Formulario ===
with st.form("formulario_emocional"):
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🧩 Cuestionario de apoyo social"):
            preguntas_apoyo = [
                "Recibe visitas de mis amigos y familiares.",
                "Recibe ayuda en asuntos relacionados con mi casa",
                "Recibe elogios y reconocimientos cuando hago bien mi trabajo",
                "Cuenta con personas que se preocupan de lo que me sucede",
                "Recibe amor y afecto",
                "Tiene la posibilidad de hablar con alguien de mis problemas personales",
                "Tiene la posibilidad de hablar con alguien de mis problemas en el trabajo o en la casa.",
                "Tiene la posibilidad de hablar con alguien de mis problemas económicos.",
                "Recibe invitaciones para distraerse y salir con otras personas.",
                "Recibe consejos útiles cuando le ocurre algún acontecimiento importante.",
                "Recibe ayuda cuando está enfermo.",
            ]
        respuestas_apoyo = []
    
        for i, pregunta in enumerate(preguntas_apoyo):
            valor = st.slider(f"{i+1}. {pregunta}", 1, 5, 3, key=f"apoyo_{i}")
            respuestas_apoyo.append(valor)

        responsabilidad = st.slider("Responsabilidad", 1, 5, 3)
        neuroticismo = st.slider("Neuroticismo", 1, 6, 3)
        extraversion = st.slider("Extraversión", 1, 6, 3)
        amabilidad = st.slider("Amabilidad", 1, 6, 3)
        apertura = st.slider("Apertura", 1, 6, 3)

        ansiedad = st.selectbox("Ansiedad", [0, 1, 2, 3])
        depresion = st.selectbox("Depresión", [0, 1, 2, 3])
        estres = st.selectbox("Estrés", [0, 1, 2, 3])

        alcohol = st.selectbox("Consumo de alcohol", ["Nunca", "Ocasionalmente", "Frecuentemente"])
        fuma_actual = st.selectbox("Fuma actualmente", ["No", "Sí, <5 cigarros/día", ">5 cigarros/día"])

    with col2:
        fue_fumador = st.selectbox("Fue fumador", ["No", "Sí, en el pasado", "Sí, sigue siéndolo"])
        alcohol_problema = st.selectbox("Problemas con alcohol", ["No", "Sí, en el pasado", "Sí, sigue teniendo actualmente"])
        drogas_pasado = st.selectbox("Consumo de drogas en el pasado", ["No", "Ocasionalmente", "Frecuentemente"])
        drogas_ahora = st.selectbox("Consumo actual de drogas", ["No", "Ocasionalmente", "Habitualmente"])
        drogas_problema = st.selectbox("Problemas con drogas", ["No", "Sí"])

        ejercicio = st.selectbox("Hace ejercicio", ["No", "Sí"])
        alimentacion = st.selectbox("Alimentación saludable", ["No", "Sí"])

    enviado = st.form_submit_button("Predecir riesgo")

# === Procesamiento y predicción ===
if enviado:
        # Calcular la puntuación total de apoyo
    suma_apoyo = sum(respuestas_apoyo)
    apoyo_valor = "Sí" if suma_apoyo >= 32 else "No"
    apoyo_cod = encoders["apoyo"].transform([apoyo_valor])[0]

    # Codificar variables y crear el diccionario
    datos = {
        "responsabilidad": responsabilidad,
        "neuroticismo": neuroticismo,
        "extraversión": extraversion,
        "amabilidad": amabilidad,
        "apertura": apertura,
        "ansiedad": ansiedad,
        "depresion": depresion,
        "estres": estres,
        "apoyo": apoyo_cod,
        "alcohol": encoders["alcohol"].transform([alcohol])[0],
        "fuma_actual": encoders["fuma_actual"].transform([fuma_actual])[0],
        "fue_fumador": encoders["fue_fumador"].transform([fue_fumador])[0],
        "alcohol_problema": encoders["alcohol_problema"].transform([alcohol_problema])[0],
        "drogas_pasado": encoders["drogas_pasado"].transform([drogas_pasado])[0],
        "drogas_ahora": encoders["drogas_ahora"].transform([drogas_ahora])[0],
        "drogas_problema": encoders["drogas_problema"].transform([drogas_problema])[0],
        "ejercicio": encoders["ejercicio"].transform([ejercicio])[0],
        "alimentacion": encoders["alimentacion"].transform([alimentacion])[0],
    }

    df = pd.DataFrame([datos])

    # Escalar
    df_escalado = scaler.transform(df)

    # Predicción
    pred = modelo.predict(df_escalado)[0]
    proba = modelo.predict_proba(df_escalado)[0][1]  # Probabilidad clase positiva

    # Mostrar resultados
    st.success(f"✅ Riesgo predicho: {'ALTO' if pred == 1 else 'BAJO'}")
    st.metric(label="Probabilidad de riesgo alto", value=f"{proba:.2%}")
