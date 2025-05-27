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

        with st.expander("🧬 Datos clínico-patológicos"):
            disease_type = st.selectbox("Tipo de enfermedad (disease_type)", ['Acinar Cell Neoplasms', 'Adenomas and Adenocarcinomas', 'Adnexal and Skin Appendage Neoplasms', 'Basal Cell Neoplasms', 'Complex Epithelial Neoplasms', 'Complex Mixed and Stromal Neoplasms', 'Cystic, Mucinous and Serous Neoplasms', 'Ductal and Lobular Neoplasms', 'Epithelial Neoplasms, NOS', 'Fibroepithelial Neoplasms', 'Fibromatous Neoplasms', 'Germ Cell Neoplasms', 'Gliomas', 'Lipomatous Neoplasms', 'Mature B-Cell Lymphomas', 'Mesothelial Neoplasms', 'Myeloid Leukemias', 'Myomatous Neoplasms', 'Nerve Sheath Tumors', 'Nevi and Melanomas', 'Paragangliomas and Glomus Tumors', 'Soft Tissue Tumors and Sarcomas, NOS', 'Squamous Cell Neoplasms', 'Synovial-like Neoplasms', 'Thymic Epithelial Neoplasms', 'Transitional Cell Papillomas and Carcinomas'])

            primary_site = st.selectbox("Localización primaria (primary_site)", ['Kidney', 'Bronchus and lung', 'Skin', 'Brain', 'Adrenal gland', 'Other and ill-defined sites', 'Retroperitoneum and peritoneum', 'Connective, subcutaneous and other soft tissues', 'Heart, mediastinum, and pleura', 'Other endocrine glands and related structures', 'Rectosigmoid junction', 'Rectum', 'Colon', 'Thyroid gland', 'Corpus uteri', 'Liver and intrahepatic bile ducts', 'Gallbladder', 'Prostate gland', 'Hypopharynx', 'Base of tongue', 'Larynx', 'Other and unspecified parts of tongue', 'Other and unspecified parts of mouth', 'Tonsil', 'Floor of mouth', 'Other and ill-defined sites in lip, oral cavity and pharynx', 'Gum', 'Palate', 'Oropharynx', 'Lip', 'Bones, joints and articular cartilage of other and unspecified sites', 'Breast', 'Testis', 'Esophagus', 'Stomach', 'Lymph nodes', 'Other and unspecified major salivary glands', 'Small intestine', 'Cervix uteri', 'Pancreas', 'Uterus, NOS', 'Hematopoietic and reticuloendothelial systems', 'Peripheral nerves and autonomic nervous system', 'Ovary', 'Meninges', 'Other and unspecified male genital organs', 'Bladder', 'Eye and adnexa', 'Thymus'])

            gender_demographic = st.selectbox("Género (gender.demographic)", ['female', 'male'])

            tissue_or_organ_of_origin_diagnoses = st.selectbox("Tejido u órgano de origen (tissue_or_organ_of_origin.diagnoses)", ['Pancreas, NOS', 'Head of pancreas', 'Body of pancreas', 'Pancreatic duct', 'Tail of pancreas', 'Overlapping lesion of pancreas'])

            primary_diagnosis_diagnoses = st.selectbox("Diagnóstico primario (primary_diagnosis.diagnoses)", ['Pancreatobiliary-type adenocarcinoma', 'Intraductal papillary mucinous neoplasm with an associated invasive carcinoma', 'Intraductal papillary-mucinous carcinoma, invasive', 'Pancreatic intraepithelial neoplasia, high grade', 'Pancreatic adenocarcinoma, NOS', 'Ductal adenocarcinoma, NOS', 'Adenocarcinoma, NOS', 'Poorly differentiated carcinoma, NOS'])

            prior_treatment_diagnoses = st.selectbox("Tratamiento previo (prior_treatment.diagnoses)", ['Yes', 'No', 'Not Reported'])

            site_of_resection_or_biopsy_diagnoses = st.selectbox("Lugar de resección o biopsia (site_of_resection_or_biopsy.diagnoses)", ['Pancreas, NOS', 'Head of pancreas', 'Body of pancreas', 'Pancreatic duct', 'Tail of pancreas', 'Overlapping lesion of pancreas'])

            treatment_type_treatments_diagnoses = st.selectbox("Tipo de tratamiento (treatment_type.treatments.diagnoses)", ["['Pharmaceutical Therapy, NOS', 'Radiation Therapy, NOS']", "['Radiation Therapy, NOS', 'Pharmaceutical Therapy, NOS']"])

            treatment_or_therapy_treatments_diagnoses = st.selectbox("¿Recibió tratamiento o terapia? (treatment_or_therapy.treatments.diagnoses)", ["['no', 'no']", "['not reported', 'not reported']", "['no', 'yes']", "['yes', 'yes']", "['yes', 'no']", "['not reported', 'no']", "['no', 'not reported']", "['yes', 'not reported']", "['not reported', 'yes']"])

            tumor_descriptor_samples = st.selectbox("Descripción del tumor (tumor_descriptor.samples)", ['Primary', 'Not Applicable', 'New Primary', 'Recurrence', 'Metastatic'])

            sample_type_samples = st.selectbox("Tipo de muestra (sample_type.samples)", ['Primary Tumor', 'Solid Tissue Normal', 'Additional - New Primary', 'Recurrent Tumor', 'Metastatic', 'Additional Metastatic', 'Primary Blood Derived Cancer - Peripheral Blood'])

            tissue_type_samples = st.selectbox("Tipo de tejido (tissue_type.samples)", ['Tumor', 'Normal'])

            tipo_cancer_TCGA = st.selectbox("Tipo de cáncer TCGA (tipo_cancer_TCGA)", ['PAAD'])

            tipo_cancer_general = st.selectbox("Tipo de cáncer general (tipo_cancer_general)", ['Páncreas'])

            duration = st.number_input("Duración (duration)", step=1)
            year_of_diagnosis_diagnoses = st.number_input("Año del diagnóstico (year_of_diagnosis.diagnoses)", step=1)
            age_at_diagnosis_diagnoses = st.number_input("Edad al diagnóstico (age_at_diagnosis.diagnoses)", step=1)
            year_of_birth_demographic = st.number_input("Año de nacimiento (year_of_birth.demographic)", step=1)
            days_to_birth_demographic = st.number_input("Días hasta el nacimiento (days_to_birth.demographic)", step=1)
            age_at_index_demographic = st.number_input("Edad en el índice (age_at_index.demographic)", step=1)


    enviado = st.form_submit_button("Predecir riesgo")

# === Procesamiento y predicción ===
if enviado:
        # Calcular la puntuación total de apoyo
    suma_apoyo = sum(respuestas_apoyo)
    apoyo_valor = "Sí" if suma_apoyo >= 32 else "No"
    apoyo_cod = encoders["apoyo"].transform([apoyo_valor])[0]

    # Codificar variables y crear el diccionario
    datos = {
    "disease_type": encoders["disease_type"].transform([disease_type])[0],
    "primary_site": encoders["primary_site"].transform([primary_site])[0],
    "gender.demographic": encoders["gender.demographic"].transform([gender_demographic])[0],
    "days_to_diagnosis.diagnoses": encoders["days_to_diagnosis.diagnoses"].transform([days_to_diagnosis_diagnoses])[0],
    "tissue_or_organ_of_origin.diagnoses": encoders["tissue_or_organ_of_origin.diagnoses"].transform([tissue_or_organ_of_origin_diagnoses])[0],
    "primary_diagnosis.diagnoses": encoders["primary_diagnosis.diagnoses"].transform([primary_diagnosis_diagnoses])[0],
    "prior_treatment.diagnoses": encoders["prior_treatment.diagnoses"].transform([prior_treatment_diagnoses])[0],
    "morphology.diagnoses": encoders["morphology.diagnoses"].transform([morphology_diagnoses])[0],
    "site_of_resection_or_biopsy.diagnoses": encoders["site_of_resection_or_biopsy.diagnoses"].transform([site_of_resection_or_biopsy_diagnoses])[0],
    "treatment_type.treatments.diagnoses": encoders["treatment_type.treatments.diagnoses"].transform([treatment_type_treatments_diagnoses])[0],
    "treatment_or_therapy.treatments.diagnoses": encoders["treatment_or_therapy.treatments.diagnoses"].transform([treatment_or_therapy_treatments_diagnoses])[0],
    "tumor_descriptor.samples": encoders["tumor_descriptor.samples"].transform([tumor_descriptor_samples])[0],
    "sample_type.samples": encoders["sample_type.samples"].transform([sample_type_samples])[0],
    "tissue_type.samples": encoders["tissue_type.samples"].transform([tissue_type_samples])[0],
    "tipo_cancer_TCGA": encoders["tipo_cancer_TCGA"].transform([tipo_cancer_TCGA])[0],
    "tipo_cancer_general": encoders["tipo_cancer_general"].transform([tipo_cancer_general])[0],
    "censored": encoders["censored"].transform([censored])[0],
    "duration": duration,
    "year_of_diagnosis.diagnoses": year_of_diagnosis_diagnoses,
    "age_at_diagnosis.diagnoses": age_at_diagnosis_diagnoses,
    "year_of_birth.demographic": year_of_birth_demographic,
    "days_to_birth.demographic": days_to_birth_demographic,
    "age_at_index.demographic": age_at_index_demographic
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
