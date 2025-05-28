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

# === Inicializar estado ===
if "enviado" not in st.session_state:
    st.session_state.enviado = False

# === Cargar todos los encoders ===
@st.cache_resource
def cargar_encoders():
    carpetas = ["Outputs/encoder_emo", "Outputs/encoder"]
    encoders = {}
    for carpeta in carpetas:
        for nombre_archivo in os.listdir(carpeta):
            if nombre_archivo.endswith(".pkl"):
                clave = nombre_archivo.replace("encoder_emo_", "").replace("encoder_", "").replace(".pkl", "")
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

            tissue_or_organ_of_origin_diagnoses = st.selectbox("Tejido u órgano de origen (tissue_or_organ_of_origin.diagnoses)", [
                                                                                                                                    "Kidney, NOS", "Lower lobe, lung", "Upper lobe, lung", "Overlapping lesion of lung", "Middle lobe, lung", "Lung, NOS", "Main bronchus", "Skin, NOS", "Cerebrum", "Brain, NOS",
                                                                                                                                    "Temporal lobe", "Frontal lobe", "Occipital lobe", "Parietal lobe", "Adrenal gland, NOS", "Head, face or neck, NOS", "Retroperitoneum", "Connective, subcutaneous and other soft tissues of trunk, NOS", 
                                                                                                                                    "Connective, subcutaneous and other soft tissues of abdomen", "Connective, subcutaneous and other soft tissues of thorax", "Medulla of adrenal gland", "Thorax, NOS", "Mediastinum, NOS", 
                                                                                                                                    "Cortex of adrenal gland", "Aortic body and other paraganglia", "Connective, subcutaneous and other soft tissues of pelvis", "Heart", "Rectosigmoid junction", "Rectum, NOS", 
                                                                                                                                    "Colon, NOS", "Thyroid gland", "Endometrium", "Isthmus uteri", "Fundus uteri", "Corpus uteri", "Intrahepatic bile duct", "Gallbladder", "Prostate gland", "Hypopharynx, NOS", 
                                                                                                                                    "Base of tongue, NOS", "Larynx, NOS", "Tongue, NOS", "Mouth, NOS", "Cheek mucosa", "Tonsil, NOS", "Floor of mouth, NOS", "Overlapping lesion of lip, oral cavity and pharynx", "Gum, NOS", 
                                                                                                                                    "Hard palate", "Oropharynx, NOS", "Anterior floor of mouth", "Retromolar area", "Border of tongue", "Palate, NOS", "Supraglottis", "Lip, NOS", "Mandible", 
                                                                                                                                    "Posterior wall of oropharynx", "Breast, NOS", "Overlapping lesion of breast", "Lower-outer quadrant of breast", "Upper-inner quadrant of breast", "Testis, NOS", 
                                                                                                                                    "Middle third of esophagus", "Upper third of esophagus", "Lower third of esophagus", "Thoracic esophagus", "Esophagus, NOS", "Cardia, NOS", "Intra-abdominal lymph nodes", 
                                                                                                                                    "Lymph nodes of head, face and neck", "Lymph nodes of axilla or arm", "Lymph nodes of inguinal region or leg", "Anterior mediastinum", "Brain stem", 
                                                                                                                                    "Connective, subcutaneous and other soft tissues of head, face, and neck", "Cerebellum, NOS", "Submandibular gland", "Stomach, NOS", "Intrathoracic lymph nodes", 
                                                                                                                                    "Jejunum", "Small intestine, NOS", "Connective, subcutaneous and other soft tissues, NOS", "Bones of skull and face and associated joints", "Specified parts of peritoneum", 
                                                                                                                                    "Cervix uteri", "Head of pancreas", "Body of pancreas", "Tail of pancreas", "Pancreas, NOS", "Overlapping lesion of pancreas", "Pleura, NOS", "Liver", 
                                                                                                                                    "Descending colon", "Sigmoid colon", "Ascending colon", "Cecum", "Transverse colon", "Splenic flexure of colon", "Hepatic flexure of colon", "Uterus, NOS", 
                                                                                                                                    "Overlapping lesion of brain", "Bone marrow", "Connective, subcutaneous and other soft tissues of lower limb and hip", 
                                                                                                                                    "Peripheral nerves and autonomic nervous system of upper limb and shoulder", "Ovary", "Connective, subcutaneous and other soft tissues of upper limb and shoulder", 
                                                                                                                                    "Spinal meninges", "Spermatic cord", "Overlapping lesion of connective, subcutaneous and other soft tissues", "Myometrium", "Posterior wall of bladder", 
                                                                                                                                    "Bladder, NOS", "Trigone of bladder", "Lateral wall of bladder", "Dome of bladder", "Anterior wall of bladder", "Ureteric orifice", 
                                                                                                                                    "Fundus of stomach", "Gastric antrum", "Body of stomach", "Ciliary body", "Choroid", "Overlapping lesion of eye and adnexa", "Thymus"
                                                                                                                                ])

            primary_diagnosis_diagnoses = st.selectbox("Diagnóstico primario (primary_diagnosis.diagnoses)", [
                                                                                                                'Clear cell adenocarcinoma, NOS', 'Renal cell carcinoma, NOS', 'Adenocarcinoma, NOS',
                                                                                                                'Solid carcinoma, NOS', 'Acinar cell carcinoma', 'Adenocarcinoma with mixed subtypes',
                                                                                                                'Bronchiolo-alveolar carcinoma, non-mucinous', 'Papillary adenocarcinoma, NOS',
                                                                                                                'Mucinous adenocarcinoma', 'Bronchiolo-alveolar adenocarcinoma, NOS',
                                                                                                                'Micropapillary carcinoma, NOS', 'Bronchio-alveolar carcinoma, mucinous',
                                                                                                                'Malignant melanoma, NOS', 'Spindle cell melanoma, NOS', 'Amelanotic melanoma',
                                                                                                                'Nodular melanoma', 'Superficial spreading melanoma', 'Epithelioid cell melanoma',
                                                                                                                'Lentigo maligna melanoma', 'Mixed epithelioid and spindle cell melanoma',
                                                                                                                'Acral lentiginous melanoma, malignant', 'Oligodendroglioma, anaplastic',
                                                                                                                'Mixed glioma', 'Astrocytoma, NOS', 'Oligodendroglioma, NOS',
                                                                                                                'Astrocytoma, anaplastic', 'Pheochromocytoma, malignant', 'Pheochromocytoma, NOS',
                                                                                                                'Extra-adrenal paraganglioma, malignant', 'Extra-adrenal paraganglioma, NOS',
                                                                                                                'Paraganglioma, NOS', 'Paraganglioma, malignant', 'Tubular adenocarcinoma',
                                                                                                                'Adenocarcinoma in tubolovillous adenoma', 'Squamous cell carcinoma, NOS',
                                                                                                                'Papillary squamous cell carcinoma', 'Basaloid squamous cell carcinoma',
                                                                                                                'Squamous cell carcinoma, keratinizing, NOS',
                                                                                                                'Squamous cell carcinoma, large cell, nonkeratinizing, NOS',
                                                                                                                'Papillary carcinoma, follicular variant', 'Papillary carcinoma, columnar cell',
                                                                                                                'Oxyphilic adenocarcinoma', 'Carcinoma, NOS', 'Nonencapsulated sclerosing carcinoma',
                                                                                                                'Follicular carcinoma, minimally invasive', 'Follicular adenocarcinoma, NOS',
                                                                                                                'Endometrioid adenocarcinoma, NOS', 'Serous cystadenocarcinoma, NOS',
                                                                                                                'Endometrioid adenocarcinoma, secretory variant', 'Carcinoma, undifferentiated, NOS',
                                                                                                                'Papillary serous cystadenocarcinoma', 'Serous surface papillary carcinoma',
                                                                                                                'Cholangiocarcinoma', 'Renal cell carcinoma, chromophobe type',
                                                                                                                'Infiltrating duct carcinoma, NOS', 'Squamous cell carcinoma, spindle cell',
                                                                                                                'Infiltrating duct and lobular carcinoma', 'Lobular carcinoma, NOS',
                                                                                                                'Pleomorphic carcinoma', 'Intraductal micropapillary carcinoma',
                                                                                                                'Metaplastic carcinoma, NOS', 'Medullary carcinoma, NOS',
                                                                                                                'Infiltrating lobular mixed with other types of carcinoma', 'Basal cell carcinoma, NOS',
                                                                                                                'Intraductal papillary adenocarcinoma with invasion', 'Phyllodes tumor, malignant',
                                                                                                                'Infiltrating duct mixed with other types of carcinoma', 'Papillary carcinoma, NOS',
                                                                                                                'Adenoid cystic carcinoma', 'Paget disease and infiltrating duct carcinoma of breast',
                                                                                                                'Apocrine adenocarcinoma', 'Seminoma, NOS', 'Embryonal carcinoma, NOS',
                                                                                                                'Mixed germ cell tumor', 'Teratoma, benign', 'Yolk sac tumor',
                                                                                                                'Teratoma, malignant, NOS', 'Teratocarcinoma',
                                                                                                                'Malignant lymphoma, large B-cell, diffuse, NOS', 'Adenocarcinoma, endocervical type',
                                                                                                                'Adenosquamous carcinoma', 'Mucinous adenocarcinoma, endocervical type',
                                                                                                                'Neuroendocrine carcinoma, NOS', 'Epithelioid mesothelioma, malignant',
                                                                                                                'Mesothelioma, biphasic, malignant', 'Hepatocellular carcinoma, NOS',
                                                                                                                'Combined hepatocellular carcinoma and cholangiocarcinoma',
                                                                                                                'Hepatocellular carcinoma, fibrolamellar', 'Hepatocellular carcinoma, clear cell type',
                                                                                                                'Hepatocellular carcinoma, spindle cell variant', 'Adrenal cortical carcinoma',
                                                                                                                'Adenocarcinoma with neuroendocrine differentiation', 'Mullerian mixed tumor',
                                                                                                                'Carcinosarcoma, NOS', 'Glioblastoma', 'Acute myeloid leukemia, NOS',
                                                                                                                'Malignant peripheral nerve sheath tumor', 'Myxoid leiomyosarcoma',
                                                                                                                'Leiomyosarcoma, NOS', 'Liposarcoma, well differentiated',
                                                                                                                'Dedifferentiated liposarcoma', 'Fibromyxosarcoma', 'Undifferentiated sarcoma',
                                                                                                                'Malignant fibrous histiocytoma', 'Giant cell sarcoma', 'Synovial sarcoma, spindle cell',
                                                                                                                'Abdominal fibromatosis', 'Synovial sarcoma, NOS', 'Synovial sarcoma, biphasic',
                                                                                                                'Pleomorphic liposarcoma', 'Aggressive fibromatosis', 'Transitional cell carcinoma',
                                                                                                                'Papillary transitional cell carcinoma', 'Carcinoma, diffuse type',
                                                                                                                'Adenocarcinoma, intestinal type', 'Signet ring cell carcinoma',
                                                                                                                'Cystadenocarcinoma, NOS', 'Spindle cell melanoma, type B',
                                                                                                                'Thymoma, type A, malignant', 'Thymoma, type B2, NOS', 'Thymoma, type AB, malignant',
                                                                                                                'Thymoma, type B2, malignant', 'Thymoma, type AB, NOS',
                                                                                                                'Thymoma, type B1, malignant', 'Thymic carcinoma, NOS',
                                                                                                                'Thymoma, type B3, malignant', 'Thymoma, type B1, NOS', 'Thymoma, type A, NOS'
                                                                                                            ])

            prior_treatment_diagnoses = st.selectbox("Tratamiento previo (prior_treatment.diagnoses)", ['Yes', 'No'])

            site_of_resection_or_biopsy_diagnoses = st.selectbox("Lugar de resección o biopsia (site_of_resection_or_biopsy.diagnoses)", [
                                                                                                                                        "Kidney, NOS", "Lower lobe, lung", "Upper lobe, lung",
                                                                                                                                        "Overlapping lesion of lung", "Middle lobe, lung", "Lung, NOS",
                                                                                                                                        "Main bronchus", "Lymph nodes of head, face and neck", "Skin, NOS",
                                                                                                                                        "Skin of upper limb and shoulder", "Lymph nodes of axilla or arm", "Lymph node, NOS",
                                                                                                                                        "Skin of trunk", "Lymph nodes of inguinal region or leg", "Skin of lower limb and hip",
                                                                                                                                        "Connective, subcutaneous and other soft tissues of lower limb and hip", "Connective, subcutaneous and other soft tissues of trunk, NOS", "Connective, subcutaneous and other soft tissues of thorax",
                                                                                                                                        "Spleen", "Connective, subcutaneous and other soft tissues of head, face, and neck", "Connective, subcutaneous and other soft tissues, NOS", "Connective, subcutaneous and other soft tissues of upper limb and shoulder",
                                                                                                                                        "Skin of scalp and neck", "Parietal lobe", "Pelvic lymph nodes",
                                                                                                                                        "Nasal cavity", "Small intestine, NOS", "Colon, NOS",
                                                                                                                                        "Adrenal gland, NOS", "Connective, subcutaneous and other soft tissues of pelvis", "Frontal lobe",
                                                                                                                                        "Intra-abdominal lymph nodes", "Brain, NOS", "Vagina, NOS", "Parotid gland", "Liver",
                                                                                                                                        "Spinal cord", "Cerebrum", "Temporal lobe", "Occipital lobe",
                                                                                                                                        "Head, face or neck, NOS", "Retroperitoneum", "Connective, subcutaneous and other soft tissues of abdomen", "Thorax, NOS",
                                                                                                                                        "Mediastinum, NOS", "Aortic body and other paraganglia", "Heart",
                                                                                                                                        "Rectosigmoid junction", "Rectum, NOS", "Thyroid gland",
                                                                                                                                        "Endometrium", "Isthmus uteri", "Fundus uteri", "Corpus uteri",
                                                                                                                                        "Intrahepatic bile duct", "Gallbladder", "Prostate gland", "Hypopharynx, NOS",
                                                                                                                                        "Base of tongue, NOS", "Larynx, NOS", "Tongue, NOS", "Mouth, NOS",
                                                                                                                                        "Cheek mucosa", "Tonsil, NOS", "Floor of mouth, NOS", "Overlapping lesion of lip, oral cavity and pharynx",
                                                                                                                                        "Gum, NOS", "Hard palate", "Oropharynx, NOS", "Anterior floor of mouth", "Retromolar area",
                                                                                                                                        "Border of tongue", "Palate, NOS", "Supraglottis", "Lip, NOS",
                                                                                                                                        "Mandible", "Posterior wall of oropharynx", "Breast, NOS",
                                                                                                                                        "Overlapping lesion of breast", "Lower-outer quadrant of breast", "Upper-inner quadrant of breast",
                                                                                                                                        "Testis, NOS", "Middle third of esophagus", "Upper third of esophagus",
                                                                                                                                        "Lower third of esophagus", "Thoracic esophagus", "Esophagus, NOS", "Cardia, NOS",
                                                                                                                                        "Anterior mediastinum", "Brain stem", "Cerebellum, NOS", "Submandibular gland",
                                                                                                                                        "Stomach, NOS", "Intrathoracic lymph nodes", "Jejunum", "Bones of skull and face and associated joints",
                                                                                                                                        "Specified parts of peritoneum", "Cervix uteri", "Endocervix",
                                                                                                                                        "Exocervix", "Head of pancreas", "Body of pancreas", "Tail of pancreas",
                                                                                                                                        "Pancreas, NOS", "Overlapping lesion of pancreas", "Pleura, NOS",
                                                                                                                                        "Cortex of adrenal gland", "Descending colon", "Sigmoid colon",
                                                                                                                                        "Hepatic flexure of colon", "Ascending colon", "Cecum", "Transverse colon",
                                                                                                                                        "Splenic flexure of colon", "Uterus, NOS", "Overlapping lesion of brain",
                                                                                                                                        "Bone marrow", "Peripheral nerves and autonomic nervous system of upper limb and shoulder",
                                                                                                                                        "Ovary", "Spinal meninges", "Spermatic cord", "Overlapping lesion of connective, subcutaneous and other soft tissues",
                                                                                                                                        "Myometrium", "Posterior wall of bladder", "Bladder, NOS",
                                                                                                                                        "Trigone of bladder", "Lateral wall of bladder", "Dome of bladder", "Anterior wall of bladder", "Ureteric orifice",
                                                                                                                                        "Fundus of stomach", "Gastric antrum", "Body of stomach", "Ciliary body", "Choroid", "Overlapping lesion of eye and adnexa","Thymus"
                                                                                                                                        ])

            treatment_type_treatments_diagnoses = st.selectbox("Tipo de tratamiento (treatment_type.treatments.diagnoses)", ["['Pharmaceutical Therapy, NOS', 'Radiation Therapy, NOS']", "['Radiation Therapy, NOS', 'Pharmaceutical Therapy, NOS']"])

            treatment_or_therapy_treatments_diagnoses = st.selectbox("¿Recibió tratamiento o terapia? (treatment_or_therapy.treatments.diagnoses)", ["['no', 'no']", "['not reported', 'not reported']", "['no', 'yes']", "['yes', 'yes']", "['yes', 'no']", "['not reported', 'no']", "['no', 'not reported']", "['yes', 'not reported']", "['not reported', 'yes']"])

            tumor_descriptor_samples = st.selectbox("Descripción del tumor (tumor_descriptor.samples)", ['Primary', 'Not Applicable', 'New Primary', 'Recurrence', 'Metastatic'])

            sample_type_samples = st.selectbox("Tipo de muestra (sample_type.samples)", ['Primary Tumor', 'Solid Tissue Normal', 'Additional - New Primary', 'Recurrent Tumor', 'Metastatic', 'Additional Metastatic', 'Primary Blood Derived Cancer - Peripheral Blood'])

            tissue_type_samples = st.selectbox("Tipo de tejido (tissue_type.samples)", ['Tumor', 'Normal'])

            tipo_cancer_TCGA = st.selectbox("Tipo de cáncer TCGA (tipo_cancer_TCGA)", [
                "KIRC", "LUAD", "SKCM", "LGG", "PCPG", "READ", "LUSC", "THCA", "UCEC", "CHOL",
                "KICH", "KIRP", "PRAD", "HNSC", "BRCA", "TGCT", "ESCA", "DLBC", "CESC", "PAAD",
                "MESO", "LIHC", "ACC", "COAD", "UCS", "GBM", "LAML", "SARC", "BLCA", "STAD",
                "OV", "UVM", "THYM"
            ])

            tipo_cancer_general = st.selectbox("Tipo de cáncer general (tipo_cancer_general)", [
                "Riñón", "Pulmón", "Melanoma cutáneo", "Cerebro", "Feocromocitoma y paraganglioma", "Recto", "Tiroides",
                "Endometrio", "Vías biliares", "Próstata", "Cabeza y cuello", "Mama", "Testículo", "Esófago",
                "Linfoma B difuso", "Cuello uterino", "Páncreas", "Mesotelioma", "Hígado", "Corteza adrenal",
                "Colon", "Sarcoma uterino", "Leucemia mieloide aguda", "Sarcoma", "Vejiga", "Estómago", "Ovario",
                "Melanoma ocular", "Timo"
            ])


            morphology_diagnoses = st.selectbox("Morfología (morphology.diagnoses)", [
                                                                                        '8310/3', '8312/3', '8140/3', '8230/3', '8550/3', '8255/3', '8252/3', '8260/3',
                                                                                        '8480/3', '8250/3', '8265/3', '8253/3', '8720/3', '8772/3', '8730/3', '8721/3',
                                                                                        '8743/3', '8771/3', '8742/3', '8770/3', '8744/3', '9451/3', '9382/3', '9400/3',
                                                                                        '9450/3', '9401/3', '8700/3', '8700/0', '8693/3', '8693/1', '8680/1', '8680/3',
                                                                                        '8211/3', '8263/3', '8070/3', '8052/3', '8083/3', '8071/3', '8072/3', '8340/3',
                                                                                        '8344/3', '8290/3', '8010/3', '8350/3', '8335/3', '8330/3', '8380/3', '8441/3',
                                                                                        '8382/3', '8020/3', '8460/3', '8461/3', '8160/3', '8317/3', '8500/3', '8074/3',
                                                                                        '8522/3', '8520/3', '8022/3', '8507/3', '8575/3', '8510/3', '8524/3', '8090/3',
                                                                                        '8503/3', '9020/3', '8523/3', '8050/3', '8200/3', '8541/3', '8401/3', '9061/3',
                                                                                        '9070/3', '9085/3', '9080/0', '9071/3', '9080/3', '9081/3', '9680/3', '8384/3',
                                                                                        '8560/3', '8482/3', '8246/3', '9052/3', '9053/3', '8170/3', '8180/3', '8171/3',
                                                                                        '8174/3', '8173/3', '8370/3', '8370/1', '8574/3', '8950/3', '8980/3', '9440/3',
                                                                                        '9861/3', '9540/3', '8896/3', '8890/3', '8851/3', '8858/3', '8811/3', '8805/3',
                                                                                        '8830/3', '8802/3', '9041/3', '8822/1', '9040/3', '9043/3', '8854/3', '8821/1',
                                                                                        '8120/3', '8130/3', '8145/3', '8144/3', '8490/3', '8440/3', '8774/3', '8581/3',
                                                                                        '8584/1', '8582/3', '8584/3', '8582/1', '8583/3', '8586/3', '8585/3', '8583/1',
                                                                                        '8581/1'
                                                                                    ])

            year_of_diagnosis_diagnoses = st.number_input("Año del diagnóstico (year_of_diagnosis.diagnoses)", step=1)
            age_at_diagnosis_diagnoses = st.number_input("Edad al diagnóstico (age_at_diagnosis.diagnoses)", step=1)
            year_of_birth_demographic = st.number_input("Año de nacimiento (year_of_birth.demographic)", step=1)
            days_to_birth_demographic = st.number_input("Días hasta el nacimiento (days_to_birth.demographic)", step=1)
            age_at_index_demographic = st.number_input("Edad en el índice (age_at_index.demographic)", step=1)
            days_to_diagnosis_diagnoses = st.number_input("Días desde el diagnóstico (days_to_diagnosis.diagnoses)", step=1)
        
        tiempo_riesgo = st.number_input("Tiempo (en días) para estimar el riesgo acumulado", min_value=1, value=365)


    enviado = st.form_submit_button("Predecir riesgo")
    if enviado:
        st.session_state.enviado = True

# === Botón para resetear formulario ===
if st.button("🔄 Resetear todo el formulario"):
    for key in list(st.session_state.keys()):
        if key.startswith("apoyo_") or key in [
            "enviado", "responsabilidad", "neuroticismo", "extraversion", "amabilidad", "apertura",
            "ansiedad", "depresion", "estres", "alcohol", "fuma_actual", "fue_fumador",
            "alcohol_problema", "drogas_pasado", "drogas_ahora", "drogas_problema",
            "ejercicio", "alimentacion", "disease_type", "primary_site", "gender_demographic",
            "tissue_or_organ_of_origin_diagnoses", "primary_diagnosis_diagnoses", "prior_treatment_diagnoses",
            "site_of_resection_or_biopsy_diagnoses", "treatment_type_treatments_diagnoses",
            "treatment_or_therapy_treatments_diagnoses", "tumor_descriptor_samples",
            "sample_type_samples", "tissue_type_samples", "tipo_cancer_TCGA", "tipo_cancer_general",
            "morphology_diagnoses", "year_of_diagnosis_diagnoses", "age_at_diagnosis_diagnoses",
            "year_of_birth_demographic", "days_to_birth_demographic", "age_at_index_demographic",
            "days_to_diagnosis_diagnoses", "tiempo_riesgo"
        ]:
            del st.session_state[key]
    st.rerun()


# === Procesamiento y predicción ===
if st.session_state.enviado:
    # Calcular la puntuación total de apoyo
    suma_apoyo = sum(respuestas_apoyo)
    apoyo_valor = "Sí" if suma_apoyo >= 32 else "No"
    apoyo_cod = encoders["apoyo"].transform([apoyo_valor])[0]

    # Codificar variables y crear el diccionario
    datos = {
        # Categóricas codificadas
        "disease_type": encoders["disease_type"].transform([disease_type])[0],
        "primary_site": encoders["primary_site"].transform([primary_site])[0],
        "gender.demographic": encoders["gender.demographic"].transform([gender_demographic])[0],
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

        # Numéricas (sin codificar)
        "year_of_diagnosis.diagnoses": year_of_diagnosis_diagnoses,
        "age_at_diagnosis.diagnoses": age_at_diagnosis_diagnoses,
        "year_of_birth.demographic": year_of_birth_demographic,
        "days_to_birth.demographic": days_to_birth_demographic,
        "age_at_index.demographic": age_at_index_demographic,
        "days_to_diagnosis.diagnoses": days_to_diagnosis_diagnoses,

        # Rasgos personales
        "responsabilidad": responsabilidad,
        "neuroticismo": neuroticismo,
        "extraversión": extraversion,
        "amabilidad": amabilidad,
        "apertura": apertura,
        "ansiedad": ansiedad,
        "depresion": depresion,
        "estres": estres,

    }


    df = pd.DataFrame([datos])
    columnas_entrenamiento = joblib.load("Outputs/data_emo/columnas_modelo.pkl")
    df = df[columnas_entrenamiento]
    df_escalado = scaler.transform(df)
    df_escalado_named = pd.DataFrame(df_escalado, columns=columnas_entrenamiento)

    # Obtener función de riesgo acumulado
    funciones_riesgo = modelo.predict(df_escalado_named)

    # Extraer valor más próximo al tiempo ingresado
    funcion_individual = funciones_riesgo[0]

    # Mostrar resultado
    st.success("✅ Predicción completada")
    st.metric(label=f"Riesgo acumulado a días", value=f"{funcion_individual:.2%}")
