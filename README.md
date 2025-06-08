# TFG_Carlota

Este repositorio contiene el desarrollo completo del Trabajo de Fin de Grado de Carlota, centrado en la creación de una herramienta basada en inteligencia artificial y emocional para asistir a profesionales médicos en la priorización de pacientes en contextos de escasez de recursos clínicos.

## Estructura del repositorio

```text
TFG_Carlota/
├── datalake/                      # Datos originales sin procesar
├── datawarehouse/                 # Datos agrupados tras limpieza inicial
├── dataframe/                     # Datos derivados y organizados para modelado
│   ├── cosas.csv
│   ├── data_emocional.csv
│   ├── data_riesgo_clinico.csv
│   ├── datos_limpios.csv
│   └── datos_limpios_sin_encoder.csv
├── outputs/                       # Resultados del preprocesamiento y modelado
│   ├── data_clin/
│   ├── data_emo/
│   ├── encoder/
│   ├── encoder_emo/
│   ├── models_clin/
│   └── models_emo/
├── Procesamiento/                # Cuadernos de exploración y preparación
│   ├── Exploracion.ipynb
│   ├── Exploracion_Avanzada.ipynb
│   ├── Extract.ipynb
│   ├── Preparacion_Inicial.ipynb
│   └── Transformacion.ipynb
├── Utilidades/
│   ├── columnas_bool.csv
│   └── diccionario_valores.ipynb
├── 01_modelo_clinic.ipynb         # Entrenamiento modelo clínico
├── 02_create_data_emo.ipynb       # Generación variables emocionales
├── 03_modelos_emo.ipynb           # Modelado emocional
├── 04_eval_model_emo.ipynb        # Evaluación de resultados emocionales
├── app.py                         # App Streamlit
└── requirements.txt               # Dependencias del proyecto
```


## Descripción del proyecto

Este proyecto explora cómo combinar datos clínicos y emocionales para estimar el riesgo de pacientes, con el fin de ayudar a los médicos a tomar decisiones informadas cuando los tratamientos disponibles son limitados. La arquitectura sigue un enfoque modular que incluye:

- Preprocesamiento de datos clínicos y emocionales.
- Entrenamiento de modelos de predicción de supervivencia.
- Visualización y evaluación del riesgo individual de los pacientes.
- Prototipo de herramienta interactiva de apoyo a la decisión clínica: https://prediccion-riesgo-clinico-emocional.onrender.com

## Requisitos

Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
