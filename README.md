# TFG_Carlota

Este repositorio contiene el desarrollo completo del Trabajo de Fin de Grado de Carlota, centrado en la creación de una herramienta basada en inteligencia artificial y emocional para asistir a profesionales médicos en la priorización de pacientes en contextos de escasez de recursos clínicos.

## Estructura del repositorio

- `datalake/`: Contiene los datos originales sin procesar.
- `datawarehouse/`: Conjunto de datos agrupados y limpiados de forma preliminar a partir del datalake.
- `dataframe/`: Datos preparados para análisis y modelado.
  - `cosas.csv`: Data con la predicción del riesgo de cada paciente generado por el modelo emocional.
  - `data_emocional.csv`: Data con las variables emocionales sintéticas creadas.
  - `data_riesgo_clinico.csv`: Data con el riesgo predicho por el modelo clínico.
  - `datos_limpios.csv`: Datos clínicos codificados y listos para modelado.
  - `datos_limpios_sin_encoder.csv`: Mismos datos anteriores sin codificar (valores originales).

- `outputs/`: Salidas del preprocesamiento y entrenamiento de modelos.
  - `data_clin/`: Scalers y conjuntos de entrenamiento y validación para el modelo clínico.
  - `data_emo/`: Columnas usadas, scalers y datasets para el modelo emocional.
  - `encoder/`: Diccionarios de codificación usados para las variables clínicas.
  - `encoder_emo/`: Diccionarios para codificar las variables emocionales.
  - `models_clin/`: Modelos clínicos entrenados (`cox_ph.pkl`, `gradient_boosting.pkl`, `svm.pkl`, `random_survival_forests.pkl`, `model_scores.csv`).
  - `models_emo/`: Modelos emocionales entrenados con las mismas variantes.

- `Procesamiento/`: Notebooks de procesamiento y exploración.
  - `Exploracion.ipynb`: Exploración básica de los datos.
  - `Exploracion_Avanzada.ipynb`: Análisis exploratorio más profundo.
  - `Extract.ipynb`: Script inicialmente diseñado para extracción (no utilizado finalmente).
  - `Preparacion_Inicial.ipynb`: Limpieza y agregación para generar el `datawarehouse`.
  - `Transformacion.ipynb`: Preparación avanzada para generar `datos_limpios`.

- `Utilidades/`: Scripts y recursos de apoyo al preprocesamiento.
  - `columnas_bool.csv`: Identificación de columnas booleanas.
  - `diccionario_valores.ipynb`: Diccionario de correspondencia de valores.

- `01_modelo_clinic.ipynb`: Entrenamiento del modelo clínico.
- `02_create_data_emo.ipynb`: Generación de las variables emocionales.
- `03_modelos_emo.ipynb`: Entrenamiento y validación del modelo emocional.
- `04_eval_model_emo.ipynb`: Visualizaciones y pruebas de evaluación del modelo emocional.
- `app.py`: Archivo base para la aplicación interactiva (Streamlit).
- `requirements.txt`: Lista de dependencias necesarias para reproducir el proyecto.


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
