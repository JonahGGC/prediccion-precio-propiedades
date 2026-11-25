# Proyecto Final 2: Predicción del Valor de Propiedades Inmobiliarias

Estadística Aplicada a la Investigación · Maestría en Ingeniería en Computación, Facultad de Ingeniería, UACH
Alumno: 345491 González Cardoza Jonahtan · Catedrático: Dr. José Saul De Lira Miramontes

## Estructura

```
app_dashboard.py                 Dashboard en Streamlit
Proyecto_final_2_analisis.ipynb  Análisis completo en Python (Jupyter)
requirements.txt                 Librerías necesarias
dataset/                         Dataset limpio (comprimido) e indicadores por estado
modelos/                         Modelo XGBoost, coeficientes de la regresión y métricas
reporte/                         Reporte en PDF, código LaTeX, figuras y logos
```

## Cómo ejecutar el dashboard

Desde esta carpeta:

```
pip install -r requirements.txt
streamlit run app_dashboard.py
```

El dashboard solo necesita lo que ya viene en `dataset/`, `modelos/` y `reporte/`.

## Cómo ejecutar el notebook

El dataset original no viene en el repositorio porque pesa 179 MB (GitHub no acepta archivos de más de 100 MB).
Descarga `realtor-data.zip.csv` del dataset "USA Real Estate Dataset" de Kaggle
(https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) y ponlo en `dataset/`.

Después ejecuta el notebook desde esta carpeta. Lee `dataset/realtor-data.zip.csv` y `dataset/indicadores_estados.csv`,
y vuelve a generar `dataset/datos_propiedades_limpios.csv.gz`, los archivos de `modelos/` y las figuras de `reporte/figuras/`.
Tarda unos 3 minutos.

## Datos

- Realtor.com (Kaggle): casas vendidas en Estados Unidos entre octubre de 2021 y mayo de 2022; quedan 579,734 después de la limpieza.
- `indicadores_estados.csv`: PIB per cápita nominal 2025 (BEA y Census Bureau) y tasas de crimen violento y homicidio 2024 (FBI) por estado.

El reporte se compila con XeLaTeX (o pdfLaTeX) dentro de la carpeta `reporte/`.
