# Dashboard del Proyecto final 2: predicción del precio de propiedades
# Ejecutar en la terminal, desde la carpeta principal del repositorio:
#     streamlit run app_dashboard.py
# Lee los datos de dataset/, los modelos de modelos/ y los logos de reporte/

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Precio de propiedades", layout="wide")
base_dir = Path(__file__).resolve().parent

labels = {"price": "Precio (USD)",
          "house_size_m2": "Superficie (m²)",
          "lot_size_m2": "Terreno (m²)",
          "bed": "Recámaras",
          "bath": "Baños",
          "gdp_per_capita": "PIB per cápita (USD)",
          "violent_crime_rate": "Crimen violento (por 100 mil)",
          "homicide_rate": "Homicidios (por 100 mil)",
          "month_index": "Mes de venta",
          "zip_price_m2": "Precio por m² de la zona (USD)"}
features = ["house_size_m2", "lot_size_m2", "bed", "bath", "gdp_per_capita", "violent_crime_rate", "month_index", "zip_price_m2"]
month_names = ["oct 2021", "nov 2021", "dic 2021", "ene 2022", "feb 2022", "mar 2022", "abr 2022", "may 2022"]


# ---------- Datos y modelos ----------
@st.cache_data
def load_data():
    houses = pd.read_csv(base_dir / "dataset" / "datos_propiedades_limpios.csv.gz", dtype={"zip_code": str}, parse_dates=["sale_date"])
    metrics = pd.read_csv(base_dir / "modelos" / "metricas_modelos.csv", index_col="Modelo")
    zip_prices = pd.read_csv(base_dir / "modelos" / "precio_m2_zona.csv", dtype={"zip_code": str}).set_index("zip_code")["zip_price_m2"]
    state_prices = pd.read_csv(base_dir / "modelos" / "precio_m2_estado.csv").set_index("state")["state_price_m2"]
    coefficients = pd.read_csv(base_dir / "modelos" / "coeficientes_regresion.csv").set_index("variable")["coefficient"]
    state_data = pd.read_csv(base_dir / "dataset" / "indicadores_estados.csv").set_index("state")
    return houses, metrics, zip_prices, state_prices, coefficients, state_data


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(str(base_dir / "modelos" / "modelo_xgboost.json"))
    return model


houses, metrics, zip_prices, state_prices, coefficients, state_data = load_data()
xgb_model = load_model()
best_name = metrics["R²"].idxmax()

by_state = houses.groupby("state").agg(houses_sold=("price", "size"),
                                       median_price=("price", "median"),
                                       median_price_m2=("price_m2", "median"),
                                       gdp_per_capita=("gdp_per_capita", "first"),
                                       violent_crime_rate=("violent_crime_rate", "first")).reset_index()
big_states = by_state[by_state["houses_sold"] >= 1000]

# ---------- Encabezado e indicadores ----------
st.title("¿Cuánto vale una casa? Predicción del precio de propiedades")
st.caption("Proyecto final 2 · Estadística Aplicada a la Investigación · Casas vendidas en Estados Unidos (Realtor.com, oct 2021 a may 2022)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Casas analizadas", f"{len(houses):,}")
col2.metric("Precio mediano", f"{houses['price'].median():,.0f} USD")
col3.metric(f"Mejor modelo: {best_name}", f"R² = {metrics.loc[best_name, 'R²']:.3f}")
col4.metric("Error medio del mejor modelo (MAE)", f"{metrics.loc[best_name, 'MAE']:,.0f} USD")

tab_intro, tab_data, tab_eda, tab_corr, tab_models, tab_pred = st.tabs(
    ["Presentación", "Datos", "Exploración", "Correlación", "Modelos", "Predicción"])

# ---------- Presentación (portada) ----------
with tab_intro:
    logo_left, header_center, logo_right = st.columns([1, 4, 1])
    logo_left.image(str(base_dir / "reporte" / "logo_uach.png"), width=120)
    header_center.markdown(
        "<div style='text-align:center'>"
        "<div style='font-size:1.9rem; font-weight:700'>Universidad Autónoma de Chihuahua</div>"
        "<div style='font-size:1.4rem; font-weight:700'>Facultad de Ingeniería</div>"
        "<div style='font-size:1.1rem; font-weight:600'>Maestría en Ingeniería en Computación</div>"
        "</div>", unsafe_allow_html=True)
    logo_right.image(str(base_dir / "reporte" / "logo_fing.png"), width=120)
    st.markdown(
        "<div style='text-align:center; margin-top:2.5rem'>"
        "<div style='font-size:2.2rem; font-weight:700; line-height:1.25'>Proyecto Final 2:<br>"
        "Predicción del Valor de Propiedades Inmobiliarias</div>"
        "<div style='font-size:1.3rem; font-style:italic; margin-top:1.5rem'>Estadística Aplicada a la Investigación</div>"
        "<div style='font-size:1.1rem'><b><i>Catedrático:</i></b> <i>Dr. José Saul De Lira Miramontes</i></div>"
        "</div>"
        "<div style='text-align:right; font-size:1.3rem; margin-top:2.5rem'>"
        "<b><i>Alumno:</i></b><br><b>345491</b> González Cardoza Jonahtan</div>"
        "<div style='text-align:center; font-size:1.2rem; margin-top:2rem'>25 de noviembre de 2026</div>",
        unsafe_allow_html=True)
    st.divider()
    st.subheader("El caso")
    st.markdown(
        "Se busca un modelo que estime el precio de una vivienda con sus características (superficie, terreno, "
        "recámaras, baños), su ubicación y los indicadores económicos del estado (PIB per cápita y criminalidad). "
        "Primero se usan correlación y regresión lineal, y después se comparan cinco modelos de *machine learning*.\n\n"
        "**Fases:** recolección y limpieza → análisis exploratorio y correlación → regresión lineal → "
        "*machine learning* → comparación con MSE, RMSE, MAE y R² → este dashboard.")

# ---------- Datos ----------
with tab_data:
    st.subheader("Fuentes")
    st.markdown(
        "- **Realtor.com (Kaggle):** 2,226,382 anuncios de Estados Unidos. Se usan solo las casas vendidas.\n"
        "- **PIB per cápita nominal 2025** por estado (BEA y Census Bureau).\n"
        "- **Crimen violento y homicidios por cada 100 mil habitantes, 2024** por estado (FBI).")
    cleaning = pd.DataFrame({"Paso": ["Archivo original", "Solo casas vendidas", "Sin datos faltantes", "Con indicadores del estado",
                                      "Rangos válidos", "Sin extremos de precio por pie²"],
                             "Registros": [2226382, 812009, 609776, 609504, 591566, len(houses)]})
    left, right = st.columns([1, 1])
    left.subheader("Limpieza")
    left.dataframe(cleaning, hide_index=True, width="stretch")
    right.subheader("Resumen")
    right.dataframe(houses[["price", "house_size_m2", "lot_size_m2", "bed", "bath"]].describe().T.rename(index=labels).round(1),
                    width="stretch")
    st.subheader("Vista de los datos")
    state_filter = st.multiselect("Filtrar por estado", sorted(houses["state"].unique()), default=[])
    view = houses[houses["state"].isin(state_filter)] if state_filter else houses
    st.dataframe(view.head(500), hide_index=True, width="stretch")
    st.caption(f"Se muestran 500 de {len(view):,} casas.")

# ---------- Exploración ----------
with tab_eda:
    variable = st.selectbox("Variable", ["price", "house_size_m2", "lot_size_m2", "price_m2"],
                            format_func=lambda v: {"price_m2": "Precio por m² (USD)"}.get(v, labels.get(v, v)))
    upper = houses[variable].quantile(0.99)
    fig = px.histogram(houses[houses[variable] <= upper], x=variable, nbins=80,
                       labels={variable: labels.get(variable, "Precio por m² (USD)")},
                       title="Distribución (sin el 1 % más alto, para que se vea mejor)")
    fig.add_vline(x=houses[variable].median(), line_dash="dash", annotation_text="Mediana")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    fig = px.box(houses[houses["price"] <= houses["price"].quantile(0.99)], x="bed", y="price", points=False,
                 labels={"bed": "Recámaras", "price": "Precio (USD)"}, title="Precio según el número de recámaras")
    left.plotly_chart(fig, width="stretch")
    fig = px.bar(big_states.sort_values("median_price"), x="median_price", y="state", orientation="h",
                 labels={"median_price": "Precio mediano (USD)", "state": ""},
                 title="Precio mediano por estado (estados con 1,000 ventas o más)", height=700)
    right.plotly_chart(fig, width="stretch")

    monthly = houses.groupby(houses["sale_date"].dt.to_period("M"))["price_m2"].median().reset_index()
    monthly["sale_date"] = monthly["sale_date"].astype(str)
    fig = px.line(monthly.iloc[1:-1], x="sale_date", y="price_m2", markers=True,
                  labels={"sale_date": "Mes de venta", "price_m2": "Precio mediano por m² (USD)"},
                  title="Datos históricos: precio mediano por m² (meses completos, nov 2021 a abr 2022)")
    st.plotly_chart(fig, width="stretch")

# ---------- Correlación ----------
with tab_corr:
    corr_vars = ["price", "house_size_m2", "lot_size_m2", "bed", "bath", "gdp_per_capita", "violent_crime_rate", "homicide_rate", "month_index"]
    method = st.radio("Coeficiente", ["pearson", "spearman"], horizontal=True,
                      format_func=lambda m: "Pearson (lineal)" if m == "pearson" else "Spearman (rangos)")
    matrix = houses[corr_vars].corr(method=method).rename(index=labels, columns=labels)
    fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", height=620)
    st.plotly_chart(fig, width="stretch")
    st.markdown("Escala: |r| < 0.3 débil, 0.3 a 0.7 moderada, ≥ 0.7 fuerte. Con más de medio millón de casas todas las "
                "correlaciones son significativas, así que importa más su tamaño.")

    left, right = st.columns(2)
    sample = houses.sample(5000, random_state=42)
    fig = px.scatter(sample, x="house_size_m2", y="price", opacity=0.35, trendline="ols",
                     labels={"house_size_m2": "Superficie (m²)", "price": "Precio (USD)"},
                     title="Superficie vs precio (5,000 casas al azar)")
    left.plotly_chart(fig, width="stretch")
    x_state = right.selectbox("Indicador del estado", ["gdp_per_capita", "violent_crime_rate", "homicide_rate"],
                              format_func=lambda v: labels[v])
    fig = px.scatter(big_states, x=x_state, y="median_price_m2", hover_name="state", trendline="ols",
                     labels={x_state: labels[x_state], "median_price_m2": "Precio mediano por m² (USD)"},
                     title=f"Por estado (r = {big_states[x_state].corr(big_states['median_price_m2']):.2f})")
    right.plotly_chart(fig, width="stretch")

# ---------- Modelos ----------
with tab_models:
    st.subheader("Comparación en el conjunto de prueba (20 % de las casas)")
    st.dataframe(metrics.style.format({"MSE": "{:.3e}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R²": "{:.4f}", "Tiempo (s)": "{:.1f}"}, na_rep="--"),
                 width="stretch")
    metric_name = st.radio("Métrica", ["R²", "RMSE", "MAE"], horizontal=True)
    fig = px.bar(metrics.reset_index().sort_values(metric_name), x=metric_name, y="Modelo", orientation="h",
                 text_auto=".3f" if metric_name == "R²" else ",.0f",
                 title=f"{metric_name} por modelo ({'más es mejor' if metric_name == 'R²' else 'menos es mejor'})")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    importance = pd.DataFrame({"Variable": [labels[f] for f in features], "Importancia": xgb_model.feature_importances_ * 100})
    fig = px.bar(importance.sort_values("Importancia"), x="Importancia", y="Variable", orientation="h",
                 title="Importancia de las variables en XGBoost (%)")
    left.plotly_chart(fig, width="stretch")
    coef_view = pd.DataFrame({"Variable": ["Intercepto"] + [labels[f] for f in features], "Coeficiente": coefficients.values})
    right.markdown("**Regresión lineal 2** (coeficientes por mínimos cuadrados, b = (X'X)⁻¹X'y)")
    right.dataframe(coef_view.style.format({"Coeficiente": "{:,.4f}"}), hide_index=True, width="stretch")

# ---------- Predicción ----------
with tab_pred:
    st.subheader("Simulador: estima el precio de una casa")
    left, right = st.columns([1, 1])
    states_list = sorted(houses["state"].unique())
    state = left.selectbox("Estado", states_list, index=states_list.index("Texas"))
    cities = houses.loc[houses["state"] == state, "city"].value_counts()
    city = left.selectbox("Ciudad (ordenadas por número de ventas)", cities.index)
    zips = houses.loc[(houses["state"] == state) & (houses["city"] == city), "zip_code"].value_counts()
    zip_code = left.selectbox("Código postal (ordenados por número de ventas)", zips.index,
                              format_func=lambda z: f"{z} ({zips[z]} ventas)")
    month = left.select_slider("Mes de venta", options=list(range(8)), value=6, format_func=lambda m: month_names[m])
    size_m2 = right.slider("Superficie construida (m²)", 30, 900, 180, step=5)
    lot_m2 = right.slider("Terreno (m²)", 50, 40000, 600, step=50)
    bed = right.slider("Recámaras", 1, 10, 3)
    bath = right.slider("Baños", 1, 10, 2)

    if zip_code in zip_prices.index:
        zone_price = zip_prices[zip_code]
        zone_text = f"precio mediano por m² del código postal {zip_code}"
    else:
        zone_price = state_prices[state]
        zone_text = f"el código postal tiene menos de 5 ventas; se usa el precio mediano por m² de {state}"
    row = pd.DataFrame([{"house_size_m2": size_m2, "lot_size_m2": lot_m2, "bed": bed, "bath": bath,
                         "gdp_per_capita": state_data.loc[state, "gdp_per_capita"],
                         "violent_crime_rate": state_data.loc[state, "violent_crime_rate"],
                         "month_index": month, "zip_price_m2": zone_price}])[features]
    price_xgb = float(xgb_model.predict(row)[0])
    price_lr = float(coefficients["intercept"] + (row.iloc[0] * coefficients[features]).sum())

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio estimado con XGBoost", f"{price_xgb:,.0f} USD")
    c2.metric("Precio con regresión lineal 2", f"{price_lr:,.0f} USD")
    c3.metric("Precio por m² de la zona", f"{zone_price:,.0f} USD/m²")
    st.caption(f"Zona: {zone_text}. PIB per cápita de {state}: {state_data.loc[state, 'gdp_per_capita']:,.0f} USD; "
               f"crimen violento: {state_data.loc[state, 'violent_crime_rate']:.1f} por 100 mil. "
               f"Error típico de XGBoost: ± {metrics.loc['XGBoost', 'MAE']:,.0f} USD (MAE).")

    similar = houses[(houses["zip_code"] == zip_code)]
    if len(similar) > 0:
        st.markdown(f"**Casas vendidas en el código postal {zip_code}:** {len(similar)} (precio mediano {similar['price'].median():,.0f} USD)")
        fig = px.scatter(similar, x="house_size_m2", y="price", hover_data=["bed", "bath", "sale_date"],
                         labels={"house_size_m2": "Superficie (m²)", "price": "Precio (USD)"})
        fig.add_scatter(x=[size_m2], y=[price_xgb], mode="markers", marker=dict(size=16, symbol="star", color="red"),
                        name="Tu casa (XGBoost)")
        st.plotly_chart(fig, width="stretch")
