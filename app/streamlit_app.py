import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
EPOCHS = ["1958-1969", "1970s", "1980s", "1990s", "2000s", "2010s", "2020-2025"]

st.set_page_config(page_title="Billboard Data Intelligence", page_icon="🎧", layout="wide")


def show_html(html: str, height: int) -> None:
    """`st.iframe` en versiones recientes; `components.html` en las anteriores."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


@st.cache_data(ttl=60, show_spinner=False)
def api(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def api_text(path: str, **params) -> str:
    r = requests.get(f"{API_URL}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=60)
    r.raise_for_status()
    return r.text


st.title("🎧 Billboard Data Intelligence")
st.caption("¿Cómo ha evolucionado el sonido de las canciones #1 del Hot 100: cambió su perfil y se volvió más homogéneo?")

try:
    meta = api("/summary")
except requests.RequestException as e:
    st.error(f"No hay resultados disponibles en {API_URL}. Ejecuta `make pipeline` y reinicia la API.\n\n{e}")
    st.stop()

tab_over, tab_sound, tab_div, tab_net, tab_quality, tab_dl = st.tabs(
    ["Resumen", "Perfil sonoro", "Diversidad sonora", "Redes de géneros", "Calidad de datos", "Descargas"])

# ---------------- Resumen ----------------
with tab_over:
    pd_ = meta["permdisp"]
    c = st.columns(5)
    c[0].metric("Canciones #1", f"{meta['n_raw']:,}")
    c[1].metric("Con audio completo", f"{meta['n_audio_complete']:,}")
    c[2].metric("F (PERMDISP)", f"{pd_['F']:.2f}")
    c[3].metric("p (permutaciones)", f"{pd_['p_value']:.4f}")
    c[4].metric("Tamaño de efecto η²", f"{pd_['eta_squared']:.3f}")
    st.markdown(f"**Decisión:** `{pd_['decision']}` — la dispersión multivariada del sonido difiere entre al menos dos épocas.")
    if meta.get("pairwise_total"):
        st.markdown(f"Comparaciones por pares significativas tras Holm: **{meta['pairwise_significant']} de {meta['pairwise_total']}**.")
    st.info("**Cómo leerlo.** Medimos qué tan lejos está cada canción de la canción típica de su época. "
            "Si las distancias fueran iguales entre épocas, barajar las etiquetas al azar daría resultados parecidos; "
            "no es el caso. El efecto es **pequeño** (η² ≈ 4%): hay diferencias reales, pero la época explica "
            "solo una parte de la variación. Son patrones, no causas, y solo cubren canciones que llegaron al #1.")
    with st.expander("Reproducibilidad"):
        st.json({k: meta[k] for k in ("generated_at_utc", "dataset_path", "dataset_sha256", "parameters", "versions")})

# ---------------- Perfil sonoro ----------------
with tab_sound:
    prof = pd.DataFrame(api("/profile"))
    feat = st.selectbox("Característica", sorted(prof["Feature"].unique()))
    d = prof[prof["Feature"] == feat].copy()
    d["Epoch"] = pd.Categorical(d["Epoch"], EPOCHS, ordered=True)
    d = d.sort_values("Epoch")
    fig = px.line(d, x="Epoch", y="mean", markers=True, error_y="std", title=f"{feat}: promedio por época (barras = ±1 desv. est.)")
    fig.update_layout(xaxis_title="Época", yaxis_title="Valor promedio")
    st.plotly_chart(fig)
    st.dataframe(d[["Epoch", "count", "mean", "median", "std"]].round(2), hide_index=True)

# ---------------- Diversidad ----------------
with tab_div:
    songs = pd.DataFrame(api("/dispersion/songs"))
    songs["Epoch"] = pd.Categorical(songs["Epoch"], EPOCHS, ordered=True)
    fig = px.box(songs.sort_values("Epoch"), x="Epoch", y="Distance_To_Centroid", points="outliers",
                 title="Distancia de cada canción al centroide de su época (menor = más homogéneo)")
    st.plotly_chart(fig)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Resumen por época")
        st.dataframe(pd.DataFrame(api("/dispersion")).round(3), hide_index=True)
    with right:
        st.subheader("Diferencia de dispersión (B − A)")
        pw = pd.DataFrame(api("/dispersion/pairwise"))
        mat = pw.pivot(index="Epoch_A", columns="Epoch_B", values="Mean_Difference_B_minus_A").reindex(index=EPOCHS, columns=EPOCHS)
        txt = pw.assign(label=pw["Mean_Difference_B_minus_A"].map("{:+.2f}".format) + pw["Significant_Holm"].map({True: "*", False: ""})) \
                .pivot(index="Epoch_A", columns="Epoch_B", values="label").reindex(index=EPOCHS, columns=EPOCHS)
        fig = px.imshow(mat, color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto")
        fig.update_traces(text=txt.fillna("").values, texttemplate="%{text}")
        st.plotly_chart(fig)
        st.caption("* significativa tras la corrección de Holm. Azul = la época B es más dispersa que la A; rojo = menos.")
    with st.expander("Todas las comparaciones por pares"):
        st.dataframe(pw.round(5), hide_index=True)

# ---------------- Redes ----------------
with tab_net:
    sources = api("/genres/sources")
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    source = c1.selectbox("Taxonomía", sources)
    epoch = c2.selectbox("Época", ["Todas"] + EPOCHS)
    min_w = c3.slider("Peso mínimo de la conexión", 1, 20, 1)
    ep = None if epoch == "Todas" else epoch
    try:
        html = api_text(f"/networks/{source}/html", epoch=ep, min_weight=min_w)
        show_html(html, 680)
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        st.warning(f"No hay red para esta combinación: {detail}")
    except requests.RequestException as e:
        st.warning(f"No se pudo generar la red: {e}")
    st.caption("Nodo = género (tamaño ∝ canciones). Conexión = canciones que comparten ambos géneros. "
               "Las posiciones son fijas para poder comparar épocas. Es una visualización exploratoria: "
               "la cercanía en el dibujo no es una medida estadística.")
    a, b = st.columns(2)
    with a:
        st.subheader("Asociaciones más fuertes")
        metric = st.radio("Ordenar por", ["Lift", "Jaccard", "Weight"], horizontal=True)
        assoc = pd.DataFrame(api(f"/genres/{source}/associations", epoch=ep, min_weight=max(min_w, 2), sort_by=metric))
        cols = ["Source", "Target", "Weight", "Jaccard", "Lift", "Association_Type"]
        st.dataframe(assoc[[c for c in cols if c in assoc]].round(3) if len(assoc) else assoc, hide_index=True)
        st.caption("Lift > 1: aparecen juntos más de lo esperado por azar. Jaccard: solapamiento entre ambos géneros.")
    with b:
        st.subheader("Prevalencia por época")
        prev = pd.DataFrame(api(f"/genres/{source}/prevalence"))
        prev = prev[prev["Percentage"] >= 2]
        prev["Epoch"] = pd.Categorical(prev["Epoch"], EPOCHS, ordered=True)
        fig = px.area(prev.sort_values("Epoch"), x="Epoch", y="Percentage", color="Tag_Canonical",
                      title="% de canciones clasificadas con cada género")
        st.plotly_chart(fig)

# ---------------- Calidad ----------------
with tab_quality:
    q = api("/quality")
    st.subheader("Chequeos lógicos")
    st.dataframe(pd.DataFrame(q["logical"]), hide_index=True)
    c1, c2 = st.columns(2)
    c1.subheader("Cobertura de audio por época")
    c1.dataframe(pd.DataFrame(q["audio_audit"]), hide_index=True)
    c2.subheader("Duplicados")
    c2.dataframe(pd.DataFrame(q["duplicates"]), hide_index=True)
    st.subheader("Columnas con más valores ausentes")
    st.dataframe(pd.DataFrame(q["missing"]), hide_index=True)
    st.caption("En las columnas de medios (obras de teatro, cine, TV) un vacío significa «no aplica», no «dato perdido».")

# ---------------- Descargas ----------------
with tab_dl:
    st.write("Libro de Excel con el resumen y las tablas relevantes (calidad, perfil sonoro, PERMDISP, géneros y asociaciones).")
    try:
        r = requests.get(f"{API_URL}/export/excel", timeout=60)
        r.raise_for_status()
        st.download_button("⬇️ Descargar billboard_results.xlsx", r.content, file_name="billboard_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except requests.RequestException as e:
        st.warning(f"Excel no disponible: {e}")
    if st.button("Recalcular análisis (999 permutaciones)"):
        with st.spinner("Ejecutando el pipeline…"):
            rr = requests.post(f"{API_URL}/pipeline/run", params={"n_permutations": 999}, timeout=300)
        st.success("Listo.") if rr.ok else st.error(rr.text)
        st.cache_data.clear()
