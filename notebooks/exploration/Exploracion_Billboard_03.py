#!/usr/bin/env python
# coding: utf-8

# # 01. Planteamiento del problema
# 
# ## Contexto
# 
# El Billboard Hot 100 es una de las principales referencias históricas para observar la evolución de la música popular en Estados Unidos. Desde 1958, las canciones que han alcanzado la primera posición reflejan cambios en las preferencias del público, las tendencias de producción musical y la diversidad de estilos que han conseguido mayor difusión comercial.
# 
# Sin embargo, alcanzar el primer lugar no significa necesariamente que las canciones compartan las mismas características sonoras, ni que la música popular haya evolucionado de manera uniforme.
# 
# Este análisis busca explorar cómo ha cambiado el sonido de las canciones que alcanzaron el número uno del Billboard Hot 100 entre 1958 y 2025, considerando tanto las transformaciones de sus características musicales como la diversidad y las relaciones entre los géneros presentes en cada época.
# 
# ## Pregunta de investigación
# 
# **¿Cómo ha evolucionado el sonido de las canciones que alcanzaron el número uno del Billboard Hot 100: ha cambiado su perfil sonoro y se ha vuelto más homogéneo con el paso del tiempo?**
# 
# ## Objetivo general
# 
# Analizar la evolución de las características sonoras y la diversidad musical de las canciones que alcanzaron el número uno del Billboard Hot 100 entre 1958 y 2025, identificando patrones temporales, diferencias entre épocas y relaciones entre géneros musicales.
# 
# ## Objetivos específicos
# 
# 1. Identificar cambios en las características sonoras de las canciones a través del tiempo.
# 2. Examinar si la diversidad sonora dentro de cada época ha aumentado, disminuido o permanecido relativamente estable.
# 3. Explorar las relaciones y patrones de coexistencia entre géneros y estilos musicales a lo largo de las décadas.
# 4. Evaluar los resultados mediante métodos estadísticos apropiados para las características y distribuciones observadas.
# 5. Comunicar los hallazgos mediante visualizaciones que faciliten su interpretación.
# 
# ## Dimensiones de análisis
# 
# ### A. Evolución del perfil sonoro
# 
# Examinar cómo han cambiado características como el tempo, la energía, la bailabilidad, la felicidad percibida, la intensidad sonora y la acústica.
# 
# ### B. Diversidad sonora
# 
# Evaluar qué tan diferentes son las canciones dentro de cada época, utilizando una medida de dispersión multivariada respecto al perfil sonoro promedio de su periodo.
# 
# ### C. Relaciones entre géneros musicales
# 
# Representar los géneros y estilos como redes de coocurrencia, donde las conexiones reflejan su presencia conjunta en las canciones. Esto permitirá explorar cómo cambian las asociaciones entre categorías musicales a través del tiempo.
# 
# ## Hipótesis de investigación
# 
# La hipótesis estadística principal se definirá después de examinar las distribuciones, la cobertura de los datos y los patrones exploratorios.
# 
# Una posibilidad que se evaluará es que la diversidad sonora de las canciones que alcanzaron el número uno haya cambiado entre las distintas épocas, incluyendo la posibilidad de una reducción de la dispersión en periodos más recientes.
# 
# No se asumirá de antemano que existe una tendencia lineal ni que la música se ha vuelto progresivamente homogénea.
# 
# ## Alcance y consideraciones
# 
# - El análisis se limita a las canciones incluidas en el conjunto de datos que alcanzaron el número uno; no representa toda la música producida o publicada durante cada época.
# - Las características sonoras disponibles pueden provenir de mediciones automatizadas y no necesariamente capturan todos los elementos musicales.
# - Las clasificaciones de género y estilo dependen de las fuentes y criterios de etiquetado disponibles.
# - Las diferencias observadas representan asociaciones y patrones temporales, no relaciones causales.
# - Las comparaciones entre épocas considerarán la disponibilidad y calidad de los datos para evitar conclusiones basadas en muestras incompletas.
# 
# ## Enfoque metodológico
# 
# El trabajo combinará análisis exploratorio, estadística descriptiva, visualización, análisis de dispersión multivariada y análisis de redes.
# 
# La selección de pruebas estadísticas dependerá de las características de los datos, sus distribuciones, la estructura de las observaciones y los supuestos de cada método. Se priorizará la interpretación y la robustez de los resultados sobre la complejidad matemática.

# In[2]:


## Celda 1 — Importaciones y configuración

from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

# Configuración general y visual
RANDOM_STATE = 42
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 6)

RAW_DATA_PATH = Path("../data/raw/Billboard_Hot_100_Number_Ones_Database.xlsx")

AUDIO_FEATURES = ["BPM", "Energy", "Danceability", "Happiness", "Loudness (dB)", "Acousticness"]
GENRE_COLUMNS = ["CDR Genre", "CDR Style", "Discogs Genre", "Discogs Style"]
IDENTIFIER_COLUMNS = ["Song", "Artist", "Date"]


# In[3]:


## Celda 2 — Carga y preservación de los datos originales
raw = pd.read_excel( RAW_DATA_PATH, sheet_name="Data", na_values=["Cannot Locate"])

# Copia de trabajo
df = raw.copy(deep=True)

print(f"Registros originales: {len(raw):,}")
print(f"Columnas originales: {raw.shape[1]:,}")

display(df.head())


# In[4]:


## Celda 3 — Validación estructural inicial 
required_columns = IDENTIFIER_COLUMNS + AUDIO_FEATURES + GENRE_COLUMNS + ["Weeks at Number One"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Faltan columnas necesarias: {missing_columns}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print(f"Dimensiones: {df.shape} | Fechas: {df['Date'].min().date()} a {df['Date'].max().date()} | Nulas: {df['Date'].isna().sum()}")


# In[5]:


## Celda 4 — Identificación de duplicados

duplicate_checks = {
    "Filas completamente duplicadas": df.duplicated().sum(),
    "Canción, artista y fecha duplicados": df.duplicated(subset=["Song", "Artist", "Date"]).sum(),
    "Canción y artista duplicados": df.duplicated(subset=["Song", "Artist"]).sum(),
}

display(pd.Series(duplicate_checks, name="Cantidad").to_frame())


# In[6]:


## Celda 5 — Variables temporales

df["Year"] = df["Date"].dt.year
period_bins = [1958, 1970, 1980, 1990, 2000, 2010, 2020, 2026]
period_labels = ["1958-1969", "1970s", "1980s", "1990s", "2000s", "2010s", "2020-2025"]

df["Epoch"] = pd.Categorical(
    pd.cut(df["Year"], bins=period_bins, labels=period_labels, right=False),
    categories=period_labels,
    ordered=True
)

display(df.groupby("Epoch", observed=False).size().rename("Songs").to_frame())


# In[7]:


## Celda 6 — Indicadores de calidad

# Valores ausentes
missing_summary = df.isna().sum().to_frame("Missing")

missing_summary["Missing_Percentage"] = (missing_summary["Missing"] / len(df) * 100).round(2)

display(missing_summary.sort_values("Missing_Percentage", ascending=False).head(20))


# In[8]:


## Celda 7 — Identificación de BPM sospechosos

df["BPM_flagged"] = df["BPM"].notna() & ((df["BPM"] < 40) | (df["BPM"] > 220))
display(df.loc[df["BPM_flagged"], ["Song", "Artist", "Date", "BPM"]])


# In[9]:


## Celda 8 — Auditoría de integridad de la muestra

df["Audio_Complete"] = df[AUDIO_FEATURES].notna().all(axis=1)

audio_audit = df.groupby("Epoch", observed=False).agg(Total_Songs=("Song", "size"), Complete_Audio=("Audio_Complete", "sum"))
audio_audit["Incomplete_Audio"] = audio_audit["Total_Songs"] - audio_audit["Complete_Audio"]
audio_audit["Coverage_Percentage"] = (audio_audit["Complete_Audio"] / audio_audit["Total_Songs"] * 100).round(2)

display(audio_audit)


# In[10]:


incomplete_audio = df.loc[~df["Audio_Complete"], ["Song", "Artist", "Date", "Epoch", *AUDIO_FEATURES, "BPM_flagged"]]

print(f"Total de registros: {len(df)}")
print(f"Registros con audio completo: {df['Audio_Complete'].sum()}")
print(f"Registros con audio incompleto: {len(incomplete_audio)}")

display(incomplete_audio)


# In[11]:


## Celda 9 — Preparación de la muestra analítica

# Copia exclusiva para análisis
analysis = df.copy(deep=True)

# El BPM sospechoso se excluye de la muestra analítica principal.
# No modificamos df ni raw.
analysis.loc[
   analysis["BPM_flagged"],
   "BPM"
] = np.nan

# Muestra con las seis características disponibles
audio_complete = analysis.dropna(
   subset=AUDIO_FEATURES
).copy()

print("Registros originales:", len(raw))
print("Registros con audio completo:", len(audio_complete))
print(
   "Registros excluidos del análisis multivariado:",
   len(analysis) - len(audio_complete)
)

assert len(raw) == 1177
assert len(audio_complete) == 1174


# In[12]:


## Celda 10 — Estadística descriptiva del perfil sonoro

audio_summary = (
    analysis
    .groupby("Epoch", observed=True)[AUDIO_FEATURES]
    .agg(["count", "mean", "median", "std"])
)

display(audio_summary.round(2))


# In[13]:


audio_means = analysis.groupby("Epoch", observed=True)[AUDIO_FEATURES].mean().reset_index()
audio_long = audio_means.melt(id_vars="Epoch", var_name="Feature", value_name="Mean_Value")

g = sns.FacetGrid(audio_long, col="Feature", col_wrap=2, sharey=False, height=3.5, aspect=1.5)
g.map_dataframe(sns.lineplot, x="Epoch", y="Mean_Value", marker="o")
g.set_titles("{col_name}")
g.set_axis_labels("Periodo", "Valor promedio")

for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=45)

g.figure.suptitle("Evolución del perfil sonoro de las canciones #1", y=1.03)
plt.tight_layout()
plt.show()


# In[14]:


def calculate_epoch_dispersion(data: pd.DataFrame, features: list[str], epoch_column: str = "Epoch"):
    X = data[features].copy()
    if X.isna().any().any():
        raise ValueError("Existen valores ausentes en las características.")
    
    means, stds = X.mean(), X.std()
    if (stds == 0).any():
        raise ValueError("No se puede estandarizar una característica constante.")
    
    Z = (X - means) / stds
    row_centroids = Z.groupby(data[epoch_column], observed=True).transform("mean")
    distances = np.sqrt(((Z - row_centroids) ** 2).sum(axis=1))
    
    result = data.copy()
    result["Distance_To_Centroid"] = distances
    scaler = pd.DataFrame({"Mean": means, "Standard_Deviation": stds})
    
    return result, Z.groupby(data[epoch_column], observed=True).mean(), scaler

dispersion_data, epoch_centroids, audio_scaler = calculate_epoch_dispersion(audio_complete, AUDIO_FEATURES)

dispersion_summary = dispersion_data.groupby("Epoch", observed=True)["Distance_To_Centroid"].agg(
    Count="count", Mean="mean", Median="median", Std="std"
)

display(dispersion_summary.round(3))


# In[15]:


plt.figure(figsize=(12, 6))

sns.boxplot(
    data=dispersion_data,
    x="Epoch",
    y="Distance_To_Centroid",
)

plt.title(
    "Dispersión del perfil sonoro dentro de cada época"
)

plt.xlabel("Periodo")
plt.ylabel("Distancia al centroide")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Celda 13 — Inspección de géneros y estilos musicales
# 
# ## Antes de construirlas, necesitamos entender cómo están almacenadas las etiquetas.
# ##  No vamos a mezclar indiscriminadamente CDR Genre, CDR Style, Discogs Genre y Discogs Style: son fuentes y niveles de clasificación diferentes.
# 
# ## Queremos descubrir cosas como:
# - Si una canción puede tener varios géneros.
# - Si las etiquetas están separadas por comas, punto y coma u otro delimitador.
# - Si existen diferencias entre la clasificación de CDR y Discogs.
# - Cuántas categorías diferentes tenemos.
# - Si hay etiquetas redundantes o inconsistentes.
# Con eso podremos diseñar una red en la que:
# - Nodo: género o estilo musical.
# - Tamaño: frecuencia de aparición en las canciones.
# - Conexión: dos géneros aparecen juntos en una misma canción.
# - Peso de conexión: cantidad de canciones que comparten esa combinación.
# - Periodo: permite comparar cómo cambian las relaciones a través del tiempo.

# In[18]:


for column in GENRE_COLUMNS:

    print(f"\n{'=' * 60}")
    print(f"VARIABLE: {column}")
    print(f"{'=' * 60}")

    print("Valores no nulos:", df[column].notna().sum())
    print("Valores únicos:", df[column].nunique())

    print("\nEtiquetas más frecuentes:")

    display(
        df[column]
        .dropna()
        .value_counts()
        .head(20)
        .to_frame("Frequency")
    )

    print("\nEjemplos originales:")

    display(
        df[column]
        .dropna()
        .sample(
            min(5, df[column].notna().sum()),
            random_state=RANDOM_STATE,
        )
    )


# ## 06. Exploración de géneros y estilos musicales
# 
# ### Contexto
# 
# Las características sonoras permiten estudiar cómo ha cambiado el perfil musical de las canciones a lo largo del tiempo. Sin embargo, estas características no describen por completo la diversidad de la música popular.
# 
# Para complementar el análisis, exploraremos las clasificaciones de géneros y estilos musicales disponibles en el conjunto de datos, procedentes de dos fuentes: CDR y Discogs.
# 
# Estas clasificaciones permiten examinar no solamente qué categorías musicales aparecen con mayor frecuencia, sino también cómo se relacionan entre sí y cómo cambian sus asociaciones a través de los distintos periodos históricos.
# 
# ### Estructura de las etiquetas
# 
# Una canción puede estar asociada a más de un género o estilo. En estos casos, las categorías aparecen almacenadas en una misma celda y separadas mediante punto y coma (`;`).
# 
# Es importante conservar las barras (`/`) y las comas (`,`) que forman parte de determinados nombres de categorías. Por ejemplo, `Funk/Soul` y `Folk, World, & Country` deben interpretarse como etiquetas individuales.
# 
# Por esta razón, será necesario transformar estas columnas a una estructura de etiquetas individuales antes de calcular frecuencias o construir relaciones.
# 
# ### Fuentes de clasificación
# 
# Se examinarán las cuatro columnas disponibles:
# 
# - **CDR Genre:** clasificación de géneros musicales.
# - **CDR Style:** clasificación de estilos musicales.
# - **Discogs Genre:** clasificación de géneros musicales según Discogs.
# - **Discogs Style:** clasificación de estilos musicales según Discogs.
# 
# No se combinarán directamente las categorías de ambas fuentes, ya que pueden utilizar criterios de clasificación, niveles de granularidad y coberturas diferentes.
# 
# ### Exploración de las relaciones entre géneros
# 
# Una vez identificadas y normalizadas las etiquetas, se explorará su coexistencia mediante redes de coocurrencia.
# 
# En estas representaciones:
# 
# - **Nodos:** representan géneros musicales.
# - **Tamaño de los nodos:** representa la frecuencia con la que aparece cada género.
# - **Conexiones:** representan géneros que aparecen asociados a una misma canción.
# - **Peso de las conexiones:** representa la cantidad de canciones en las que aparece cada combinación.
# - **Periodo histórico:** permite examinar cómo cambian estas relaciones a través del tiempo.
# 
# El análisis permitirá identificar categorías recurrentes, asociaciones frecuentes y posibles cambios en la estructura de los géneros musicales presentes entre las canciones que alcanzaron el número uno.
# 
# ### Alcance e interpretación
# 
# Las relaciones identificadas representan coexistencia de etiquetas dentro de una misma canción. No implican necesariamente influencia musical, evolución directa de un género hacia otro ni relaciones causales.
# 
# Las redes se utilizarán como herramientas exploratorias para complementar el análisis de las características sonoras y comprender mejor la diversidad musical a lo largo del periodo estudiado.

# In[21]:


## Celda 14 — Normalización de las etiquetas
def split_music_tags(value):
    """
    Separates multiple music tags stored in a single cell.

    The semicolon is the only delimiter.
    Slashes and commas are preserved as part of tag names.
    """
    if pd.isna(value):
        return []

    return [
        tag.strip()
        for tag in str(value).split(";")
        if tag.strip()
    ]
examples = [
    "Rock;Pop",
    "Funk/Soul",
    "Folk, World, & Country",
    "Electronic;Funk/Soul;Pop",
]

for example in examples:
    print(example, "->", split_music_tags(example))


# In[28]:


## Celda 15 — Transformación a formato longitudinal

def create_tag_dataset(
    data: pd.DataFrame,
    tag_column: str,
    source: str,
    taxonomy: str,
) -> pd.DataFrame:

    result = data[
        ["Song", "Artist", "Date", "Epoch", tag_column]
    ].copy()

    result["Record_ID"] = data.index
    result["Tag"] = result[tag_column].apply(
        split_music_tags
    )

    result = result.explode("Tag")

    result = result.dropna(subset=["Tag"])

    result = result[
        ["Record_ID", "Song", "Artist", "Date", "Epoch", "Tag"]
    ].copy()

    result["Source"] = source
    result["Taxonomy"] = taxonomy

    return result.reset_index(drop=True)


# In[30]:


cdr_genres = create_tag_dataset(
    df,
    "CDR Genre",
    source="CDR",
    taxonomy="Genre",
)

cdr_styles = create_tag_dataset(
    df,
    "CDR Style",
    source="CDR",
    taxonomy="Style",
)

discogs_genres = create_tag_dataset(
    df,
    "Discogs Genre",
    source="Discogs",
    taxonomy="Genre",
)

discogs_styles = create_tag_dataset(
    df,
    "Discogs Style",
    source="Discogs",
    taxonomy="Style",
)


# In[32]:


## Celda 16 — Comparación de las taxonomías

tag_datasets = {
    "CDR Genre": cdr_genres,
    "CDR Style": cdr_styles,
    "Discogs Genre": discogs_genres,
    "Discogs Style": discogs_styles,
}

taxonomy_summary = []

for name, dataset in tag_datasets.items():

    taxonomy_summary.append({
        "Taxonomy": name,
        "Song_Tag_Associations": len(dataset),
        "Unique_Tags": dataset["Tag"].nunique(),
        "Songs_Covered": dataset["Record_ID"].nunique(),
    })

taxonomy_summary = pd.DataFrame(taxonomy_summary)

display(taxonomy_summary)


# In[36]:


# ¿Qué fuente ofrece mayor cobertura?
# ¿Cuál presenta mayor granularidad?
# ¿Cuántas etiquetas individuales existen realmente?
# ¿Qué cantidad de asociaciones canción-etiqueta tenemos?

for name, dataset in tag_datasets.items():

    print(f"\n{name}")

    display(
        dataset["Tag"]
        .value_counts()
        .head(20)
        .rename("Frequency")
        .to_frame()
    )


# In[38]:


## Celda 17 — Comparar etiquetas originales y etiquetas individuales
for name, dataset in tag_datasets.items():

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    print("Asociaciones canción-etiqueta:", len(dataset))
    print("Canciones clasificadas:", dataset["Record_ID"].nunique())
    print("Etiquetas individuales:", dataset["Tag"].nunique())

    print("\nEtiquetas individuales:")

    display(
        dataset["Tag"]
        .value_counts()
        .rename("Frequency")
        .to_frame()
    )


# In[40]:


## Celda 18 — Identificar canciones con múltiples etiquetas
for name, dataset in tag_datasets.items():

    tags_per_song = (
        dataset.groupby("Record_ID")["Tag"]
        .agg(list)
    )

    multi_tag_songs = tags_per_song[
        tags_per_song.apply(len) > 1
    ]

    print(f"\n{name}")
    print("Canciones con múltiples etiquetas:", len(multi_tag_songs))

    display(
        multi_tag_songs
        .head(10)
        .to_frame("Tags")
    )


# In[42]:


## Celda 19 — Detectar variantes de escritura

import re

def normalize_tag_format(tag):
    tag = str(tag).strip()
    tag = re.sub(r"\s+", " ", tag)
    tag = re.sub(r"\s*/\s*", "/", tag)
    return tag

for name, dataset in tag_datasets.items():

    variants = dataset[["Tag"]].drop_duplicates().copy()

    variants["Normalized_Tag"] = (
        variants["Tag"].apply(normalize_tag_format)
    )

    formatting_variants = (
        variants.groupby("Normalized_Tag")["Tag"]
        .agg(list)
    )

    formatting_variants = formatting_variants[
        formatting_variants.apply(len) > 1
    ]

    print(f"\n{name} — Posibles variantes")

    display(formatting_variants.to_frame("Original_Variants"))


# In[ ]:


## preparar las etiquetas para el análisis temporal y las redes

# ¿Qué géneros aparecen con mayor frecuencia en cada época?
# ¿Qué géneros suelen coexistir en una misma canción?
# ¿Cómo cambia esa coexistencia entre períodos históricos?


# In[44]:


# Variantes cuya equivalencia textual hemos identificado.
TAG_CANONICAL_MAP = {
    "Funk / Soul": "Funk/Soul",
}


def canonicalize_tag(tag: str) -> str:
    if pd.isna(tag):
        return tag

    tag = " ".join(str(tag).strip().split())
    return TAG_CANONICAL_MAP.get(tag, tag)


tag_datasets_clean = {}

for name, tag_df in tag_datasets.items():
    cleaned = tag_df.copy()
    cleaned["Tag_Canonical"] = cleaned["Tag"].apply(canonicalize_tag)

    tag_datasets_clean[name] = cleaned


# Verificar qué cambió.
for name, tag_df in tag_datasets_clean.items():
    changes = (
        tag_df.loc[tag_df["Tag"] != tag_df["Tag_Canonical"],
                   ["Tag", "Tag_Canonical"]]
        .drop_duplicates()
    )

    print(f"\n{name}")
    if changes.empty:
        print("No se identificaron cambios.")
    else:
        display(changes)


# In[ ]:


\[
\text{Prevalencia}_{g,p} =
\frac{\text{Canciones del período }p\text{ con género }g}
{\text{Canciones clasificadas del período }p}
\times 100
\]


# In[67]:


# Celda 21 — Evolución temporal de los géneros
def calculate_tag_evolution(
    data: pd.DataFrame,
    epoch_column: str = "Epoch",
    tag_column: str = "Tag_Canonical",
) -> pd.DataFrame:

    required_columns = ["Record_ID", epoch_column, tag_column]
    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    working = data[required_columns].dropna().copy()

    # Una canción cuenta una sola vez por etiqueta y período.
    working = working.drop_duplicates(
        subset=["Record_ID", epoch_column, tag_column]
    )

    # Canciones clasificadas por período.
    denominators = (
        working.groupby(epoch_column, observed=True)["Record_ID"]
        .nunique()
    )

    # Frecuencia de cada etiqueta por período.
    evolution = (
        working.groupby(
            [epoch_column, tag_column],
            observed=True
        )["Record_ID"]
        .nunique()
        .rename("Song_Count")
        .reset_index()
    )

    # Convertir explícitamente a valores numéricos.
    evolution["Classified_Songs"] = (
        evolution[epoch_column]
        .map(denominators.to_dict())
        .astype("int64")
    )

    evolution["Song_Count"] = evolution["Song_Count"].astype("int64")

    evolution["Percentage"] = (
        evolution["Song_Count"]
        .div(evolution["Classified_Songs"])
        .mul(100)
    )

    return evolution

genre_evolution = {
    "CDR Genre": calculate_tag_evolution(
        tag_datasets_clean["CDR Genre"]
    ),
    "Discogs Genre": calculate_tag_evolution(
        tag_datasets_clean["Discogs Genre"]
    ),
}

for source, evolution in genre_evolution.items():
    print(f"\n{'=' * 60}")
    print(source)
    print(f"{'=' * 60}")

    display(
        evolution.sort_values(
            ["Epoch", "Percentage"],
            ascending=[True, False]
        )
    )


# In[69]:


## Celda 22 — Construcción de redes de coocurrencia
from itertools import combinations

def build_cooccurrence_network(
    data: pd.DataFrame,
    epoch_column: str = "Epoch",
    tag_column: str = "Tag_Canonical",
    group_by_epoch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    required_columns = ["Record_ID", epoch_column, tag_column]
    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    working = data[required_columns].dropna().copy()

    # Evitar que una etiqueta duplicada genere relaciones duplicadas.
    working = working.drop_duplicates(
        subset=["Record_ID", epoch_column, tag_column]
    )

    # Frecuencia de cada nodo.
    nodes = (
        working.groupby(tag_column)["Record_ID"]
        .nunique()
        .rename("Song_Count")
        .reset_index()
        .rename(columns={tag_column: "Tag"})
    )

    grouping_columns = ["Record_ID", epoch_column]

    # Obtener las etiquetas de cada canción.
    songs = (
        working.groupby(grouping_columns, observed=True)[tag_column]
        .apply(lambda tags: sorted(set(tags)))
        .reset_index(name="Tags")
    )

    edge_records = []

    for _, row in songs.iterrows():
        tags = row["Tags"]

        for tag_a, tag_b in combinations(tags, 2):
            record = {
                "Source": tag_a,
                "Target": tag_b,
                "Record_ID": row["Record_ID"],
            }

            if group_by_epoch:
                record[epoch_column] = row[epoch_column]

            edge_records.append(record)

    if edge_records:
        edges_raw = pd.DataFrame(edge_records)

        aggregation_columns = ["Source", "Target"]

        if group_by_epoch:
            aggregation_columns.append(epoch_column)

        edges = (
            edges_raw.groupby(aggregation_columns)["Record_ID"]
            .nunique()
            .rename("Weight")
            .reset_index()
        )
    else:
        columns = ["Source", "Target", "Weight"]

        if group_by_epoch:
            columns.append(epoch_column)

        edges = pd.DataFrame(columns=columns)

    return nodes, edges


# In[71]:


genre_networks = {}

for source in ["CDR Genre", "Discogs Genre"]:

    data = tag_datasets_clean[source]

    nodes, edges = build_cooccurrence_network(
        data,
        group_by_epoch=False,
    )

    nodes_by_epoch, edges_by_epoch = build_cooccurrence_network(
        data,
        group_by_epoch=True,
    )

    genre_networks[source] = {
        "nodes": nodes,
        "edges": edges,
        "nodes_by_epoch": nodes_by_epoch,
        "edges_by_epoch": edges_by_epoch,
    }

    print(f"\n{'=' * 60}")
    print(source)
    print(f"{'=' * 60}")

    print(f"Nodos: {len(nodes)}")
    print(f"Relaciones únicas: {len(edges)}")

    display(
        edges.sort_values("Weight", ascending=False).head(15)
    )


# In[65]:


## Celda 23 — Comparación de las redes por época

def summarize_epoch_networks(networks, period_labels):
    """
    Resume las relaciones de coocurrencia por período histórico.
    """

    summaries = {}

    for source, network in networks.items():

        edges = network["edges_by_epoch"]
        epoch_results = []

        for epoch in period_labels:

            period_edges = edges[
                edges["Epoch"] == epoch
            ]

            if period_edges.empty:
                continue

            # Número de relaciones distintas.
            unique_relationships = len(period_edges)

            # Total de coocurrencias acumuladas.
            total_cooccurrences = period_edges["Weight"].sum()

            # Relación con mayor peso.
            strongest_edge = period_edges.loc[
                period_edges["Weight"].idxmax()
            ]

            epoch_results.append({
                "Epoch": epoch,
                "Unique_Relationships": unique_relationships,
                "Total_Cooccurrences": total_cooccurrences,
                "Strongest_Relationship": (
                    f"{strongest_edge['Source']} — "
                    f"{strongest_edge['Target']}"
                ),
                "Strongest_Weight": strongest_edge["Weight"],
            })

        summary = pd.DataFrame(epoch_results)

        summaries[source] = summary

        print("\n" + "=" * 70)
        print(f"REDES COMPARATIVAS — {source}")
        print("=" * 70)

        display(summary)

        # Mostrar las 10 relaciones más frecuentes de cada época.
        for epoch in period_labels:

            period_edges = edges[
                edges["Epoch"] == epoch
            ]

            if period_edges.empty:
                continue

            print(f"\nPeríodo: {epoch}")

            display(
                period_edges
                .sort_values("Weight", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

    return summaries


epoch_network_summaries = summarize_epoch_networks(
    networks=genre_networks,
    period_labels=period_labels,
)


# In[ ]:


\[
\text{Coocurrencia por 100 canciones} =
\frac{\text{Canciones con ambas etiquetas}}
{\text{Canciones clasificadas en el período}}
\times 100
\]


# In[73]:


## Celda 24 — Normalización de las relaciones por período

def normalize_epoch_cooccurrences(
    networks,
    tag_datasets,
    period_labels,
):
    normalized_results = {}

    for source, network in networks.items():

        edges = network["edges_by_epoch"]
        data = tag_datasets[source]

        # Canciones clasificadas por período.
        denominators = (
            data.groupby("Epoch", observed=True)["Record_ID"]
            .nunique()
            .to_dict()
        )

        period_results = []

        for epoch in period_labels:

            period_edges = edges[
                edges["Epoch"] == epoch
            ].copy()

            classified_songs = denominators.get(epoch, 0)

            if period_edges.empty or classified_songs == 0:
                continue

            period_edges["Classified_Songs"] = classified_songs

            period_edges["Percentage"] = (
                period_edges["Weight"]
                / classified_songs
                * 100
            )

            period_results.append(period_edges)

        if period_results:
            normalized_results[source] = pd.concat(
                period_results,
                ignore_index=True,
            )
        else:
            normalized_results[source] = pd.DataFrame()

    return normalized_results


normalized_genre_networks = normalize_epoch_cooccurrences(
    networks=genre_networks,
    tag_datasets=tag_datasets_clean,
    period_labels=period_labels,
)


for source, results in normalized_genre_networks.items():

    print("\n" + "=" * 70)
    print(f"COOCURRENCIAS NORMALIZADAS — {source}")
    print("=" * 70)

    display(
        results.sort_values(
            ["Epoch", "Percentage"],
            ascending=[True, False]
        )
    )


# In[75]:


## Celda 25 — Matriz de prevalencia por época
discogs_genre_matrix = (
    genre_evolution["Discogs Genre"]
    .pivot(
        index="Tag_Canonical",
        columns="Epoch",
        values="Percentage",
    )
    .reindex(columns=period_labels)
    .fillna(0)
)

display(
    discogs_genre_matrix.round(2)
)


# In[77]:


cdr_genre_matrix = (
    genre_evolution["CDR Genre"]
    .pivot(
        index="Tag_Canonical",
        columns="Epoch",
        values="Percentage",
    )
    .reindex(columns=period_labels)
    .fillna(0)
)

display(cdr_genre_matrix.round(2))


# In[79]:


## Celda 26 — Matrices de prevalencia con cobertura correcta
def build_prevalence_matrix(
    evolution,
    original_tag_data,
    period_labels,
):

    matrix = (
        evolution.pivot(
            index="Tag_Canonical",
            columns="Epoch",
            values="Percentage",
        )
        .reindex(columns=period_labels)
        .fillna(0)
    )

    # Identificar períodos sin ninguna canción clasificada.
    coverage = (
        original_tag_data.groupby(
            "Epoch",
            observed=True
        )["Record_ID"]
        .nunique()
    )

    for epoch in period_labels:
        if coverage.get(epoch, 0) == 0:
            matrix[epoch] = np.nan

    return matrix


genre_matrices = {}

for source in ["CDR Genre", "Discogs Genre"]:

    genre_matrices[source] = build_prevalence_matrix(
        evolution=genre_evolution[source],
        original_tag_data=tag_datasets_clean[source],
        period_labels=period_labels,
    )

    print(f"\n{'=' * 60}")
    print(source)
    print(f"{'=' * 60}")

    display(genre_matrices[source].round(2))


# In[81]:


## Celda 27 — Auditoría de cobertura de las fuentes

total_songs_by_epoch = (
    df.groupby("Epoch", observed=True)
    .size()
    .reindex(period_labels, fill_value=0)
)

coverage_records = []

for source in ["CDR Genre", "Discogs Genre"]:

    source_data = tag_datasets_clean[source]

    classified = (
        source_data.groupby("Epoch", observed=True)["Record_ID"]
        .nunique()
        .reindex(period_labels, fill_value=0)
    )

    for epoch in period_labels:

        total_songs = total_songs_by_epoch.loc[epoch]
        classified_songs = classified.loc[epoch]

        coverage_records.append({
            "Source": source,
            "Epoch": epoch,
            "Total_Songs": total_songs,
            "Classified_Songs": classified_songs,
            "Missing_Classification": (
                total_songs - classified_songs
            ),
            "Coverage_Percentage": (
                classified_songs / total_songs * 100
                if total_songs > 0 else np.nan
            ),
        })

genre_coverage = pd.DataFrame(coverage_records)

display(
    genre_coverage.round(2)
)


# In[83]:


# Celda 27 — Evolución temporal de los géneros

import matplotlib.pyplot as plt
import numpy as np

# Géneros seleccionados para la comparación.
discogs_genres = [
    "Pop",
    "Rock",
    "Funk/Soul",
    "Electronic",
    "Hip Hop",
]

cdr_genres = [
    "Pop",
    "Rock",
    "Funk/Soul",
    "Electronic/Dance",
    "Hip Hop",
]

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(17, 6),
    sharey=True
)

plot_configs = [
    {
        "ax": axes[0],
        "source": "Discogs Genre",
        "genres": discogs_genres,
        "title": "Discogs Genre",
    },
    {
        "ax": axes[1],
        "source": "CDR Genre",
        "genres": cdr_genres,
        "title": "CDR Genre",
    },
]

for config in plot_configs:
    ax = config["ax"]
    matrix = genre_matrices[config["source"]]

    for genre in config["genres"]:
        if genre not in matrix.index:
            continue

        values = matrix.loc[genre].reindex(period_labels)

        ax.plot(
            period_labels,
            values,
            marker="o",
            linewidth=2,
            markersize=5,
            label=genre
        )

    ax.set_title(config["title"], fontsize=14, fontweight="bold")
    ax.set_xlabel("Historical period")
    ax.set_ylabel("Classified songs (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Genre", frameon=False)

fig.suptitle(
    "Evolution of genre prevalence among Billboard Number One songs",
    fontsize=16,
    fontweight="bold"
)

fig.text(
    0.5,
    0.01,
    "Note: Percentages use classified songs as denominator. "
    "Multiple genres per song are allowed. "
    "CDR has no classification coverage for 2020–2025.",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.06, 1, 0.93])
plt.show()


# In[85]:


## Celda 28 — Red global de coocurrencias
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Recuperar las relaciones globales de Discogs.
discogs_edges = genre_networks["Discogs Genre"]["edges"]

# Recuperar las frecuencias de cada género desde el dataset limpio.
discogs_tags = tag_datasets_clean["Discogs Genre"]

discogs_node_counts = (
    discogs_tags
    .groupby("Tag_Canonical")["Record_ID"]
    .nunique()
    .to_dict()
)

# Construir el grafo.
G = nx.Graph()

for genre, count in discogs_node_counts.items():
    G.add_node(genre, frequency=count)

for _, edge in discogs_edges.iterrows():
    G.add_edge(
        edge["Source"],
        edge["Target"],
        weight=edge["Weight"]
    )

# Calcular posiciones una sola vez.
# Estas posiciones se reutilizarán en las redes por período.
pos = nx.spring_layout(
    G,
    seed=42,
    weight="weight",
    k=1.2,
    iterations=200
)

# Tamaño de nodos según frecuencia.
node_sizes = [
    250 + G.nodes[node]["frequency"] * 2
    for node in G.nodes()
]

# Grosor de aristas según coocurrencia.
max_weight = max(
    (data["weight"] for _, _, data in G.edges(data=True)),
    default=1
)

edge_widths = [
    0.5 + 5 * (data["weight"] / max_weight)
    for _, _, data in G.edges(data=True)
]

plt.figure(figsize=(15, 11))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color="#80B1D3",
    edgecolors="#263238",
    linewidths=1,
    alpha=0.9
)

nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths,
    edge_color="#78909C",
    alpha=0.55
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=9,
    font_weight="bold"
)

plt.title(
    "Global genre co-occurrence network — Discogs",
    fontsize=16,
    fontweight="bold"
)

plt.figtext(
    0.5,
    0.02,
    "Node size represents genre frequency. "
    "Edge width represents the number of songs sharing both genres.",
    ha="center",
    fontsize=9
)

plt.axis("off")
plt.tight_layout()
plt.show()


# In[87]:


## Celda 29 — Redes comparativas por período

from matplotlib.lines import Line2D

# Relaciones por período.
discogs_edges_by_epoch = (
    genre_networks["Discogs Genre"]["edges_by_epoch"]
)

# Frecuencia de cada género dentro de cada período.
discogs_epoch_tags = tag_datasets_clean["Discogs Genre"]

epoch_node_counts = (
    discogs_epoch_tags
    .groupby(["Epoch", "Tag_Canonical"], observed=True)["Record_ID"]
    .nunique()
    .to_dict()
)

# Escala común para los grosores de las aristas.
global_max_weight = discogs_edges_by_epoch["Weight"].max()

fig, axes = plt.subplots(
    nrows=4,
    ncols=2,
    figsize=(18, 23)
)

axes = axes.flatten()

for i, epoch in enumerate(period_labels):
    ax = axes[i]

    epoch_edges = discogs_edges_by_epoch[
        discogs_edges_by_epoch["Epoch"] == epoch
    ]

    # Construir la red del período.
    G_epoch = nx.Graph()

    # Añadir géneros presentes en ese período.
    for genre in discogs_node_counts:
        count = epoch_node_counts.get((epoch, genre), 0)

        if count > 0:
            G_epoch.add_node(genre, frequency=count)

    # Añadir relaciones observadas.
    for _, edge in epoch_edges.iterrows():
        G_epoch.add_edge(
            edge["Source"],
            edge["Target"],
            weight=edge["Weight"]
        )

    # Solo dibujar los nodos que aparecen en el período.
    active_nodes = list(G_epoch.nodes())

    node_sizes_epoch = [
        100 + G_epoch.nodes[node]["frequency"] * 5
        for node in active_nodes
    ]

    edge_widths_epoch = [
        0.5 + 5 * (data["weight"] / global_max_weight)
        for _, _, data in G_epoch.edges(data=True)
    ]

    # Reutilizar las posiciones de la red global.
    nx.draw_networkx_nodes(
        G_epoch,
        pos,
        nodelist=active_nodes,
        node_size=node_sizes_epoch,
        node_color="#80B1D3",
        edgecolors="#263238",
        linewidths=0.8,
        ax=ax
    )

    nx.draw_networkx_edges(
        G_epoch,
        pos,
        width=edge_widths_epoch,
        edge_color="#78909C",
        alpha=0.6,
        ax=ax
    )

    nx.draw_networkx_labels(
        G_epoch,
        pos,
        font_size=8,
        ax=ax
    )

    ax.set_title(
        f"{epoch}\n"
        f"{G_epoch.number_of_nodes()} genres | "
        f"{G_epoch.number_of_edges()} relationships",
        fontsize=11,
        fontweight="bold"
    )

    ax.axis("off")

# Eliminar el panel sobrante.
axes[-1].axis("off")

fig.suptitle(
    "Evolution of genre co-occurrence networks — Discogs",
    fontsize=17,
    fontweight="bold",
    y=0.99
)

fig.text(
    0.5,
    0.015,
    "Node positions are fixed using the global network. "
    "Node size represents genre frequency within each period. "
    "Edge width represents shared-song frequency using a common scale.",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.show()


# In[101]:


## Celda 30 — Indicadores explicativos de la red global

# Número de géneros diferentes conectados a cada nodo.
degree = dict(G.degree())

# Suma de los pesos de todas las conexiones de cada nodo.
strength = dict(G.degree(weight="weight"))

# Frecuencia individual de cada género.
genre_frequency = nx.get_node_attributes(G, "frequency")

# Porcentaje de canciones clasificadas que contienen cada género.
total_classified_songs = (
    discogs_tags["Record_ID"].nunique()
)

node_explainability = pd.DataFrame({
    "Genre": list(G.nodes()),
    "Song_Count": [
        genre_frequency[node] for node in G.nodes()
    ],
    "Song_Prevalence": [
        genre_frequency[node] / total_classified_songs * 100
        for node in G.nodes()
    ],
    "Connected_Genres": [
        degree[node] for node in G.nodes()
    ],
    "Cooccurrence_Strength": [
        strength[node] for node in G.nodes()
    ],
})

node_explainability = (
    node_explainability
    .sort_values(
        ["Cooccurrence_Strength", "Song_Count"],
        ascending=False
    )
    .reset_index(drop=True)
)

display(
    node_explainability.round(2)
)


# In[ ]:





# In[103]:


## Celda 31 — Ranking de coocurrencias globales

edge_explainability = discogs_edges.copy()

edge_explainability["Classified_Songs"] = total_classified_songs

edge_explainability["Cooccurrence_Percentage"] = (
    edge_explainability["Weight"]
    / edge_explainability["Classified_Songs"]
    * 100
)

edge_explainability = (
    edge_explainability
    .rename(columns={
        "Source": "Genre_A",
        "Target": "Genre_B",
        "Weight": "Shared_Songs",
    })
    [[
        "Genre_A",
        "Genre_B",
        "Shared_Songs",
        "Classified_Songs",
        "Cooccurrence_Percentage",
    ]]
    .sort_values(
        "Shared_Songs",
        ascending=False
    )
    .reset_index(drop=True)
)

display(
    edge_explainability.head(15).round(2)
)


# In[ ]:





# In[ ]:





# In[105]:


## Celda 32 — Evolución temporal de las coocurrencias

# Número de canciones clasificadas por período.
classified_by_epoch = (
    discogs_tags
    .groupby("Epoch", observed=True)["Record_ID"]
    .nunique()
)

temporal_edges = discogs_edges_by_epoch.copy()

temporal_edges["Classified_Songs"] = (
    temporal_edges["Epoch"]
    .map(classified_by_epoch.to_dict())
    .astype("int64")
)

temporal_edges["Cooccurrence_Percentage"] = (
    temporal_edges["Weight"]
    / temporal_edges["Classified_Songs"]
    * 100
)

temporal_edges["Genre_Pair"] = (
    temporal_edges["Source"]
    + " + "
    + temporal_edges["Target"]
)

temporal_edges = temporal_edges.sort_values(
    ["Epoch", "Weight"],
    ascending=[True, False]
)

display(
    temporal_edges[
        [
            "Epoch",
            "Genre_Pair",
            "Weight",
            "Classified_Songs",
            "Cooccurrence_Percentage",
        ]
    ].round(2)
)


# In[107]:


## Celda 33 — Asociación dominante por período

dominant_edges = (
    temporal_edges
    .sort_values(
        ["Epoch", "Weight"],
        ascending=[True, False]
    )
    .groupby("Epoch", observed=True)
    .head(1)
    [[
        "Epoch",
        "Genre_Pair",
        "Weight",
        "Classified_Songs",
        "Cooccurrence_Percentage",
    ]]
    .reset_index(drop=True)
)

display(dominant_edges.round(2))


# In[109]:


## Celda 34 — Red global interactiva con PyVis

from pyvis.network import Network
from IPython.display import HTML, display

net = Network(
    height="750px",
    width="100%",
    bgcolor="#FFFFFF",
    font_color="#263238",
    directed=False,
    notebook=True
)

# Desactivar la física para conservar las posiciones
# calculadas previamente con NetworkX.
net.toggle_physics(False)

# Agregar nodos con información explicativa.
for node, attributes in G.nodes(data=True):

    frequency = attributes["frequency"]
    prevalence = frequency / total_classified_songs * 100

    net.add_node(
        node,
        label=node,
        title=(
            f"<b>{node}</b><br>"
            f"Songs: {frequency}<br>"
            f"Prevalence: {prevalence:.2f}%<br>"
            f"Connected genres: {degree[node]}<br>"
            f"Co-occurrence strength: {strength[node]}"
        ),
        size=15 + frequency * 0.12,
        color="#80B1D3",
        x=float(pos[node][0] * 1000),
        y=float(pos[node][1] * 1000),
        physics=False
    )

# Agregar conexiones con información explicativa.
for source, target, attributes in G.edges(data=True):

    weight = attributes["weight"]
    percentage = weight / total_classified_songs * 100

    net.add_edge(
        source,
        target,
        value=weight,
        title=(
            f"<b>{source} + {target}</b><br>"
            f"Shared songs: {weight}<br>"
            f"Prevalence: {percentage:.2f}%"
        ),
        color="#90A4AE"
    )

# Mostrar la red.
display(HTML(net.generate_html()))


# In[111]:


## Celda 35 — Métricas de asociación: Jaccard y Lift

# Universo de canciones clasificadas por Discogs.
# Cada canción debe contar una sola vez por género.
genre_records = (
    discogs_tags[
        ["Record_ID", "Tag_Canonical"]
    ]
    .dropna()
    .drop_duplicates()
)

total_songs = genre_records["Record_ID"].nunique()

# Frecuencia individual de cada género.
genre_counts = (
    genre_records
    .groupby("Tag_Canonical")["Record_ID"]
    .nunique()
)

# Recuperar las coocurrencias globales.
association_metrics = discogs_edges.copy()

# Incorporar frecuencias individuales de cada género.
association_metrics["Genre_A_Count"] = (
    association_metrics["Source"].map(genre_counts)
)

association_metrics["Genre_B_Count"] = (
    association_metrics["Target"].map(genre_counts)
)

association_metrics["Classified_Songs"] = total_songs

# Probabilidades individuales.
association_metrics["Genre_A_Prevalence"] = (
    association_metrics["Genre_A_Count"] / total_songs
)

association_metrics["Genre_B_Prevalence"] = (
    association_metrics["Genre_B_Count"] / total_songs
)

# Probabilidad observada de coocurrencia.
association_metrics["Joint_Prevalence"] = (
    association_metrics["Weight"] / total_songs
)

# Jaccard: intersección / unión.
association_metrics["Jaccard"] = (
    association_metrics["Weight"]
    / (
        association_metrics["Genre_A_Count"]
        + association_metrics["Genre_B_Count"]
        - association_metrics["Weight"]
    )
)

# Lift: probabilidad observada / probabilidad esperada.
association_metrics["Expected_Shared_Songs"] = (
    association_metrics["Genre_A_Count"]
    * association_metrics["Genre_B_Count"]
    / total_songs
)

association_metrics["Lift"] = (
    association_metrics["Weight"]
    / association_metrics["Expected_Shared_Songs"]
)

# Clasificación descriptiva de la asociación.
association_metrics["Association_Type"] = np.select(
    [
        association_metrics["Lift"] > 1,
        association_metrics["Lift"] < 1,
    ],
    [
        "Positive association",
        "Negative association",
    ],
    default="Approximately independent"
)

# Preparar tabla final.
association_metrics = (
    association_metrics
    .rename(columns={
        "Source": "Genre_A",
        "Target": "Genre_B",
        "Weight": "Shared_Songs",
    })
    [[
        "Genre_A",
        "Genre_B",
        "Shared_Songs",
        "Genre_A_Count",
        "Genre_B_Count",
        "Jaccard",
        "Expected_Shared_Songs",
        "Lift",
        "Association_Type",
    ]]
    .sort_values("Lift", ascending=False)
    .reset_index(drop=True)
)

display(association_metrics.round({
    "Jaccard": 4,
    "Expected_Shared_Songs": 2,
    "Lift": 3,
}))


# In[119]:


# Minimum support required for exploratory interpretation
MIN_SHARED_SONGS = 5

supported_associations = association_metrics.loc[
    association_metrics["Shared_Songs"] >= MIN_SHARED_SONGS
].copy()

# More neutral interpretation of Lift
supported_associations["Association_Type"] = np.select(
    [
        supported_associations["Lift"] > 1.001,
        supported_associations["Lift"] < 0.999,
    ],
    [
        "Above independence expectation",
        "Below independence expectation",
    ],
    default="Approximately independent",
)

# Associations appearing more frequently than expected
top_positive_associations = (
    supported_associations.loc[
        supported_associations["Lift"] > 1
    ]
    .sort_values("Lift", ascending=False)
    .head(15)
)

# Associations appearing less frequently than expected
top_negative_associations = (
    supported_associations.loc[
        supported_associations["Lift"] < 1
    ]
    .sort_values("Lift", ascending=True)
    .head(15)
)

# Most recurrent combinations, regardless of Lift
top_jaccard_associations = (
    supported_associations
    .sort_values("Jaccard", ascending=False)
    .head(15)
)

display(
    top_jaccard_associations[
        [
            "Genre_A",
            "Genre_B",
            "Shared_Songs",
            "Jaccard",
            "Lift",
            "Association_Type",
        ]
    ].round(3)
)


# In[121]:


# Inspect temporal network structures

for source, network in genre_networks.items():
    print(f"\n{'=' * 60}")
    print(f"SOURCE: {source}")
    print("=" * 60)

    print("\nNodes by epoch:")
    print(network["nodes_by_epoch"].head())
    print("\nColumns:", network["nodes_by_epoch"].columns.tolist())

    print("\nEdges by epoch:")
    print(network["edges_by_epoch"].head())
    print("\nColumns:", network["edges_by_epoch"].columns.tolist())


# In[123]:


## Celda 37 — Frecuencia de géneros por período
# Temporal genre node frequencies

nodes_by_epoch = {}

for source in ["CDR Genre", "Discogs Genre"]:

    tags = tag_datasets_clean[source].copy()

    tags = tags[
        ["Record_ID", "Epoch", "Tag_Canonical"]
    ].dropna()

    tags = tags.drop_duplicates(
        subset=["Record_ID", "Epoch", "Tag_Canonical"]
    )

    temporal_nodes = (
        tags.groupby(
            ["Epoch", "Tag_Canonical"],
            observed=True
        )["Record_ID"]
        .nunique()
        .reset_index(name="Song_Count")
        .rename(columns={"Tag_Canonical": "Tag"})
    )

    nodes_by_epoch[source] = temporal_nodes

    print(f"\n{'=' * 60}")
    print(f"SOURCE: {source}")
    print("=" * 60)

    display(
        temporal_nodes.sort_values(
            ["Epoch", "Song_Count"],
            ascending=[True, False]
        ).head(15)
    )


# In[125]:


## Celda 38 — Métricas temporales de asociación

def calculate_temporal_associations(
    tag_data,
    edges_by_epoch
):
    records = []

    for epoch, epoch_edges in edges_by_epoch.groupby(
        "Epoch",
        observed=True
    ):

        epoch_tags = tag_data.loc[
            tag_data["Epoch"] == epoch,
            ["Record_ID", "Tag_Canonical"]
        ].dropna().drop_duplicates()

        total_songs = epoch_tags["Record_ID"].nunique()

        genre_counts = (
            epoch_tags.groupby("Tag_Canonical")["Record_ID"]
            .nunique()
        )

        for _, edge in epoch_edges.iterrows():

            genre_a = edge["Source"]
            genre_b = edge["Target"]
            shared = int(edge["Weight"])

            count_a = int(genre_counts.get(genre_a, 0))
            count_b = int(genre_counts.get(genre_b, 0))

            union = count_a + count_b - shared

            expected = (
                count_a * count_b / total_songs
                if total_songs > 0 else np.nan
            )

            jaccard = shared / union if union > 0 else np.nan

            lift = (
                shared / expected
                if expected > 0 else np.nan
            )

            records.append({
                "Epoch": epoch,
                "Genre_A": genre_a,
                "Genre_B": genre_b,
                "Shared_Songs": shared,
                "Genre_A_Count": count_a,
                "Genre_B_Count": count_b,
                "Classified_Songs": total_songs,
                "Jaccard": jaccard,
                "Lift": lift,
            })

    return pd.DataFrame(records)


# In[127]:


discogs_temporal_associations = calculate_temporal_associations(
    tag_datasets_clean["Discogs Genre"],
    genre_networks["Discogs Genre"]["edges_by_epoch"]
)

display(
    discogs_temporal_associations.sort_values(
        ["Epoch", "Shared_Songs"],
        ascending=[True, False]
    ).head(20)
)


# In[129]:


cdr_temporal_associations = calculate_temporal_associations(
    tag_datasets_clean["CDR Genre"],
    genre_networks["CDR Genre"]["edges_by_epoch"]
)

display(
    cdr_temporal_associations.sort_values(
        ["Epoch", "Shared_Songs"],
        ascending=[True, False]
    ).head(20)
)


# In[131]:


# Validate temporal association metrics

for source, associations in [
    ("Discogs Genre", discogs_temporal_associations),
    ("CDR Genre", cdr_temporal_associations),
]:

    if associations.empty:
        print(f"{source}: no temporal associations found.")
        continue

    assert (
        associations["Shared_Songs"]
        <= associations[["Genre_A_Count", "Genre_B_Count"]].min(axis=1)
    ).all()

    assert associations["Jaccard"].between(0, 1).all()

    assert (associations["Lift"] > 0).all()

    print(f"{source}: all validations passed.")


# In[133]:


## Celda 39 — Preparar posiciones globales de los nodos
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets

from IPython.display import display, clear_output

# Select the main taxonomy
SOURCE = "Discogs Genre"

temporal_nodes = nodes_by_epoch[SOURCE].copy()
temporal_edges = genre_networks[SOURCE]["edges_by_epoch"].copy()
temporal_metrics = discogs_temporal_associations.copy()

# Build aggregate graph to calculate fixed node positions
G = nx.Graph()

global_nodes = genre_networks[SOURCE]["nodes"]

for _, row in global_nodes.iterrows():
    G.add_node(row["Tag"])

for _, row in genre_networks[SOURCE]["edges"].iterrows():
    G.add_edge(
        row["Source"],
        row["Target"],
        weight=row["Weight"]
    )

# Calculate positions only once
pos = nx.spring_layout(
    G,
    seed=42,
    weight="weight",
    k=1.2,
    iterations=200
)

# Stable genre colors
genres = sorted(G.nodes())

palette = px.colors.qualitative.Plotly

genre_colors = {
    genre: palette[i % len(palette)]
    for i, genre in enumerate(genres)
}

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print("Fixed node positions calculated.")


# In[135]:


## Celda 40 — Función para construir la red interactiva

def plot_genre_network(
    epoch,
    metric="Shared_Songs",
    min_support=2
):
    # Filter nodes for selected period
    period_nodes = temporal_nodes.loc[
        temporal_nodes["Epoch"] == epoch
    ].copy()

    if period_nodes.empty:
        print(f"No classified genres available for {epoch}.")
        return

    # Filter associations for selected period
    period_edges = temporal_metrics.loc[
        temporal_metrics["Epoch"] == epoch
    ].copy()

    period_edges = period_edges.loc[
        period_edges["Shared_Songs"] >= min_support
    ].copy()

    # Graph traces
    edge_traces = []

    metric_labels = {
        "Shared_Songs": "Shared songs",
        "Jaccard": "Jaccard",
        "Lift": "Lift"
    }

    for _, edge in period_edges.iterrows():

        source = edge["Genre_A"]
        target = edge["Genre_B"]

        x0, y0 = pos[source]
        x1, y1 = pos[target]

        value = edge[metric]

        if metric == "Shared_Songs":
            width = max(1, min(8, value / 5))
        elif metric == "Jaccard":
            width = max(1, min(8, value * 20))
        else:
            width = max(
                1,
                min(8, abs(np.log2(value)) * 2)
            )

        association = (
            "Above independence expectation"
            if edge["Lift"] > 1
            else "Below independence expectation"
            if edge["Lift"] < 1
            else "Approximately independent"
        )

        hover_text = (
            f"<b>{source} + {target}</b><br>"
            f"Shared songs: {edge['Shared_Songs']}<br>"
            f"Jaccard: {edge['Jaccard']:.3f}<br>"
            f"Lift: {edge['Lift']:.3f}<br>"
            f"Association: {association}"
        )

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=width,
                    color="#8492A6"
                ),
                hoverinfo="text",
                text=[hover_text, hover_text],
                showlegend=False
            )
        )

    # Node information
    node_x = []
    node_y = []
    node_sizes = []
    node_colors = []
    node_text = []

    for _, node in period_nodes.iterrows():

        genre = node["Tag"]
        count = node["Song_Count"]

        x, y = pos[genre]

        node_x.append(x)
        node_y.append(y)
        node_sizes.append(15 + count * 0.12)
        node_colors.append(genre_colors[genre])

        node_text.append(
            f"<b>{genre}</b><br>"
            f"Songs: {count}<br>"
            f"Period: {epoch}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=period_nodes["Tag"],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="white"),
            opacity=0.9
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=edge_traces + [node_trace]
    )

    fig.update_layout(
        title=(
            f"Discogs Genre Co-occurrence Network — {epoch}"
            f"<br><sup>Edge metric: {metric_labels[metric]}</sup>"
        ),
        template="plotly_white",
        height=750,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )

    fig.show()


# In[145]:


## Celda 41 — Controles interactivos
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

MIN_SUPPORT = 2

epochs = [
    "1958-1969",
    "1970s",
    "1980s",
    "1990s",
    "2000s",
    "2010s",
    "2020-2025"
]

metrics = {
    "Shared songs": "Shared_Songs",
    "Jaccard": "Jaccard",
    "Lift": "Lift"
}

fig = go.Figure()
visibility_groups = {}
combo_titles = []

# Generate one group of traces per period and metric
for epoch in epochs:

    for metric_label, metric_column in metrics.items():

        period_edges = temporal_metrics.loc[
            (temporal_metrics["Epoch"] == epoch) &
            (temporal_metrics["Shared_Songs"] >= MIN_SUPPORT)
        ].copy()

        period_nodes = temporal_nodes.loc[
            temporal_nodes["Epoch"] == epoch
        ].copy()

        start_index = len(fig.data)

        # If there are no classified songs, create an informative placeholder
        if period_nodes.empty:

            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="text",
                    text=["No classified songs in this period"],
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        else:

            # Individual edges
            for _, edge in period_edges.iterrows():

                source = edge["Genre_A"]
                target = edge["Genre_B"]

                x0, y0 = pos[source]
                x1, y1 = pos[target]

                value = edge[metric_column]

                if metric_column == "Shared_Songs":
                    width = max(1, min(8, value / 5))
                elif metric_column == "Jaccard":
                    width = max(1, min(8, value * 20))
                else:
                    width = max(
                        1,
                        min(8, abs(np.log2(value)) * 2)
                    )

                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(
                            width=width,
                            color="#8492A6"
                        ),
                        hoverinfo="text",
                        text=[
                            f"{source} + {target}<br>"
                            f"Shared songs: {edge['Shared_Songs']}<br>"
                            f"Jaccard: {edge['Jaccard']:.3f}<br>"
                            f"Lift: {edge['Lift']:.3f}",
                            ""
                        ],
                        showlegend=False
                    )
                )

            # Nodes
            node_x, node_y = [], []
            node_sizes, node_colors, node_hover = [], [], []

            for _, node in period_nodes.iterrows():

                genre = node["Tag"]
                count = node["Song_Count"]

                x, y = pos[genre]

                node_x.append(x)
                node_y.append(y)
                node_sizes.append(15 + count * 0.12)
                node_colors.append(genre_colors[genre])

                node_hover.append(
                    f"<b>{genre}</b><br>"
                    f"Songs: {count}<br>"
                    f"Period: {epoch}"
                )

            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=period_nodes["Tag"],
                    textposition="top center",
                    hovertext=node_hover,
                    hoverinfo="text",
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        line=dict(width=1, color="white"),
                        opacity=0.9
                    ),
                    showlegend=False
                )
            )

        end_index = len(fig.data)

        combo = f"{epoch} — {metric_label}"
        combo_titles.append(combo)

        visibility_groups[combo] = [
            i >= start_index and i < end_index
            for i in range(len(fig.data))
        ]

# Show first combination initially
first_combo = combo_titles[0]

for i, visible in enumerate(visibility_groups[first_combo]):
    fig.data[i].visible = visible

# Native Plotly dropdown
buttons = []

for combo in combo_titles:

    buttons.append(
        dict(
            label=combo,
            method="update",
            args=[
                {"visible": visibility_groups[combo]},
                {
                    "title": (
                        f"Discogs Genre Network — {combo}"
                        f"<br><sup>Minimum support: {MIN_SUPPORT} songs</sup>"
                    )
                }
            ]
        )
    )

fig.update_layout(
    title=f"Discogs Genre Network — {first_combo}",
    template="plotly_white",
    height=800,
    hovermode="closest",
    showlegend=False,
    margin=dict(l=30, r=30, t=120, b=30),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0,
            y=1.12,
            xanchor="left",
            yanchor="top"
        )
    ],
    annotations=[
        dict(
            text="Select period and metric:",
            x=0,
            y=1.19,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left"
        )
    ]
)

# Export standalone HTML
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "genre_network_explorer.html"

fig.write_html(
    output_path,
    include_plotlyjs=True,
    full_html=True
)

print(f"HTML generated: {output_path.resolve()}")
print(f"Available views: {len(combo_titles)}")


# In[147]:


# Cell 42 — Prepare data for PERMDISP

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

RANDOM_SEED = 42
N_PERMUTATIONS = 9999
ALPHA = 0.05

# Use the previously prepared analytical dataset
permdisp_data = audio_complete[
    ["Song", "Artist", "Date", "Epoch"] + AUDIO_FEATURES
].copy()

# Remove observations without a valid period
permdisp_data = permdisp_data.dropna(
    subset=["Epoch"] + AUDIO_FEATURES
).copy()

# Preserve chronological order
permdisp_data["Epoch"] = pd.Categorical(
    permdisp_data["Epoch"],
    categories=period_labels,
    ordered=True
)

# Standardize the six audio features globally
X = permdisp_data[AUDIO_FEATURES].to_numpy(dtype=float)

feature_means = X.mean(axis=0)
feature_stds = X.std(axis=0, ddof=1)

if np.any(feature_stds == 0):
    raise ValueError("A constant audio feature was detected.")

X_standardized = (X - feature_means) / feature_stds

# Group labels
group_labels = permdisp_data["Epoch"].astype(str).to_numpy()

# Check sample size and group distribution
epoch_counts = (
    permdisp_data.groupby("Epoch", observed=True)
    .size()
    .rename("Song_Count")
    .reset_index()
)

print("Observations:", len(permdisp_data))
print("Audio features:", len(AUDIO_FEATURES))
print("Historical periods:", permdisp_data["Epoch"].nunique())

display(epoch_counts)


# In[151]:


## Celda 43 — PERMDISP permutation test

def calculate_permdisp_distances(X, groups):
    distances = np.zeros(len(X), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        group_data = X[mask]
        centroid = group_data.mean(axis=0)
        distances[mask] = np.linalg.norm(group_data - centroid, axis=1)
    return distances

def calculate_permdisp_statistic(X, groups):
    distances = calculate_permdisp_distances(X, groups)
    group_distances = [distances[groups == group] for group in np.unique(groups)]
    statistic = f_oneway(*group_distances).statistic
    return float(statistic), distances

# Observed statistic
observed_f, observed_distances = calculate_permdisp_statistic(X_standardized, group_labels)

# Permutation test
rng = np.random.default_rng(RANDOM_SEED)
permuted_statistics = np.empty(N_PERMUTATIONS, dtype=float)

for i in range(N_PERMUTATIONS):
    permuted_groups = rng.permutation(group_labels)
    permuted_f, _ = calculate_permdisp_statistic(X_standardized, permuted_groups)
    permuted_statistics[i] = permuted_f

# Monte Carlo permutation p-value
p_value = (1 + np.sum(permuted_statistics >= observed_f)) / (N_PERMUTATIONS + 1)

# Effect size: eta squared on distances to centroids (corrigiendo 'groups' por 'group_labels' y usando conteo n_i)
grand_mean = observed_distances.mean()
ss_total = np.sum((observed_distances - grand_mean) ** 2)

ss_between = sum(
    np.sum(group_labels == group) * (observed_distances[group_labels == group].mean() - grand_mean) ** 2
    for group in np.unique(group_labels)
)

eta_squared = ss_between / ss_total
reject_null = p_value < ALPHA

permdisp_results = pd.Series({
    "Test": "PERMDISP",
    "Test statistic (F)": observed_f,
    "P-value": p_value,
    "Permutations": N_PERMUTATIONS,
    "Alpha": ALPHA,
    "Eta squared": eta_squared,
    "Decision": "Reject H0" if reject_null else "Fail to reject H0"
})

display(permdisp_results)


# In[153]:


# Cell 44 — Descriptive dispersion by historical period

permdisp_data["Distance_To_Centroid"] = observed_distances

dispersion_summary = (
    permdisp_data.groupby("Epoch", observed=True)
    ["Distance_To_Centroid"]
    .agg(
        Song_Count="count",
        Mean_Distance="mean",
        Median_Distance="median",
        Standard_Deviation="std",
        Minimum="min",
        Maximum="max"
    )
    .reset_index()
)

display(
    dispersion_summary.sort_values(
        "Mean_Distance",
        ascending=False
    )
)


# In[155]:


## Celda 45 — Pruebas post hoc por pares


from itertools import combinations
from scipy.stats import f_oneway
import numpy as np
import pandas as pd

# Configuración
N_PERMUTATIONS_PAIRWISE = 9999
RANDOM_SEED_PAIRWISE = 42
ALPHA = 0.05

def pairwise_permdisp(X_pair, labels_pair):
    """
    PERMDISP basado en centroides:
    compara las distancias euclidianas al centroide de dos grupos.
    """
    groups = np.unique(labels_pair)

    if len(groups) != 2:
        raise ValueError("Esta función requiere exactamente dos grupos.")

    def calculate_distances(X, labels):
        distances = np.zeros(len(X), dtype=float)

        for group in groups:
            mask = labels == group
            centroid = X[mask].mean(axis=0)
            distances[mask] = np.linalg.norm(
                X[mask] - centroid, axis=1
            )

        return distances

    def calculate_f(X, labels):
        distances = calculate_distances(X, labels)

        group_distances = [
            distances[labels == group] for group in groups
        ]

        return float(f_oneway(*group_distances).statistic), distances

    observed_f, observed_distances = calculate_f(X_pair, labels_pair)

    rng = np.random.default_rng(RANDOM_SEED_PAIRWISE)
    permuted_statistics = np.empty(N_PERMUTATIONS_PAIRWISE)

    for i in range(N_PERMUTATIONS_PAIRWISE):
        permuted_labels = rng.permutation(labels_pair)
        permuted_f, _ = calculate_f(X_pair, permuted_labels)
        permuted_statistics[i] = permuted_f

    p_value = (
        1 + np.sum(permuted_statistics >= observed_f)
    ) / (N_PERMUTATIONS_PAIRWISE + 1)

    # Diferencia de distancias promedio: B - A
    mean_a = observed_distances[labels_pair == groups[0]].mean()
    mean_b = observed_distances[labels_pair == groups[1]].mean()

    mean_difference = mean_b - mean_a

    # Tamaño de efecto eta cuadrado para la comparación de dos grupos
    grand_mean = observed_distances.mean()
    ss_total = np.sum((observed_distances - grand_mean) ** 2)

    ss_between = sum(
        np.sum(labels_pair == group)
        * (
            observed_distances[labels_pair == group].mean()
            - grand_mean
        ) ** 2
        for group in groups
    )

    eta_squared = (
        ss_between / ss_total if ss_total > 0 else np.nan
    )

    return {
        "F_Statistic": observed_f,
        "P_Value": p_value,
        "Mean_Distance_A": mean_a,
        "Mean_Distance_B": mean_b,
        "Mean_Difference_B_minus_A": mean_difference,
        "Eta_Squared": eta_squared,
    }


# Ejecutar las 21 comparaciones
pairwise_results = []

for epoch_a, epoch_b in combinations(period_labels, 2):

    mask = np.isin(group_labels, [epoch_a, epoch_b])

    X_pair = X_standardized[mask]
    labels_pair = group_labels[mask]

    result = pairwise_permdisp(X_pair, labels_pair)

    pairwise_results.append({
        "Epoch_A": epoch_a,
        "Epoch_B": epoch_b,
        **result,
    })

pairwise_results = pd.DataFrame(pairwise_results)

print(f"Comparaciones ejecutadas: {len(pairwise_results)}")
display(pairwise_results.round(5))


# In[156]:


# Celda 46 — Corrección de Holm

from statsmodels.stats.multitest import multipletests

reject_holm, p_adjusted, _, _ = multipletests(
    pairwise_results["P_Value"],
    alpha=ALPHA,
    method="holm"
)

pairwise_results["P_Value_Holm"] = p_adjusted
pairwise_results["Significant_Holm"] = reject_holm

pairwise_results["Conclusion"] = np.where(
    reject_holm,
    "Significant difference",
    "No significant difference"
)

pairwise_results = pairwise_results.sort_values(
    "P_Value_Holm"
).reset_index(drop=True)

print("Comparaciones significativas después de Holm:")
print(
    pairwise_results["Significant_Holm"].sum(),
    "de",
    len(pairwise_results)
)

display(
    pairwise_results[
        [
            "Epoch_A",
            "Epoch_B",
            "F_Statistic",
            "P_Value",
            "P_Value_Holm",
            "Mean_Difference_B_minus_A",
            "Eta_Squared",
            "Conclusion",
        ]
    ].round(5)
)


# In[167]:


import matplotlib.pyplot as plt
import numpy as np

# Matriz: 1 = diferencia significativa; 0 = no significativa
n_periods = len(period_labels)

significance_matrix = np.full(
    (n_periods, n_periods),
    np.nan
)

for _, row in pairwise_results.iterrows():
    i = period_labels.index(row["Epoch_A"])
    j = period_labels.index(row["Epoch_B"])

    value = int(row["Significant_Holm"])

    significance_matrix[i, j] = value
    significance_matrix[j, i] = value

fig, ax = plt.subplots(figsize=(10, 8))

image = ax.imshow(
    significance_matrix,
    cmap="Greens",
    vmin=0,
    vmax=1
)

ax.set_xticks(range(n_periods))
ax.set_yticks(range(n_periods))
ax.set_xticklabels(period_labels, rotation=35, ha="right")
ax.set_yticklabels(period_labels)

for i in range(n_periods):
    for j in range(n_periods):
        value = significance_matrix[i, j]

        if np.isnan(value):
            continue

        label = "Sí" if value == 1 else "No"

        ax.text(
            j, i, label,
            ha="center",
            va="center",
            color="black"
        )

ax.set_title(
    "Diferencias de dispersión entre períodos\n"
    "Comparaciones por pares con corrección de Holm"
)
ax.set_xlabel("Período B")
ax.set_ylabel("Período A")

colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
colorbar.ax.set_yticklabels(["No significativa", "Significativa"])

plt.tight_layout()
plt.show()


# In[173]:


significance_matrix


# In[157]:


## Celda 48 — Gráfico de diferencias de dispersión

def pairwise_permdisp(X_pair, labels_pair, group_order):
    """
    PERMDISP basado en centroides para dos grupos.
    group_order conserva el orden A y B de la comparación.
    """
    groups = np.asarray(group_order)

    def calculate_distances(X, labels):
        distances = np.zeros(len(X), dtype=float)

        for group in groups:
            mask = labels == group
            centroid = X[mask].mean(axis=0)
            distances[mask] = np.linalg.norm(
                X[mask] - centroid, axis=1
            )

        return distances

    def calculate_f(X, labels):
        distances = calculate_distances(X, labels)
        group_distances = [
            distances[labels == group] for group in groups
        ]
        return float(f_oneway(*group_distances).statistic), distances

    observed_f, observed_distances = calculate_f(X_pair, labels_pair)

    rng = np.random.default_rng(RANDOM_SEED_PAIRWISE)
    permuted_statistics = np.empty(N_PERMUTATIONS_PAIRWISE)

    for i in range(N_PERMUTATIONS_PAIRWISE):
        permuted_labels = rng.permutation(labels_pair)
        permuted_f, _ = calculate_f(X_pair, permuted_labels)
        permuted_statistics[i] = permuted_f

    p_value = (
        1 + np.sum(permuted_statistics >= observed_f)
    ) / (N_PERMUTATIONS_PAIRWISE + 1)

    mean_a = observed_distances[labels_pair == groups[0]].mean()
    mean_b = observed_distances[labels_pair == groups[1]].mean()

    grand_mean = observed_distances.mean()
    ss_total = np.sum((observed_distances - grand_mean) ** 2)

    ss_between = sum(
        np.sum(labels_pair == group)
        * (
            observed_distances[labels_pair == group].mean()
            - grand_mean
        ) ** 2
        for group in groups
    )

    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "F_Statistic": observed_f,
        "P_Value": p_value,
        "Mean_Distance_A": mean_a,
        "Mean_Distance_B": mean_b,
        "Mean_Difference_B_minus_A": mean_b - mean_a,
        "Eta_Squared": eta_squared,
    }


# In[158]:


result = pairwise_permdisp(
    X_pair,
    labels_pair,
    group_order=(epoch_a, epoch_b)
)


# In[165]:


result


# In[159]:


plot_data = pairwise_results.sort_values(
    "Mean_Difference_B_minus_A"
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 9))

colors = np.where(
    plot_data["Significant_Holm"],
    "tab:green",
    "tab:gray"
)

y_positions = np.arange(len(plot_data))

ax.scatter(
    plot_data["Mean_Difference_B_minus_A"],
    y_positions,
    c=colors,
    s=65,
    zorder=3
)

ax.axvline(0, linestyle="--", linewidth=1, color="black")

labels = (
    plot_data["Epoch_A"] + " vs. " + plot_data["Epoch_B"]
)

ax.set_yticks(y_positions)
ax.set_yticklabels(labels)

ax.set_xlabel("Diferencia de distancia promedio (B − A)")
ax.set_title(
    "Diferencias de dispersión entre períodos\n"
    "Verde: significativa después de Holm | Gris: no significativa"
)

ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

Decisión: rechazamos H₀.

para el informe:

### Validación de la hipótesis: evolución de la diversidad sonora

La prueba PERMDISP identificó diferencias estadísticamente significativas en la dispersión multivariada de las características acústicas de las canciones que alcanzaron el número uno del Billboard Hot 100 entre los períodos históricos analizados (F = 8.58; p = 0.0001; η² = 0.0422).

Las comparaciones post hoc, con corrección de Holm para las 21 pruebas realizadas, identificaron ocho diferencias significativas. Todas corresponden a comparaciones entre los períodos comprendidos entre 1958 y 1999 y las décadas de 2000 y 2010. En estas últimas se observaron menores distancias promedio al centroide, lo que indica una mayor concentración de los perfiles acústicos.

No se identificaron diferencias significativas entre las décadas de 2000 y 2010. Asimismo, aunque el período 2020–2025 presentó un incremento descriptivo de la dispersión, este no fue estadísticamente significativo frente a las dos décadas anteriores después de aplicar la corrección por comparaciones múltiples.

En conjunto, los resultados respaldan la existencia de cambios históricos en la diversidad sonora de las canciones número uno. La evidencia apunta a una etapa de mayor homogeneidad acústica durante los años 2000 y 2010, pero no permite afirmar que exista una tendencia continua de homogeneización hasta la actualidad.
# In[ ]:





# In[ ]:




