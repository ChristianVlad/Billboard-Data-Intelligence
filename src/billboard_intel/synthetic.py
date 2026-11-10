"""Datos sintéticos con la misma estructura que el Excel real (solo para tests, CI y demos; NO son datos reales)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config

GENRES = ["Pop", "Rock", "Funk/Soul", "Hip Hop", "Electronic"]


def make_synthetic(n=420, seed=0) -> pd.DataFrame:
    """Dataset sintético con las mismas columnas clave; la dispersión es menor en 2000-2019."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(np.sort(rng.integers(
        pd.Timestamp("1958-08-04").value // 10**9, pd.Timestamp("2025-01-25").value // 10**9, n)), unit="s")
    years = dates.year.to_numpy()
    spread = np.where((years >= 2000) & (years < 2020), 0.6, 1.0)
    df = pd.DataFrame({"Song": [f"Song {i}" for i in range(n)], "Artist": [f"Artist {i % 150}" for i in range(n)],
                       "Date": dates, "Weeks at Number One": rng.integers(1, 6, n)})
    base = {"BPM": 115, "Energy": 60, "Danceability": 62, "Happiness": 60, "Loudness (dB)": -9, "Acousticness": 30}
    scale = {"BPM": 22, "Energy": 18, "Danceability": 14, "Happiness": 22, "Loudness (dB)": 3, "Acousticness": 25}
    for f in config.AUDIO_FEATURES:
        df[f] = (base[f] + rng.normal(0, scale[f], n) * spread).round(0)
    for c in ["Energy", "Danceability", "Happiness", "Acousticness"]:
        df[c] = df[c].clip(0, 100)
    for r in ["Rating 1", "Rating 2", "Rating 3"]:
        df[r] = rng.integers(1, 11, n)
    df["Overall Rating"] = df[["Rating 1", "Rating 2", "Rating 3"]].mean(axis=1).round(2)
    df["Divisiveness"] = ((df["Rating 1"] - df["Rating 2"]).abs() + (df["Rating 1"] - df["Rating 3"]).abs()
                          + (df["Rating 2"] - df["Rating 3"]).abs()) / 3
    df["Length (Sec)"] = rng.integers(120, 300, n)
    df["Intro Length(Sec)"] = rng.integers(0, 30, n)
    df["Instrumental Length(Sec)"] = rng.integers(0, 40, n)

    def tags(k_max):
        return [";".join(rng.choice(GENRES, rng.integers(1, k_max + 1), replace=False)) for _ in range(n)]

    df["Discogs Genre"], df["Discogs Style"] = tags(3), tags(2)
    df["CDR Genre"], df["CDR Style"] = tags(2), tags(2)
    df.loc[rng.choice(n, 25, replace=False), "CDR Genre"] = np.nan
    df.loc[3, "BPM"] = 16          # BPM sospechoso
    df.loc[5, "Energy"] = np.nan   # audio incompleto
    df.loc[10, "Discogs Genre"] = "Funk / Soul"  # variante de escritura
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic.xlsx"
    make_synthetic(n=1177, seed=1).to_excel(out, sheet_name=config.SHEET_NAME, index=False)
    print(f"Dataset sintético escrito en {out}")
