"""Auditoría de calidad: cada función devuelve una tabla; `run_quality_checks` las reúne."""
from __future__ import annotations

import pandas as pd

from . import config


def duplicate_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "Filas completamente duplicadas": df.duplicated().sum(),
        "Canción, artista y fecha duplicados": df.duplicated(subset=config.IDENTIFIER_COLUMNS).sum(),
        "Canción y artista duplicados": df.duplicated(subset=["Song", "Artist"]).sum(),
    }
    return pd.Series(checks, name="Cantidad").rename_axis("Chequeo").reset_index()


def missing_summary(df: pd.DataFrame, top: int | None = 20) -> pd.DataFrame:
    out = df.isna().sum().to_frame("Missing")
    out["Missing_Percentage"] = (out["Missing"] / len(df) * 100).round(2)
    out = out.sort_values("Missing_Percentage", ascending=False).rename_axis("Column").reset_index()
    return out.head(top) if top else out


def audio_audit(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Epoch", observed=False)
    out = g.agg(Total_Songs=("Song", "size"), Complete_Audio=("Audio_Complete", "sum"))
    out["Incomplete_Audio"] = out["Total_Songs"] - out["Complete_Audio"]
    out["Coverage_Percentage"] = (out["Complete_Audio"] / out["Total_Songs"] * 100).round(2)
    return out.reset_index()


def logical_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Restricciones lógicas del diccionario de datos. Omite (y registra) las que no aplican."""
    rows = []

    def add(name, needed, fn):
        if all(c in df.columns for c in needed):
            rows.append({"Chequeo": name, "Violaciones": int(fn().sum()), "Estado": "evaluado"})
        else:
            rows.append({"Chequeo": name, "Violaciones": None, "Estado": "omitido (columna ausente)"})

    r = ["Rating 1", "Rating 2", "Rating 3"]
    add("Overall Rating ≠ media de los 3 jueces", r + ["Overall Rating"],
        lambda: (df["Overall Rating"] - df[r].mean(axis=1)).abs() > 0.01)
    add("Divisiveness ≠ distancia media por pares", r + ["Divisiveness"],
        lambda: (df["Divisiveness"] - ((df[r[0]] - df[r[1]]).abs() + (df[r[0]] - df[r[2]]).abs()
                                       + (df[r[1]] - df[r[2]]).abs()) / 3).abs() > 0.01)
    add("Intro más larga que la canción", ["Intro Length(Sec)", "Length (Sec)"],
        lambda: df["Intro Length(Sec)"] > df["Length (Sec)"])
    add("Parte instrumental más larga que la canción", ["Instrumental Length(Sec)", "Length (Sec)"],
        lambda: df["Instrumental Length(Sec)"] > df["Length (Sec)"])
    add("Semanas en el #1 < 1", ["Weeks at Number One"], lambda: df["Weeks at Number One"] < 1)
    add(f"BPM fuera de [{config.BPM_MIN}, {config.BPM_MAX}]", ["BPM"],
        lambda: df["BPM"].notna() & ~df["BPM"].between(config.BPM_MIN, config.BPM_MAX))
    spot = ["Energy", "Danceability", "Happiness", "Acousticness"]
    add("Métricas Spotify fuera de [0, 100]", spot, lambda: ((df[spot] < 0) | (df[spot] > 100)).any(axis=1))
    add("Loudness > 0 dB", ["Loudness (dB)"], lambda: df["Loudness (dB)"] > 0)
    return pd.DataFrame(rows)


def run_quality_checks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "quality_duplicates": duplicate_checks(df),
        "quality_missing": missing_summary(df),
        "quality_logical": logical_checks(df),
        "quality_audio_audit": audio_audit(df),
    }


def validate_analysis_sample(raw: pd.DataFrame, audio_complete: pd.DataFrame,
                             expected_raw: int | None = None, expected_complete: int | None = None) -> None:
    """Falla (AssertionError) si la muestra analítica no es coherente."""
    assert audio_complete[config.AUDIO_FEATURES].notna().all().all(), "audio_complete contiene nulos"
    assert audio_complete["Epoch"].notna().all(), "hay canciones sin época"
    assert len(audio_complete) <= len(raw)
    if expected_raw is not None:
        assert len(raw) == expected_raw, f"se esperaban {expected_raw} filas y hay {len(raw)}"
    if expected_complete is not None:
        assert len(audio_complete) == expected_complete, (
            f"se esperaban {expected_complete} filas completas y hay {len(audio_complete)}")
