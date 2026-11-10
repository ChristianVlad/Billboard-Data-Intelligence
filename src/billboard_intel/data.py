"""Ingestión y preparación de la muestra analítica."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_raw(path: Path | str | None = None, sheet_name: str | None = None) -> pd.DataFrame:
    """Lee el Excel original sin modificarlo (la copia de trabajo se hace en `prepare`)."""
    return pd.read_excel(path or config.RAW_DATA_PATH, sheet_name=sheet_name or config.SHEET_NAME,
                         na_values=config.NA_VALUES)


def file_sha256(path: Path | str | None = None) -> str:
    """Huella del archivo fuente: permite comprobar que dos ejecuciones usaron el mismo dato."""
    h = hashlib.sha256()
    with open(path or config.RAW_DATA_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_required_columns(df: pd.DataFrame, required: list[str] | None = None) -> None:
    missing = [c for c in (required or config.REQUIRED_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {missing}")


def assign_epoch(years: pd.Series) -> pd.Categorical:
    cut = pd.cut(years, bins=config.PERIOD_BINS, labels=config.PERIOD_LABELS, right=False)
    return pd.Categorical(cut, categories=config.PERIOD_LABELS, ordered=True)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Copia de trabajo con fecha, año, época y banderas de calidad. `raw` no se modifica."""
    check_required_columns(raw)
    df = raw.copy(deep=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Epoch"] = assign_epoch(df["Year"])
    df["BPM_flagged"] = df["BPM"].notna() & ((df["BPM"] < config.BPM_MIN) | (df["BPM"] > config.BPM_MAX))
    df["Audio_Complete"] = df[config.AUDIO_FEATURES].notna().all(axis=1)
    return df


def analysis_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (analysis, audio_complete).

    El BPM sospechoso se anula SOLO en `analysis` (nunca en `df`/`raw`); `audio_complete`
    conserva las filas con las seis características disponibles.
    """
    analysis = df.copy(deep=True)
    analysis.loc[analysis["BPM_flagged"], "BPM"] = np.nan
    audio_complete = analysis.dropna(subset=config.AUDIO_FEATURES + ["Epoch"]).copy()
    return analysis, audio_complete
