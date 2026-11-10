"""Constantes y parámetros del análisis. Único lugar donde se definen."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = Path(os.getenv("BDI_DATA_PATH", ROOT / "data" / "raw" / "Billboard_Hot_100_Number_Ones_Database.xlsx"))
SHEET_NAME = "Data"
ARTIFACTS_DIR = Path(os.getenv("BDI_ARTIFACTS_DIR", ROOT / "artifacts"))

NA_VALUES = ["Cannot Locate"]  # "N/A" ya lo trata como nulo pandas por defecto

AUDIO_FEATURES = ["BPM", "Energy", "Danceability", "Happiness", "Loudness (dB)", "Acousticness"]
GENRE_COLUMNS = ["CDR Genre", "CDR Style", "Discogs Genre", "Discogs Style"]
IDENTIFIER_COLUMNS = ["Song", "Artist", "Date"]
REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + AUDIO_FEATURES + GENRE_COLUMNS + ["Weeks at Number One"]

PERIOD_BINS = [1958, 1970, 1980, 1990, 2000, 2010, 2020, 2026]
PERIOD_LABELS = ["1958-1969", "1970s", "1980s", "1990s", "2000s", "2010s", "2020-2025"]

BPM_MIN, BPM_MAX = 40, 220  # fuera de este rango se considera error de medición

# Etiquetas equivalentes detectadas en la auditoría de nombres
TAG_CANONICAL_MAP = {"Funk / Soul": "Funk/Soul"}

# Taxonomías que se procesan por defecto para las redes (los estilos son más ruidosos)
DEFAULT_NETWORK_SOURCES = ["Discogs Genre", "CDR Genre"]

RANDOM_STATE = 42
N_PERMUTATIONS = 9999
ALPHA = 0.05
