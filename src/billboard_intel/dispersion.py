"""Diversidad sonora: distancia al centroide de cada época y prueba PERMDISP.

Idea en una frase: medimos qué tan lejos está cada canción de la "canción típica" de su época
y preguntamos si esas distancias difieren entre épocas más de lo que cabría esperar al azar
(barajando las etiquetas de época miles de veces).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import config


# ---------- utilidades numéricas (vectorizadas: ~10k permutaciones en segundos) ----------
def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError("Existen valores ausentes en las características.")
    sd = X.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError("No se puede estandarizar una característica constante.")
    return (X - X.mean(axis=0)) / sd


def _distances(X: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=k).astype(float)
    centroids = np.column_stack(
        [np.bincount(codes, weights=X[:, j], minlength=k) for j in range(X.shape[1])]) / counts[:, None]
    return np.sqrt(((X - centroids[codes]) ** 2).sum(axis=1))


def _anova(d: np.ndarray, codes: np.ndarray, k: int) -> tuple[float, float]:
    """Devuelve (F, eta²) del ANOVA de una vía sobre las distancias."""
    counts = np.bincount(codes, minlength=k).astype(float)
    means = np.bincount(codes, weights=d, minlength=k) / counts
    ssb = float((counts * (means - d.mean()) ** 2).sum())
    ssw = float(((d - means[codes]) ** 2).sum())
    n = len(d)
    f = (ssb / (k - 1)) / (ssw / (n - k)) if ssw > 0 else np.inf
    eta = ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan
    return f, eta


def _encode(labels, order):
    present = set(labels)
    order = [o for o in order if o in present]
    mapping = {g: i for i, g in enumerate(order)}
    return np.array([mapping[x] for x in labels]), order


def holm(p: np.ndarray) -> np.ndarray:
    """Corrección de Holm-Bonferroni (equivalente a statsmodels `multipletests(method='holm')`)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ---------- PERMDISP ----------
def permdisp(X: np.ndarray, labels, order: list[str] | None = None,
             n_perm: int = config.N_PERMUTATIONS, seed: int = config.RANDOM_STATE) -> dict:
    """PERMDISP global: F sobre distancias al centroide + p-valor por permutación de etiquetas."""
    labels = np.asarray(labels).astype(str)
    codes, groups = _encode(labels, order or sorted(set(labels)))
    k = len(groups)
    d = _distances(X, codes, k)
    f_obs, eta = _anova(d, codes, k)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        pc = codes[rng.permutation(len(codes))]
        f, _ = _anova(_distances(X, pc, k), pc, k)
        count += f >= f_obs
    return {"F": float(f_obs), "p_value": float((1 + count) / (n_perm + 1)), "eta_squared": float(eta),
            "n": int(len(codes)), "groups": groups, "n_permutations": int(n_perm), "distances": d}


def pairwise_permdisp(X: np.ndarray, labels, order: list[str],
                      n_perm: int = config.N_PERMUTATIONS, seed: int = config.RANDOM_STATE,
                      alpha: float = config.ALPHA) -> pd.DataFrame:
    """Todas las comparaciones por pares (A = época anterior, B = posterior) con Holm."""
    labels = np.asarray(labels).astype(str)
    rows = []
    for a, b in combinations([g for g in order if g in set(labels)], 2):
        mask = np.isin(labels, [a, b])
        Xp, lp = X[mask], labels[mask]
        codes = (lp == b).astype(int)
        d = _distances(Xp, codes, 2)
        f_obs, eta = _anova(d, codes, 2)
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_perm):
            pc = codes[rng.permutation(len(codes))]
            f, _ = _anova(_distances(Xp, pc, 2), pc, 2)
            count += f >= f_obs
        rows.append({"Epoch_A": a, "Epoch_B": b, "F_Statistic": f_obs,
                     "P_Value": (1 + count) / (n_perm + 1),
                     "Mean_Distance_A": d[codes == 0].mean(), "Mean_Distance_B": d[codes == 1].mean(),
                     "Mean_Difference_B_minus_A": d[codes == 1].mean() - d[codes == 0].mean(),
                     "Eta_Squared": eta})
    out = pd.DataFrame(rows)
    out["P_Value_Holm"] = holm(out["P_Value"].to_numpy())
    out["Significant_Holm"] = out["P_Value_Holm"] < alpha
    out["Conclusion"] = np.where(out["Significant_Holm"], "Significant difference", "No significant difference")
    return out.sort_values("P_Value_Holm").reset_index(drop=True)


# ---------- tablas listas para reportar ----------
def epoch_profile(analysis: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Perfil sonoro por época en formato largo: una fila por (Epoch, Feature) con count/mean/median/std."""
    features = features or config.AUDIO_FEATURES
    rows = []
    for epoch, grp in analysis.groupby("Epoch", observed=True):
        for f in features:
            s = grp[f]
            rows.append({"Epoch": epoch, "Feature": f, "count": int(s.count()), "mean": s.mean(),
                         "median": s.median(), "std": s.std()})
    return pd.DataFrame(rows)


def run_dispersion(audio_complete: pd.DataFrame, features: list[str] | None = None,
                   n_perm: int = config.N_PERMUTATIONS, seed: int = config.RANDOM_STATE,
                   alpha: float = config.ALPHA, pairwise: bool = True) -> dict[str, pd.DataFrame]:
    """Pipeline completo de la hipótesis: distancias, PERMDISP global y comparaciones por pares."""
    features = features or config.AUDIO_FEATURES
    data = audio_complete.dropna(subset=features + ["Epoch"]).copy()
    X = standardize(data[features].to_numpy(dtype=float))
    labels = data["Epoch"].astype(str).to_numpy()
    order = config.PERIOD_LABELS

    glob = permdisp(X, labels, order, n_perm, seed)
    data["Distance_To_Centroid"] = glob["distances"]

    summary = (data.groupby("Epoch", observed=True)["Distance_To_Centroid"]
               .agg(Song_Count="count", Mean_Distance="mean", Median_Distance="median",
                    Standard_Deviation="std", Minimum="min", Maximum="max").reset_index())
    songs = data[["Song", "Artist", "Date", "Epoch", "Distance_To_Centroid"]].reset_index(drop=True)
    decision = "Reject H0" if glob["p_value"] < alpha else "Fail to reject H0"
    g_df = pd.DataFrame([{"Test": "PERMDISP", "F": glob["F"], "P_Value": glob["p_value"],
                          "Eta_Squared": glob["eta_squared"], "N": glob["n"],
                          "Permutations": glob["n_permutations"], "Alpha": alpha, "Decision": decision}])
    out = {"dispersion_summary": summary, "dispersion_songs": songs, "permdisp_global": g_df}
    if pairwise:
        out["permdisp_pairwise"] = pairwise_permdisp(X, labels, order, n_perm, seed, alpha)
    return out
