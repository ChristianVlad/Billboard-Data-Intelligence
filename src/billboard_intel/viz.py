"""Visualizaciones: figuras de matplotlib (notebook/informe) y red interactiva con PyVis (API/dashboard)."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B", "#B279A2",
           "#FF9DA6", "#9D755D", "#BAB0AC", "#2E7D32", "#8E24AA", "#00838F", "#6D4C41"]


def genre_colors(genres) -> dict[str, str]:
    """Color estable por género (orden alfabético), igual en todas las épocas."""
    return {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted(set(genres)))}


# ---------- matplotlib ----------
def plot_profile_evolution(profile: pd.DataFrame):
    feats = list(profile["Feature"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, f in zip(axes.ravel(), feats):
        d = profile[profile["Feature"] == f]
        ax.plot(d["Epoch"], d["mean"], marker="o", color=PALETTE[0])
        ax.fill_between(d["Epoch"], d["mean"] - d["std"], d["mean"] + d["std"], alpha=0.15, color=PALETTE[0])
        ax.set_title(f)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Perfil sonoro promedio por época (banda = ±1 desviación estándar)")
    fig.tight_layout()
    return fig


def plot_dispersion_boxplot(songs: pd.DataFrame):
    order = [e for e in config.PERIOD_LABELS if e in set(songs["Epoch"].astype(str))]
    data = [songs.loc[songs["Epoch"].astype(str) == e, "Distance_To_Centroid"] for e in order]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_title("Dispersión del perfil sonoro dentro de cada época")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Distancia al centroide de su época")
    fig.tight_layout()
    return fig


def plot_pairwise_heatmap(pairwise: pd.DataFrame):
    """Diferencia de dispersión (B − A) entre épocas; * = significativa tras Holm."""
    order = config.PERIOD_LABELS
    diff = pd.DataFrame(np.nan, index=order, columns=order)
    sig = pd.DataFrame("", index=order, columns=order)
    for _, r in pairwise.iterrows():
        diff.loc[r["Epoch_A"], r["Epoch_B"]] = r["Mean_Difference_B_minus_A"]
        sig.loc[r["Epoch_A"], r["Epoch_B"]] = "*" if r["Significant_Holm"] else ""
    lim = np.nanmax(np.abs(diff.to_numpy()))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(diff.to_numpy(dtype=float), cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order)
    for i in range(len(order)):
        for j in range(len(order)):
            v = diff.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}{sig.iloc[i, j]}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Diferencia de distancia media (columna − fila)")
    ax.set_title("Cambio de dispersión entre épocas (* significativo, Holm)")
    fig.tight_layout()
    return fig


# ---------- PyVis ----------
def network_html(nodes: pd.DataFrame, edges: pd.DataFrame, positions: dict, total_songs: int,
                 height: str = "650px", min_weight: int = 1, colors: dict | None = None,
                 title: str | None = None) -> str:
    """HTML autocontenido (sin CDN) de una red de coocurrencia con posiciones fijas."""
    from pyvis.network import Network  # import perezoso: solo la API/notebook lo necesitan

    net = Network(height=height, width="100%", bgcolor="#FFFFFF", font_color="#263238",
                  directed=False, notebook=False, cdn_resources="in_line", heading=title or "")
    net.toggle_physics(False)
    colors = colors or genre_colors(nodes["Tag"])
    fmax = max(float(nodes["Song_Count"].max()), 1.0) if len(nodes) else 1.0

    strength: dict[str, int] = {}
    degree: dict[str, int] = {}
    kept = edges[edges["Weight"] >= min_weight]
    for _, e in kept.iterrows():
        for n in (e["Source"], e["Target"]):
            strength[n] = strength.get(n, 0) + int(e["Weight"])
            degree[n] = degree.get(n, 0) + 1

    for _, r in nodes.iterrows():
        tag, freq = r["Tag"], int(r["Song_Count"])
        x, y = positions.get(tag, (0.0, 0.0))
        net.add_node(tag, label=tag, x=x * 600, y=y * 600, physics=False,
                     color=colors.get(tag, "#999999"), size=12 + 38 * (freq / fmax) ** 0.5,
                     title=(f"{tag}\nSongs: {freq} ({freq / total_songs * 100:.1f}% of classified)\n"
                            f"Connected genres: {degree.get(tag, 0)}\nCo-occurrence strength: {strength.get(tag, 0)}"))
    for _, e in kept.iterrows():
        net.add_edge(e["Source"], e["Target"], value=int(e["Weight"]), color="#90A4AE",
                     title=f"{e['Source']} + {e['Target']}\nShared songs: {int(e['Weight'])}")
    html = net.generate_html(notebook=False)
    # pyvis enlaza bootstrap por CDN solo para el estilo del encabezado: se elimina para que funcione sin internet.
    html = re.sub(r"<link[^>]*bootstrap[^>]*>\s*", "", html)
    html = re.sub(r"<script[^>]*bootstrap[^>]*></script>\s*", "", html)
    return html
