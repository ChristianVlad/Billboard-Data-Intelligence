"""Géneros y estilos: etiquetas → redes de coocurrencia → métricas de asociación."""
from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from . import config


# ---------- etiquetas ----------
def split_music_tags(value) -> list[str]:
    """';' es el único delimitador. '/' y ',' forman parte de nombres ('Funk/Soul', 'Folk, World, & Country')."""
    if pd.isna(value):
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]


def canonicalize_tag(tag):
    if pd.isna(tag):
        return tag
    tag = " ".join(str(tag).strip().split())
    return config.TAG_CANONICAL_MAP.get(tag, tag)


def build_tag_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Formato largo: una fila por (canción, etiqueta). `Record_ID` = índice de la fila original."""
    out = df[["Song", "Artist", "Date", "Epoch", column]].copy()
    out["Record_ID"] = df.index
    out["Tag"] = out[column].apply(split_music_tags)
    out = out.explode("Tag").dropna(subset=["Tag"])
    out["Tag_Canonical"] = out["Tag"].apply(canonicalize_tag)
    return out[["Record_ID", "Song", "Artist", "Date", "Epoch", "Tag", "Tag_Canonical"]].reset_index(drop=True)


def build_tag_tables(df: pd.DataFrame, sources: list[str] | None = None) -> dict[str, pd.DataFrame]:
    return {s: build_tag_table(df, s) for s in (sources or config.GENRE_COLUMNS)}


def _unique_assignments(tags: pd.DataFrame) -> pd.DataFrame:
    return tags[["Record_ID", "Epoch", "Tag_Canonical"]].dropna().drop_duplicates()


# ---------- prevalencia ----------
def tag_prevalence(tags: pd.DataFrame, by_epoch: bool = True) -> pd.DataFrame:
    """% de canciones clasificadas (global o por época) que tienen cada etiqueta. Una canción cuenta una vez."""
    w = _unique_assignments(tags)
    if by_epoch:
        den = w.groupby("Epoch", observed=True)["Record_ID"].nunique()
        out = (w.groupby(["Epoch", "Tag_Canonical"], observed=True)["Record_ID"].nunique()
               .rename("Song_Count").reset_index())
        out["Classified_Songs"] = out["Epoch"].astype(str).map({str(k): v for k, v in den.items()}).astype("int64")
    else:
        out = w.groupby("Tag_Canonical")["Record_ID"].nunique().rename("Song_Count").reset_index()
        out["Classified_Songs"] = int(w["Record_ID"].nunique())
    out["Percentage"] = out["Song_Count"] / out["Classified_Songs"] * 100
    return out


# ---------- coocurrencia ----------
def cooccurrence(tags: pd.DataFrame, by_epoch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (nodes, edges). Peso = nº de canciones que tienen ambas etiquetas."""
    w = _unique_assignments(tags)
    node_keys = ["Epoch", "Tag_Canonical"] if by_epoch else ["Tag_Canonical"]
    nodes = (w.groupby(node_keys, observed=True)["Record_ID"].nunique().rename("Song_Count")
             .reset_index().rename(columns={"Tag_Canonical": "Tag"}))

    records = []
    for (rid, epoch), grp in w.groupby(["Record_ID", "Epoch"], observed=True):
        for a, b in combinations(sorted(set(grp["Tag_Canonical"])), 2):
            records.append((a, b, rid, epoch))
    cols = ["Source", "Target", "Record_ID", "Epoch"]
    raw = pd.DataFrame(records, columns=cols)
    agg = ["Source", "Target"] + (["Epoch"] if by_epoch else [])
    if raw.empty:
        return nodes, pd.DataFrame(columns=agg + ["Weight"])
    edges = raw.groupby(agg, observed=True)["Record_ID"].nunique().rename("Weight").reset_index()
    return nodes, edges


def association_metrics(tags: pd.DataFrame, by_epoch: bool = False) -> pd.DataFrame:
    """Jaccard (solapamiento) y Lift (observado / esperado si fueran independientes) por par de etiquetas."""
    w = _unique_assignments(tags)
    _, edges = cooccurrence(tags, by_epoch)
    if edges.empty:
        return edges.assign(Jaccard=[], Lift=[])
    if by_epoch:
        cnt = w.groupby(["Epoch", "Tag_Canonical"], observed=True)["Record_ID"].nunique().to_dict()
        tot = w.groupby("Epoch", observed=True)["Record_ID"].nunique().to_dict()
        ca = [cnt[(e, s)] for e, s in zip(edges["Epoch"], edges["Source"])]
        cb = [cnt[(e, t)] for e, t in zip(edges["Epoch"], edges["Target"])]
        total = np.array([tot[e] for e in edges["Epoch"]], dtype=float)
    else:
        cnt = w.groupby("Tag_Canonical")["Record_ID"].nunique().to_dict()
        ca = [cnt[s] for s in edges["Source"]]
        cb = [cnt[t] for t in edges["Target"]]
        total = np.full(len(edges), float(w["Record_ID"].nunique()))
    out = edges.copy()
    out["Genre_A_Count"], out["Genre_B_Count"], out["Classified_Songs"] = ca, cb, total.astype(int)
    out["Jaccard"] = out["Weight"] / (out["Genre_A_Count"] + out["Genre_B_Count"] - out["Weight"])
    out["Expected_Shared_Songs"] = out["Genre_A_Count"] * out["Genre_B_Count"] / total
    out["Lift"] = out["Weight"] / out["Expected_Shared_Songs"]
    out["Association_Type"] = np.select([out["Lift"] > 1, out["Lift"] < 1],
                                        ["Positive association", "Negative association"],
                                        default="Approximately independent")
    return out


# ---------- grafos ----------
def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame, min_weight: int = 1) -> nx.Graph:
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["Tag"], frequency=int(r["Song_Count"]))
    for _, r in edges[edges["Weight"] >= min_weight].iterrows():
        G.add_edge(r["Source"], r["Target"], weight=int(r["Weight"]))
    return G


def layout_positions(G: nx.Graph, seed: int = config.RANDOM_STATE) -> dict[str, tuple[float, float]]:
    """Posiciones calculadas UNA vez sobre la red global: así los nodos no se mueven entre épocas."""
    if G.number_of_nodes() == 0:
        return {}
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=1.2, iterations=200)
    return {n: (float(x), float(y)) for n, (x, y) in pos.items()}


def node_metrics(G: nx.Graph, total_songs: int) -> pd.DataFrame:
    rows = [{"Genre": n, "Song_Count": d.get("frequency", 0),
             "Song_Prevalence": d.get("frequency", 0) / total_songs * 100 if total_songs else np.nan,
             "Connected_Genres": G.degree(n), "Cooccurrence_Strength": G.degree(n, weight="weight")}
            for n, d in G.nodes(data=True)]
    cols = ["Genre", "Song_Count", "Song_Prevalence", "Connected_Genres", "Cooccurrence_Strength"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["Cooccurrence_Strength", "Song_Count"], ascending=False).reset_index(drop=True)
