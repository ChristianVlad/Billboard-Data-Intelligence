from __future__ import annotations

import threading

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from billboard_intel import config, pipeline, viz

app = FastAPI(title="Billboard Data Intelligence API", version="0.1.0",
              description="Resultados del análisis de la evolución del sonido de los #1 del Hot 100.")

_lock = threading.Lock()
_cache: dict = {"key": None, "results": None}


def _json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def get_results() -> pipeline.Results:
    meta = config.ARTIFACTS_DIR / "meta.json"
    if not meta.exists():
        raise HTTPException(503, "Sin resultados. Ejecuta `make pipeline` o POST /pipeline/run.")
    key = meta.stat().st_mtime_ns
    with _lock:
        if _cache["key"] != key:
            _cache.update(key=key, results=pipeline.load_results())
        return _cache["results"]


def _table(name: str) -> pd.DataFrame:
    t = get_results().tables
    if name not in t:
        raise HTTPException(404, f"Tabla '{name}' no disponible.")
    return t[name]


def _source_slug(source: str) -> str:
    meta = get_results().meta
    valid = {pipeline.slug(s): s for s in meta["network_sources"]}
    s = source.lower().replace(" ", "_")
    if s not in valid:
        raise HTTPException(404, f"Fuente '{source}' no disponible. Opciones: {list(valid.values())}")
    return s


def _check_epoch(epoch: str | None) -> None:
    if epoch and epoch not in config.PERIOD_LABELS:
        raise HTTPException(422, f"Época inválida. Opciones: {config.PERIOD_LABELS}")


# ---------- estado ----------
@app.get("/health")
def health():
    return {"status": "ok", "results_available": (config.ARTIFACTS_DIR / "meta.json").exists()}


@app.get("/summary")
def summary():
    return get_results().meta


# ---------- sonido y diversidad ----------
@app.get("/epochs")
def epochs():
    return {"epochs": config.PERIOD_LABELS, "audio_coverage": _json(_table("quality_audio_audit"))}


@app.get("/profile")
def profile(feature: str | None = Query(None, description="Una de las 6 características de audio")):
    df = _table("audio_profile")
    if feature:
        if feature not in config.AUDIO_FEATURES:
            raise HTTPException(422, f"Característica inválida. Opciones: {config.AUDIO_FEATURES}")
        df = df[df["Feature"] == feature]
    return _json(df)


@app.get("/dispersion")
def dispersion_summary():
    return _json(_table("dispersion_summary"))


@app.get("/dispersion/songs")
def dispersion_songs():
    return _json(_table("dispersion_songs"))


@app.get("/dispersion/permdisp")
def permdisp_global():
    return _json(_table("permdisp_global"))[0]


@app.get("/dispersion/pairwise")
def permdisp_pairwise():
    return _json(_table("permdisp_pairwise"))


# ---------- géneros y redes ----------
@app.get("/genres/sources")
def genre_sources():
    return get_results().meta["network_sources"]


@app.get("/genres/{source}/prevalence")
def prevalence(source: str):
    return _json(_table(f"genre_prevalence__{_source_slug(source)}"))


@app.get("/genres/{source}/associations")
def associations(source: str, epoch: str | None = None, min_weight: int = Query(1, ge=1),
                 sort_by: str = Query("Lift", pattern="^(Lift|Jaccard|Weight)$"), limit: int = Query(25, ge=1, le=500)):
    _check_epoch(epoch)
    s = _source_slug(source)
    df = _table(f"assoc_epoch__{s}" if epoch else f"assoc_global__{s}")
    if epoch:
        df = df[df["Epoch"] == epoch]
    df = df[df["Weight"] >= min_weight].sort_values(sort_by, ascending=False).head(limit)
    return _json(df)


@app.get("/genres/{source}/node-metrics")
def node_metrics(source: str):
    return _json(_table(f"node_metrics__{_source_slug(source)}"))


def _network(source: str, epoch: str | None, min_weight: int):
    _check_epoch(epoch)
    s = _source_slug(source)
    if epoch:
        nodes = _table(f"nodes_epoch__{s}")
        nodes = nodes[nodes["Epoch"] == epoch][["Tag", "Song_Count"]]
        edges = _table(f"cooc_epoch__{s}")
        edges = edges[edges["Epoch"] == epoch][["Source", "Target", "Weight"]]
        prev = _table(f"genre_prevalence__{s}")
        prev = prev[prev["Epoch"] == epoch]
        if prev.empty:
            raise HTTPException(404, f"'{source}' no tiene canciones clasificadas en {epoch} (p. ej. CDR Genre está vacío en 2020-2025).")
        total = int(prev["Classified_Songs"].max())
    else:
        nodes, edges = _table(f"nodes_global__{s}"), _table(f"cooc_global__{s}")
        total = int(_table(f"genre_prevalence__{s}")["Classified_Songs"].max())
    layout = _table(f"layout__{s}")
    pos = {r.Tag: (r.x, r.y) for r in layout.itertuples()}
    all_nodes = _table(f"nodes_global__{s}")["Tag"]
    return nodes, edges, pos, total, viz.genre_colors(all_nodes)


@app.get("/networks/{source}")
def network_json(source: str, epoch: str | None = None, min_weight: int = Query(1, ge=1)):
    nodes, edges, pos, total, colors = _network(source, epoch, min_weight)
    edges = edges[edges["Weight"] >= min_weight]
    return JSONResponse({"epoch": epoch or "all", "classified_songs": total,
                         "nodes": _json(nodes.assign(Color=nodes["Tag"].map(colors))),
                         "edges": _json(edges)})


@app.get("/networks/{source}/html", response_class=HTMLResponse)
def network_html(source: str, epoch: str | None = None, min_weight: int = Query(1, ge=1),
                 height: str = "650px"):
    nodes, edges, pos, total, colors = _network(source, epoch, min_weight)
    return viz.network_html(nodes, edges, pos, total, height=height, min_weight=min_weight, colors=colors)


# ---------- calidad y exportación ----------
@app.get("/quality")
def quality_tables():
    t = get_results().tables
    return {k.removeprefix("quality_"): _json(v) for k, v in t.items() if k.startswith("quality_")}


@app.get("/export/excel")
def export_excel():
    path = config.ARTIFACTS_DIR / "billboard_results.xlsx"
    if not path.exists():
        raise HTTPException(503, "El Excel no existe. Ejecuta el pipeline.")
    return FileResponse(path, filename=path.name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.post("/pipeline/run")
def run_pipeline(n_permutations: int = Query(999, ge=99, le=20000, description="Permutaciones (9999 en el análisis final)")):
    """Recalcula todo y guarda los artefactos. Con 9999 permutaciones tarda ~15-30 s."""
    try:
        res = pipeline.run_pipeline(n_perm=n_permutations)
    except FileNotFoundError as e:
        raise HTTPException(404, f"No se encontró el dataset: {e}")
    pipeline.save_results(res)
    return res.meta
