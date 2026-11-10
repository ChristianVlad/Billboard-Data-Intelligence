"""Orquestación: dato crudo → tablas de resultados → artefactos (CSV + Excel + metadatos)."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from . import config, data, dispersion, genres, quality


@dataclass
class Results:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def slug(source: str) -> str:
    return source.lower().replace(" ", "_")


def run_pipeline(path: Path | str | None = None, n_perm: int | None = None,
                 sources: list[str] | None = None, pairwise: bool = True,
                 raw: pd.DataFrame | None = None) -> Results:
    """Ejecuta todo el análisis. `raw` permite inyectar datos (tests) sin leer el Excel."""
    n_perm = n_perm or config.N_PERMUTATIONS
    sources = sources or config.DEFAULT_NETWORK_SOURCES
    src_path = Path(path or config.RAW_DATA_PATH)
    if raw is None:
        raw = data.load_raw(src_path)

    df = data.prepare(raw)
    analysis, audio_complete = data.analysis_sample(df)
    quality.validate_analysis_sample(raw, audio_complete)

    tables: dict[str, pd.DataFrame] = {}
    tables.update(quality.run_quality_checks(df))
    tables["audio_profile"] = dispersion.epoch_profile(analysis)
    tables.update(dispersion.run_dispersion(audio_complete, n_perm=n_perm, pairwise=pairwise))

    for source in sources:
        s = slug(source)
        tags = genres.build_tag_table(df, source)
        nodes, edges = genres.cooccurrence(tags)
        _, edges_ep = genres.cooccurrence(tags, by_epoch=True)
        nodes_ep, _ = genres.cooccurrence(tags, by_epoch=True)
        G = genres.build_graph(nodes, edges)
        pos = genres.layout_positions(G)
        total = int(tags["Record_ID"].nunique())
        tables[f"genre_prevalence__{s}"] = genres.tag_prevalence(tags)
        tables[f"nodes_global__{s}"] = nodes
        tables[f"nodes_epoch__{s}"] = nodes_ep
        tables[f"cooc_global__{s}"] = edges
        tables[f"cooc_epoch__{s}"] = edges_ep
        tables[f"assoc_global__{s}"] = genres.association_metrics(tags)
        tables[f"assoc_epoch__{s}"] = genres.association_metrics(tags, by_epoch=True)
        tables[f"node_metrics__{s}"] = genres.node_metrics(G, total)
        tables[f"layout__{s}"] = pd.DataFrame([{"Tag": k, "x": v[0], "y": v[1]} for k, v in pos.items()])

    g = tables["permdisp_global"].iloc[0]
    pw = tables.get("permdisp_pairwise")
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(src_path.name),
        "dataset_sha256": data.file_sha256(src_path) if src_path.exists() else None,
        "n_raw": int(len(raw)),
        "n_audio_complete": int(len(audio_complete)),
        "n_bpm_flagged": int(df["BPM_flagged"].sum()),
        "periods": config.PERIOD_LABELS,
        "audio_features": config.AUDIO_FEATURES,
        "network_sources": sources,
        "parameters": {"n_permutations": int(n_perm), "alpha": config.ALPHA, "seed": config.RANDOM_STATE},
        "permdisp": {"F": float(g["F"]), "p_value": float(g["P_Value"]),
                     "eta_squared": float(g["Eta_Squared"]), "decision": str(g["Decision"])},
        "pairwise_significant": int(pw["Significant_Holm"].sum()) if pw is not None else None,
        "pairwise_total": int(len(pw)) if pw is not None else None,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
    }
    return Results(tables=tables, meta=meta)


# ---------- persistencia ----------
def save_results(res: Results, out_dir: Path | str | None = None, excel: bool = True) -> Path:
    out = Path(out_dir or config.ARTIFACTS_DIR)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name, df in res.tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    (out / "meta.json").write_text(json.dumps(res.meta, indent=2, ensure_ascii=False))
    if excel:
        export_excel(res, out / "billboard_results.xlsx")
    return out


def load_results(out_dir: Path | str | None = None) -> Results:
    out = Path(out_dir or config.ARTIFACTS_DIR)
    meta_path = out / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No hay resultados en {out}. Ejecuta `make pipeline`.")
    tables = {p.stem: pd.read_csv(p) for p in sorted((out / "tables").glob("*.csv"))}
    return Results(tables=tables, meta=json.loads(meta_path.read_text()))


def _sheet_plan(res: Results) -> list[tuple[str, str]]:
    plan = [("Calidad_Duplicados", "quality_duplicates"), ("Calidad_Faltantes", "quality_missing"),
            ("Calidad_Logica", "quality_logical"), ("Cobertura_Audio", "quality_audio_audit"),
            ("Perfil_Sonoro", "audio_profile"), ("Dispersion_Epoca", "dispersion_summary"),
            ("PERMDISP_Global", "permdisp_global"), ("PERMDISP_Pares", "permdisp_pairwise"),
            ("Dispersion_Canciones", "dispersion_songs")]
    for source in res.meta.get("network_sources", []):
        s, short = slug(source), source.replace(" ", "")
        plan += [(f"Prev_{short}", f"genre_prevalence__{s}"), (f"Coocurr_{short}", f"cooc_global__{s}"),
                 (f"Coocurr_Epoca_{short}", f"cooc_epoch__{s}"), (f"Asoc_{short}", f"assoc_global__{s}"),
                 (f"Asoc_Epoca_{short}", f"assoc_epoch__{s}"), (f"Nodos_{short}", f"node_metrics__{s}")]
    return [(sheet, key) for sheet, key in plan if key in res.tables]


def _flatten_meta(meta: dict) -> pd.DataFrame:
    rows = []

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            rows.append({"Campo": prefix, "Valor": ", ".join(map(str, obj)) if isinstance(obj, list) else obj})

    walk("", meta)
    return pd.DataFrame(rows)


def export_excel(res: Results, path: Path | str) -> Path:
    """Libro de Excel con un resumen y las tablas relevantes (encabezados con formato y columnas ajustadas)."""
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        _flatten_meta(res.meta).to_excel(xl, sheet_name="Resumen", index=False)
        for sheet, key in _sheet_plan(res):
            t = res.tables[key].copy()
            num = t.select_dtypes("number").columns
            t[num] = t[num].round(4)
            t.to_excel(xl, sheet_name=sheet[:31], index=False)
        for ws in xl.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="305496")
            for i, col in enumerate(ws.columns, start=1):
                width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
                ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 10), 60)
    return path
