"""Línea de comandos: `python -m billboard_intel` ejecuta el pipeline y guarda los artefactos."""
from __future__ import annotations

import argparse

from . import config, pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="billboard_intel", description="Ejecuta el análisis y guarda artefactos.")
    ap.add_argument("--data", help="Ruta al Excel (por defecto data/raw/Billboard_Hot_100_Number_Ones_Database.xlsx)")
    ap.add_argument("--out", help="Carpeta de artefactos (por defecto ./artifacts)")
    ap.add_argument("--n-perm", type=int, default=config.N_PERMUTATIONS, help="Permutaciones (9999 en el análisis final)")
    ap.add_argument("--sources", nargs="*", help="Taxonomías para las redes (por defecto Discogs Genre, CDR Genre)")
    ap.add_argument("--no-pairwise", action="store_true", help="Omite las 21 comparaciones por pares (más rápido)")
    args = ap.parse_args(argv)

    res = pipeline.run_pipeline(path=args.data, n_perm=args.n_perm, sources=args.sources, pairwise=not args.no_pairwise)
    out = pipeline.save_results(res, args.out)
    m = res.meta
    print(f"Artefactos en {out}")
    print(f"  canciones: {m['n_raw']} (audio completo: {m['n_audio_complete']})")
    print(f"  PERMDISP: F={m['permdisp']['F']:.3f}  p={m['permdisp']['p_value']:.4f}  "
          f"eta²={m['permdisp']['eta_squared']:.4f}  → {m['permdisp']['decision']}")
    if m["pairwise_total"]:
        print(f"  pares significativos (Holm): {m['pairwise_significant']}/{m['pairwise_total']}")
    return 0
