import json

import pandas as pd
from openpyxl import load_workbook

from billboard_intel import pipeline


def test_meta_is_complete(results):
    m = results.meta
    assert m["n_raw"] == 420 and m["n_audio_complete"] <= 418
    assert m["pairwise_total"] == 21
    assert set(m["permdisp"]) == {"F", "p_value", "eta_squared", "decision"}


def test_save_and_load_roundtrip(results, tmp_path):
    out = pipeline.save_results(results, tmp_path)
    loaded = pipeline.load_results(out)
    assert set(loaded.tables) == set(results.tables)
    assert loaded.meta == json.loads((out / "meta.json").read_text())
    pd.testing.assert_frame_equal(loaded.tables["dispersion_summary"], results.tables["dispersion_summary"],
                                  check_dtype=False, check_categorical=False)


def test_excel_has_expected_sheets(results, tmp_path):
    out = pipeline.save_results(results, tmp_path)
    wb = load_workbook(out / "billboard_results.xlsx")
    assert {"Resumen", "PERMDISP_Global", "PERMDISP_Pares", "Coocurr_DiscogsGenre"} <= set(wb.sheetnames)
    assert all(len(n) <= 31 for n in wb.sheetnames)


def test_load_results_without_artifacts_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        pipeline.load_results(tmp_path)


def test_pipeline_from_xlsx(synthetic_xlsx):
    res = pipeline.run_pipeline(path=synthetic_xlsx, n_perm=49, pairwise=False)
    assert res.meta["dataset_sha256"] and "permdisp_pairwise" not in res.tables


def test_cli_end_to_end(synthetic_xlsx, tmp_path, capsys):
    from billboard_intel.cli import main
    assert main(["--data", str(synthetic_xlsx), "--out", str(tmp_path), "--n-perm", "49", "--no-pairwise"]) == 0
    assert (tmp_path / "meta.json").exists() and (tmp_path / "billboard_results.xlsx").exists()
    assert "PERMDISP" in capsys.readouterr().out
