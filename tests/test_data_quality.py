import numpy as np
import pandas as pd
import pytest

from billboard_intel import data, quality


def test_prepare_does_not_mutate_raw(synthetic_raw):
    before = synthetic_raw.copy(deep=True)
    data.prepare(synthetic_raw)
    pd.testing.assert_frame_equal(synthetic_raw, before)


def test_missing_required_columns_raises(synthetic_raw):
    with pytest.raises(ValueError, match="Faltan columnas"):
        data.prepare(synthetic_raw.drop(columns=["Energy"]))


def test_epoch_assignment_boundaries():
    ep = data.assign_epoch(pd.Series([1958, 1969, 1970, 1999, 2000, 2019, 2020, 2025]))
    assert list(ep.astype(str)) == ["1958-1969", "1958-1969", "1970s", "1990s", "2000s", "2010s", "2020-2025", "2020-2025"]


def test_suspicious_bpm_only_nulled_in_analysis(synthetic_raw):
    df = data.prepare(synthetic_raw)
    analysis, audio_complete = data.analysis_sample(df)
    assert df["BPM_flagged"].sum() >= 1
    assert df.loc[3, "BPM"] == 16                      # df conserva el valor original
    assert np.isnan(analysis.loc[3, "BPM"])            # analysis lo anula
    assert 3 not in audio_complete.index               # y la fila sale de la muestra multivariada
    quality.validate_analysis_sample(synthetic_raw, audio_complete)


def test_logical_checks_skip_missing_columns(synthetic_raw):
    df = data.prepare(synthetic_raw).drop(columns=["Rating 1"])
    out = quality.logical_checks(df)
    assert (out["Estado"] == "omitido (columna ausente)").sum() >= 2
    assert out.loc[out["Chequeo"].str.startswith("Semanas"), "Violaciones"].iloc[0] == 0


def test_no_duplicates_in_synthetic(synthetic_raw):
    out = quality.duplicate_checks(data.prepare(synthetic_raw))
    assert out["Cantidad"].iloc[:2].sum() == 0
