"""Reproduce los números del notebook original. Se omite si el Excel real no está en data/raw/."""
import pytest

from billboard_intel import config, data, dispersion, quality

pytestmark = [pytest.mark.golden,
              pytest.mark.skipif(not config.RAW_DATA_PATH.exists(), reason="falta el Excel real en data/raw/")]


@pytest.fixture(scope="module")
def sample():
    raw = data.load_raw()
    df = data.prepare(raw)
    analysis, audio_complete = data.analysis_sample(df)
    return raw, audio_complete


def test_sample_sizes(sample):
    raw, audio_complete = sample
    quality.validate_analysis_sample(raw, audio_complete, expected_raw=1177, expected_complete=1174)
    counts = audio_complete["Epoch"].value_counts().reindex(config.PERIOD_LABELS).tolist()
    assert counts == [226, 253, 229, 139, 129, 116, 82]


def test_global_permdisp_matches_notebook(sample):
    _, audio_complete = sample
    out = dispersion.run_dispersion(audio_complete, n_perm=999, pairwise=False)
    g = out["permdisp_global"].iloc[0]
    assert g["F"] == pytest.approx(8.579641, abs=1e-4)
    assert g["Eta_Squared"] == pytest.approx(0.042248, abs=1e-5)
    assert g["P_Value"] < 0.005 and g["Decision"] == "Reject H0"
