import numpy as np
import pytest
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

from billboard_intel import dispersion as dp


def test_holm_matches_statsmodels():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.0001, 0.2])
    expected = multipletests(p, method="holm")[1]
    np.testing.assert_allclose(dp.holm(p), expected)


def test_anova_matches_scipy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    codes = rng.integers(0, 4, 200)
    d = dp._distances(X, codes, 4)
    f, eta = dp._anova(d, codes, 4)
    assert f == pytest.approx(f_oneway(*[d[codes == g] for g in range(4)]).statistic)
    assert 0 <= eta <= 1


def test_standardize_rejects_nan_and_constants():
    with pytest.raises(ValueError):
        dp.standardize(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        dp.standardize(np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]))


def test_permdisp_detects_real_difference_and_not_noise():
    rng = np.random.default_rng(2)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    same = rng.normal(size=(300, 3))
    diff = np.vstack([rng.normal(scale=0.5, size=(150, 3)), rng.normal(scale=1.5, size=(150, 3))])
    assert dp.permdisp(diff, labels, n_perm=499)["p_value"] < 0.01
    assert dp.permdisp(same, labels, n_perm=499)["p_value"] > 0.05


def test_permdisp_is_reproducible():
    rng = np.random.default_rng(3)
    X, labels = rng.normal(size=(120, 3)), np.array(["A", "B", "C"] * 40)
    a, b = dp.permdisp(X, labels, n_perm=99, seed=7), dp.permdisp(X, labels, n_perm=99, seed=7)
    assert a["F"] == b["F"] and a["p_value"] == b["p_value"]


def test_run_dispersion_on_synthetic(results):
    t = results.tables
    assert len(t["permdisp_pairwise"]) == 21
    assert t["permdisp_pairwise"]["P_Value_Holm"].between(0, 1).all()
    assert (t["permdisp_pairwise"]["P_Value_Holm"] >= t["permdisp_pairwise"]["P_Value"] - 1e-12).all()
    assert len(t["dispersion_songs"]) == results.meta["n_audio_complete"]
    # el dataset sintético se construyó con menor dispersión en 2000-2019
    s = t["dispersion_summary"].set_index("Epoch")["Mean_Distance"]
    assert s["2000s"] < s["1970s"]
    assert t["permdisp_global"]["Decision"].iloc[0] == "Reject H0"
