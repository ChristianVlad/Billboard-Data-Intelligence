import pytest
from fastapi.testclient import TestClient

from billboard_intel import config, pipeline


@pytest.fixture(scope="module")
def client(results, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    pipeline.save_results(results, out)
    old = config.ARTIFACTS_DIR
    config.ARTIFACTS_DIR = out
    from api import main
    main._cache.update(key=None, results=None)
    yield TestClient(main.app)
    config.ARTIFACTS_DIR = old


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "results_available": True}


def test_summary_and_permdisp(client):
    assert client.get("/summary").json()["n_raw"] == 420
    g = client.get("/dispersion/permdisp").json()
    assert g["Decision"] in {"Reject H0", "Fail to reject H0"}


def test_pairwise_and_dispersion(client):
    assert len(client.get("/dispersion/pairwise").json()) == 21
    assert len(client.get("/dispersion").json()) == 7


def test_profile_filter_and_validation(client):
    assert {r["Feature"] for r in client.get("/profile", params={"feature": "Energy"}).json()} == {"Energy"}
    assert client.get("/profile", params={"feature": "Nope"}).status_code == 422


def test_network_json_and_html(client):
    j = client.get("/networks/Discogs Genre", params={"epoch": "2000s"}).json()
    assert j["nodes"] and all("Color" in n for n in j["nodes"])
    r = client.get("/networks/Discogs Genre/html", params={"epoch": "2000s", "min_weight": 2})
    assert r.status_code == 200 and "<script" in r.text and "vis" in r.text.lower()


def test_invalid_inputs(client):
    assert client.get("/networks/Discogs Genre", params={"epoch": "1850s"}).status_code == 422
    assert client.get("/networks/Unknown Source").status_code == 404
    assert client.get("/genres/Discogs Genre/associations", params={"sort_by": "bad"}).status_code == 422


def test_associations_filtering(client):
    rows = client.get("/genres/Discogs Genre/associations", params={"min_weight": 5, "limit": 3}).json()
    assert len(rows) <= 3 and all(r["Weight"] >= 5 for r in rows)


def test_excel_download(client):
    r = client.get("/export/excel")
    assert r.status_code == 200 and r.content[:2] == b"PK"


def test_no_results_returns_503(tmp_path):
    from api import main
    old = config.ARTIFACTS_DIR
    config.ARTIFACTS_DIR = tmp_path
    try:
        assert TestClient(main.app).get("/summary").status_code == 503
    finally:
        config.ARTIFACTS_DIR = old


def test_epoch_without_classified_songs_returns_404(tmp_path):
    """CDR Genre está vacío en 2020-2025 en el dataset real: la API debe explicarlo, no fallar."""
    from api import main
    from billboard_intel.pipeline import run_pipeline, save_results
    from billboard_intel.synthetic import make_synthetic
    raw = make_synthetic(n=300, seed=5)
    raw.loc[raw["Date"] >= "2020-01-01", "CDR Genre"] = None
    save_results(run_pipeline(raw=raw, n_perm=49, pairwise=False), tmp_path)
    old = config.ARTIFACTS_DIR
    config.ARTIFACTS_DIR = tmp_path
    main._cache.update(key=None, results=None)
    try:
        c = TestClient(main.app)
        r = c.get("/networks/CDR Genre", params={"epoch": "2020-2025"})
        assert r.status_code == 404 and "2020-2025" in r.json()["detail"]
        assert c.get("/networks/Discogs Genre", params={"epoch": "2020-2025"}).status_code == 200
    finally:
        config.ARTIFACTS_DIR = old
        main._cache.update(key=None, results=None)
