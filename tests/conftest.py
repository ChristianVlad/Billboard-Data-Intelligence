import pytest

from billboard_intel import config
from billboard_intel.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_raw():
    return make_synthetic()


@pytest.fixture(scope="session")
def synthetic_xlsx(tmp_path_factory, synthetic_raw):
    path = tmp_path_factory.mktemp("data") / "synthetic.xlsx"
    synthetic_raw.to_excel(path, sheet_name=config.SHEET_NAME, index=False)
    return path


@pytest.fixture(scope="session")
def results(synthetic_raw):
    from billboard_intel.pipeline import run_pipeline
    return run_pipeline(raw=synthetic_raw, n_perm=199)
