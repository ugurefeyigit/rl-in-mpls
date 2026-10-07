"""Shared pytest configuration (kept out of pytest.ini, a V1-governed file)."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running equivalence tests (deselect with -m 'not slow')")
