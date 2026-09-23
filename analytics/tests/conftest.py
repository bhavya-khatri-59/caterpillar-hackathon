import pytest

from analytics.service import AnalyticsService


@pytest.fixture(scope="session")
def svc():
    return AnalyticsService(db_path=":memory:", seed_fixtures=True)
