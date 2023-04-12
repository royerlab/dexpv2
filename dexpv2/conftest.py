import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """pytest command line parser"""
    parser.addoption("--interactive", action="store_true", default=False)


@pytest.fixture()
def interactive_test(pytestconfig: pytest.Config) -> bool:
    """Interactive test fixture from pytest parser"""
    return pytestconfig.getoption("interactive")
