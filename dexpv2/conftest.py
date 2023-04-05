import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """pytest command line parser"""
    parser.addoption("--display", action="store_true", default=False)


@pytest.fixture()
def display_test(pytestconfig: pytest.Config) -> bool:
    """Display test fixture from pytest parser"""
    return pytestconfig.getoption("display")
