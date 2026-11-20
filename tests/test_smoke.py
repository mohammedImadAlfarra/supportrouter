import supportrouter


def test_package_imports_with_version() -> None:
    assert supportrouter.__version__ == "0.1.0"
