import importlib


def test_package_importable() -> None:
    assert importlib.import_module("interviewer") is not None
