import importlib


def test_package_importable() -> None:
    assert importlib.import_module("interview_kit") is not None
