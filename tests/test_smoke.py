"""Smoke tests — verify the package imports and config resolves.

Confirms the scaffold is wired correctly before any real logic lands.
"""

from courtvision import __version__, config


def test_version():
    assert __version__ == "0.1.0"


def test_config_paths_resolve():
    # DATA_DIR should live directly under the project root.
    assert config.DATA_DIR.parent == config.ROOT_DIR
    assert config.RAW_DIR.name == "raw"
