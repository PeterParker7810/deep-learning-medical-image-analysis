from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def write_image():
    def _write(path: Path, color=(90, 110, 130)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color).save(path)
        return path
    return _write

