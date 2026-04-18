from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_keeps_gymnasium_in_legacy_extra_only() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    optional = data["project"]["optional-dependencies"]["legacy"]
    assert all("gymnasium[box2d]" not in dependency for dependency in dependencies)
    assert any("gymnasium[box2d]" in dependency for dependency in optional)
