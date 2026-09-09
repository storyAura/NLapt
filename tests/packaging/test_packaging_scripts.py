"""Guard the one-click scripts against drifting from the pyproject extras.

``Launch-NLapt.bat`` and ``Build-NLapt.bat`` are the entry points the README
recommends to non-developers. They used to install only PySide6 + Pillow, so a
fresh machine got a GUI whose every LLM / translate / llama-server call failed
with "The 'httpx' package is required for LLM HTTP clients" — and a bundle
built there shipped without httpx because PyInstaller only warns about a
missing hidden import. These tests pin every runtime extra of ``pyproject.toml``
to both scripts and to the spec's build-time preflight.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
LAUNCH_BAT = ROOT / "Launch-NLapt.bat"
BUILD_BAT = ROOT / "Build-NLapt.bat"
SPEC = ROOT / "packaging" / "nlapt.spec"

RUNTIME_EXTRAS = ("gui", "images", "llm", "local")
# Distribution name (pip) -> top-level import name (python -c "import ...").
IMPORT_NAMES = {"Pillow": "PIL"}
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9_.-]+")


def _runtime_distributions() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    names: list[str] = []
    for extra in RUNTIME_EXTRAS:
        for requirement in extras[extra]:
            match = _REQUIREMENT_NAME.match(requirement)
            assert match, requirement
            names.append(match.group(0))
    return names


def _pip_install_lines(script: Path) -> list[str]:
    lines = script.read_text(encoding="ascii").splitlines()
    return [line for line in lines if "pip install" in line]


def _import_check_lines(script: Path) -> list[str]:
    lines = script.read_text(encoding="ascii").splitlines()
    return [line for line in lines if '-c "import ' in line]


@pytest.mark.parametrize("script", [LAUNCH_BAT, BUILD_BAT], ids=["launch", "build"])
def test_bat_installs_every_runtime_extra(script: Path) -> None:
    installed = " ".join(_pip_install_lines(script))
    missing = [name for name in _runtime_distributions() if name not in installed.split()]
    assert not missing, f"{script.name} never installs {missing}"


@pytest.mark.parametrize("script", [LAUNCH_BAT, BUILD_BAT], ids=["launch", "build"])
def test_bat_import_check_covers_every_runtime_extra(script: Path) -> None:
    checked = " ".join(_import_check_lines(script))
    expected = [IMPORT_NAMES.get(name, name) for name in _runtime_distributions()]
    missing = [name for name in expected if not re.search(rf"\b{name}\b", checked)]
    assert not missing, f"{script.name} import check skips {missing}"


def test_launch_bat_requires_httpx_before_starting() -> None:
    # httpx is a hard requirement (every LLM request), so it must sit in the
    # aborting install block, not only the best-effort optional one.
    text = LAUNCH_BAT.read_text(encoding="ascii")
    required_block = text.split("Optional local-inference", 1)[0]
    assert 'import PySide6, PIL, httpx"' in required_block
    assert "pip install PySide6 Pillow httpx" in required_block


def test_spec_preflight_lists_every_runtime_import() -> None:
    source = SPEC.read_text(encoding="utf-8")
    match = re.search(r"REQUIRED_RUNTIME_MODULES\s*=\s*\((.*?)\)", source, re.S)
    assert match, "nlapt.spec lost its REQUIRED_RUNTIME_MODULES preflight"
    listed = set(re.findall(r'"([^"]+)"', match.group(1)))
    expected = {IMPORT_NAMES.get(name, name) for name in _runtime_distributions()}
    assert expected <= listed, f"spec preflight misses {expected - listed}"
    assert "_preflight_runtime_modules()" in source
