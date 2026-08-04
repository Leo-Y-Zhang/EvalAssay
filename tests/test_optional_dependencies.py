"""The optional dependencies must stay optional.

The package installs with numpy and scipy and nothing else. Everything the
model-free layer does - loading a corpus, running the four detectors, auditing a
simulated model, rendering a report - works at that footprint, which is what
lets someone with no compute and no machine-learning stack check a benchmark.

That property is one careless top-level import away from being false, and the
failure is silent to anyone whose environment happens to have the library.
These tests read the source rather than trusting the environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).parent.parent / "src"

OPTIONAL = frozenset({"torch", "transformers", "pandas", "pyarrow", "anthropic", "sklearn"})
"""Libraries that must never be needed to import the package."""

REQUIRED = frozenset({"numpy", "scipy"})
"""Libraries the package genuinely depends on, declared in the project metadata."""


def _module_level_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Top-level imports that actually execute when the module is imported.

    Imports guarded by ``if TYPE_CHECKING`` do not run, so they are excluded.

    Args:
        tree: A parsed module.

    Returns:
        ``(line number, root module name)`` pairs.
    """
    found: list[tuple[int, str]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            found.extend((statement.lineno, alias.name.split(".")[0]) for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            found.append((statement.lineno, statement.module.split(".")[0]))
    return found


def _source_files() -> list[Path]:
    return sorted(SOURCE.rglob("*.py"))


def test_there_is_source_to_check() -> None:
    # Guards against a path change turning every test below into a vacuous pass.
    assert len(_source_files()) > 10


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_optional_dependency_is_imported_at_module_level(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [(line, name) for line, name in _module_level_imports(tree) if name in OPTIONAL]
    assert offenders == [], (
        f"{path.name} imports an optional dependency at module level: {offenders}. "
        "Move it inside the function that needs it, so the package still imports "
        "without it."
    )


def test_the_optional_dependencies_are_imported_somewhere() -> None:
    # The mirror of the rule above: a dependency declared and never imported is
    # supply-chain surface the project does not actually have.
    imported: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

    declared_and_used = {"torch", "transformers", "pandas", "anthropic"}
    assert declared_and_used <= imported, (
        f"declared but never imported: {sorted(declared_and_used - imported)}"
    )


def test_the_required_dependencies_are_the_only_ones_imported_at_module_level() -> None:
    third_party: set[str] = set()
    local_packages = {"evalassay", "__future__"}
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for _, name in _module_level_imports(tree):
            third_party.add(name)

    standard_library = {
        "argparse",
        "ast",
        "collections",
        "csv",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "platform",
        "re",
        "sys",
        "typing",
        "unicodedata",
    }
    external = third_party - local_packages - standard_library
    assert external <= REQUIRED, (
        f"module-level imports outside the required set: {sorted(external - REQUIRED)}"
    )
