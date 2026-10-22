"""The package is layered by processing stage; imports may only point to the same or a lower layer.

Order (low to high): schema < data < features < models < training < activity < evaluation.
Scripts combine the layers and are not checked here.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "activity_index"
LAYERS = ("schema", "data", "features", "models", "training", "activity", "evaluation")
RANK = {name: i for i, name in enumerate(LAYERS)}


def layer_of(module: str) -> str:
    """``activity_index.data.pan_asgm`` -> ``data``."""
    return module.split(".")[1]


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_imports(path: Path) -> set[str]:
    """Absolute ``activity_index.*`` modules imported anywhere in the file (including inside functions)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: use absolute imports"
            names = [node.module] if node.module else []
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        found.update(n for n in names if n.startswith("activity_index."))
    return found


def modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        name = module_name(path)
        if name != "activity_index":
            yield name, path


def test_every_module_belongs_to_a_layer():
    for name, _ in modules():
        assert layer_of(name) in RANK, f"{name} is not in a known layer"


def test_imports_point_to_the_same_or_a_lower_layer():
    for name, path in modules():
        src = layer_of(name)
        for dst in package_imports(path):
            assert RANK[layer_of(dst)] <= RANK[src], f"{name} imports {dst} from a higher layer"


def test_models_and_estimator_only_use_the_schema():
    for name, path in modules():
        if layer_of(name) == "models":
            allowed = ("schema", "models")
        elif name == "activity_index.activity.estimator":
            allowed = ("schema",)
        else:
            continue
        for dst in package_imports(path):
            assert layer_of(dst) in allowed, f"{name} imports {dst}"
