"""analytics/ may import only shared/, the standard library and third-party packages."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
FORBIDDEN = {"engine", "companion", "web", "api", "content"}


def test_no_cross_module_imports():
    bad = []
    for py in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            bad += [f"{py.name}: {n}" for n in names if n.split(".")[0] in FORBIDDEN]
    assert not bad, bad
