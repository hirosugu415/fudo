"""自然を扱うモジュールが、人間側（シミュレーションの判断・表示など）を import していないこと（神の手が入り込まない構造）。"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "fudo"
FORBIDDEN = ("sim", "viewer", "agents", "market", "markets", "economy")


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield a.name


def test_nature_modules_do_not_import_human_side():
    for pkg in ("world", "nature"):
        for f in (SRC / pkg).glob("*.py"):
            for mod in _imports(f):
                parts = mod.split(".")
                assert not any(p in FORBIDDEN for p in parts), f"{f.name} が {mod} を import しています"
