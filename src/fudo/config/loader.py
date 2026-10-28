"""設定ファイルの読み込み。

コード側に既定値を置かない（SPEC 原則6）。キーが欠けていたら ConfigError にする。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

GENERATOR_FILES = (
    "frame",
    "external",
    "geology",
    "erosion",
    "climate",
    "hydrology",
    "minerals",
    "forest",
    "fishery",
    "description",
    "soil",
)


BEHAVIOR_FILES = (
    "outside",
    "demography",
    "farming",
    "fishing",
    "households",
    "commerce",
    "transport",
    "land",
)


class ConfigError(KeyError):
    pass


class Section:
    """辞書を属性でたどれるようにした読み取り専用の包み。欠けたキーは場所つきでエラーにする。"""

    __slots__ = ("_data", "_path")

    def __init__(self, data: dict, path: str):
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: 辞書が必要です")
        self._data = data
        self._path = path

    def _wrap(self, value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return Section(value, path)
        if isinstance(value, list):
            return [self._wrap(v, f"{path}[{i}]") for i, v in enumerate(value)]
        return value

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self[key]

    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            raise ConfigError(f"設定 {self._path}.{key} がありません")
        return self._wrap(self._data[key], f"{self._path}.{key}")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return list(self._data.keys())

    def items(self):
        return [(k, self[k]) for k in self._data]

    @property
    def raw(self) -> dict:
        return self._data


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_dir() -> Path:
    return project_root() / "config"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: 中身が辞書ではありません")
    return data


def canonical_hash(obj: Any) -> str:
    """内容のハッシュ。キーの順番やYAMLのコメントには影響されない。"""
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_generator_config(config_dir: Path | None = None) -> Section:
    d = Path(config_dir) if config_dir else default_config_dir()
    data = {name: _load_yaml(d / "generator" / f"{name}.yaml") for name in GENERATOR_FILES}
    return Section(data, "generator")


def load_behavior_config(config_dir: Path | None = None) -> Section:
    """エージェントの行動の係数（run_id に含まれる）。"""
    d = Path(config_dir) if config_dir else default_config_dir()
    data = {name: _load_yaml(d / "behavior" / f"{name}.yaml") for name in BEHAVIOR_FILES}
    return Section(data, "behavior")


def load_simulation_config(config_dir: Path | None = None) -> Section:
    d = Path(config_dir) if config_dir else default_config_dir()
    return Section(_load_yaml(d / "simulation.yaml"), "simulation")


def load_physics_config(path: Path | None = None) -> Section:
    p = Path(path) if path else default_config_dir() / "physics" / "default.yaml"
    return Section(_load_yaml(p), "physics")


def load_storage_config(config_dir: Path | None = None) -> Section:
    d = Path(config_dir) if config_dir else default_config_dir()
    return Section(_load_yaml(d / "storage.yaml"), "storage")


def load_viewer_config(config_dir: Path | None = None) -> Section:
    d = Path(config_dir) if config_dir else default_config_dir()
    return Section(_load_yaml(d / "viewer.yaml"), "viewer")
