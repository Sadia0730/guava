from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import torch


CACHE_VERSION = 1


def tree_to_cpu(value: Any, floating_dtype=torch.float16) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.to(floating_dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: tree_to_cpu(item, floating_dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree_to_cpu(item, floating_dtype) for item in value)
    return value


def tree_index(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return value[index]
    if isinstance(value, dict):
        return {key: tree_index(item, index) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree_index(item, index) for item in value)
    return value


def tree_stack(values: list[Any]) -> Any:
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values)
    if isinstance(first, dict):
        return {key: tree_stack([value[key] for value in values]) for key in first}
    if first is None:
        return None
    if isinstance(first, (int, float, bool)):
        return torch.tensor(values)
    return values


def tree_to(value: Any, device: torch.device | str, dtype=torch.float32) -> Any:
    if torch.is_tensor(value):
        value = value.to(device)
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: tree_to(item, device, dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree_to(item, device, dtype) for item in value)
    return value


def file_fingerprint(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherCacheWriter:
    def __init__(self, root: str | Path, metadata: dict[str, Any] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, dict[str, Any]] = {}
        self.metadata = {"version": CACHE_VERSION, **(metadata or {})}
        self.shard_number = 0

    def add_shard(self, paths: Iterable[str | Path], outputs: dict[str, Any]) -> Path:
        paths = [str(Path(path).resolve()) for path in paths]
        shard_name = f"shard_{self.shard_number:06d}.pt"
        shard_path = self.root / shard_name
        torch.save(tree_to_cpu(outputs), shard_path)
        for offset, path in enumerate(paths):
            if path in self.index:
                raise ValueError(f"duplicate teacher-cache path: {path}")
            self.index[path] = {"shard": shard_name, "offset": offset}
        self.shard_number += 1
        return shard_path

    def close(self) -> Path:
        payload = {"metadata": {**self.metadata, "items": len(self.index)}, "items": self.index}
        path = self.root / "index.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class TeacherOutputCache:
    def __init__(self, root: str | Path, max_open_shards: int = 4):
        self.root = Path(root)
        payload = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if payload["metadata"].get("version") != CACHE_VERSION:
            raise ValueError("unsupported teacher cache version")
        self.metadata = payload["metadata"]
        self.index = payload["items"]
        self.max_open_shards = max_open_shards
        self._shards: OrderedDict[str, Any] = OrderedDict()

    def _shard(self, name: str) -> Any:
        if name in self._shards:
            self._shards.move_to_end(name)
            return self._shards[name]
        value = torch.load(self.root / name, map_location="cpu", weights_only=True)
        self._shards[name] = value
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return value

    def get(self, path: str | Path) -> dict[str, Any]:
        key = str(Path(path).resolve())
        if key not in self.index:
            raise KeyError(f"teacher output is not cached: {key}")
        location = self.index[key]
        return tree_index(self._shard(location["shard"]), int(location["offset"]))

    def batch(self, paths: Iterable[str | Path], device: torch.device | str) -> dict[str, Any]:
        values = [self.get(path) for path in paths]
        return tree_to(tree_stack(values), device)
