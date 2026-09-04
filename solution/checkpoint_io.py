from __future__ import annotations

import os
import pathlib
import pickle
from pathlib import Path
from typing import Any

import torch


def _resolve_path_class(module: str, name: str):
    if not name.endswith("Path"):
        raise AttributeError
    if os.name == "nt":
        if "Pure" in name or "Posix" in name:
            return pathlib.PurePosixPath
        return pathlib.PureWindowsPath
    if "Pure" in name or "Windows" in name:
        return pathlib.PureWindowsPath
    return pathlib.PurePosixPath


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module.startswith("pathlib"):
            try:
                return _resolve_path_class(module, name)
            except AttributeError:
                pass
        return super().find_class(module, name)


class _CompatPickleModule:
    Unpickler = _CompatUnpickler
    Pickler = pickle.Pickler
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def _is_pathlib_pickle_error(exc: BaseException) -> bool:
    msg = str(exc)
    if isinstance(exc, ModuleNotFoundError) and "pathlib" in msg:
        return True
    if isinstance(exc, AttributeError) and "pathlib" in msg:
        return True
    if "pathlib._local" in msg:
        return True
    if "PosixPath" in msg or "WindowsPath" in msg or "PurePosixPath" in msg or "PureWindowsPath" in msg:
        return True
    return False


def torch_load_compat(path: str | Path, *, map_location: Any = "cpu", weights_only: bool = False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as exc:
        if not _is_pathlib_pickle_error(exc):
            raise
    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
        pickle_module=_CompatPickleModule,
    )
