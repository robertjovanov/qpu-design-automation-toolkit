from __future__ import annotations

"""
Notebook-friendly helpers for nextnano++ runs.

Typical usage in Jupyter:

    from src.nextnanopp_tools import *

    inp = load_input_file("device.in")
    inp = run_input_file(inp, outputdirectory="runs/my_device")

    plot_convergence("runs/my_device/bias_00000")
    plot_structure_linecut("runs/my_device", cut="1d_z_BG2")
    plot_bias_plane("runs/my_device", "bandedges", variable="HH")
    plot_quantum_occupation("runs/my_device", band="HH")

ASCII 3D .vtr files can be loaded directly by this module. nextnanopy is still
required for input execution, sweeps, and general DataFile integration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from itertools import permutations
import json
import mmap
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_PRODUCT = "nextnano++"
_LABEL_RE = re.compile(r"^(?P<name>[^\[]+?)(?:\[(?P<unit>[^\]]*)\])?$")
_REGION_COLUMN_RE = re.compile(r"^region_(?P<index>\d+)(?:\[(?P<unit>[^\]]*)\])?$")
_SWEEP_NUMBER_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SWEEP_RUN_COLUMNS = (
    "sweep_variable",
    "sweep_value",
    "run_root",
    "run_name",
    "value_source",
    "bias_dir",
    "complete",
    "required_outputs",
    "outputs_available",
    "error",
)


@dataclass(frozen=True)
class AxisData:
    name: str
    value: np.ndarray
    unit: str = ""
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VariableData:
    name: str
    value: np.ndarray
    unit: str = ""
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputDataset:
    path: Path
    coords: dict[str, AxisData]
    variables: dict[str, VariableData]
    source: str = "custom"

    @property
    def ndim(self) -> int:
        return len(self.coords)

    @property
    def coord_names(self) -> list[str]:
        return list(self.coords.keys())

    @property
    def variable_names(self) -> list[str]:
        return list(self.variables.keys())

    def get_coord(self, name: str) -> AxisData:
        key = _resolve_name(name, self.coords)
        return self.coords[key]

    def get_variable(self, name: str) -> VariableData:
        key = _resolve_name(name, self.variables)
        return self.variables[key]


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def structure(self) -> Path:
        return self.root / "Structure"

    @property
    def strain(self) -> Path:
        return self.root / "Strain"

    @property
    def quantum(self) -> Path:
        return self.bias0() / "Quantum"

    def bias_dirs(self) -> list[Path]:
        return get_bias_dirs(self.root)

    def bias0(self) -> Path:
        return get_bias_dir(self.root, bias=0)


@dataclass(frozen=True)
class FldMeta:
    ndim: int
    dims: tuple[int, ...]
    veclen: int
    labels: list[str]
    coord_skips: dict[int, int]
    var_skips: dict[int, int]
    lines: list[str]


def _import_nextnanopy():
    try:
        import nextnanopy as nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nextnanopy is required for nextnano++ input execution, sweeps, and "
            "DataFolder navigation. Install it in your notebook environment first."
        ) from exc
    return nn


def _import_matplotlib_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _import_plotly_go():
    import plotly.graph_objects as go

    return go


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _parse_label(label: str) -> tuple[str, str, str]:
    stripped = label.strip()
    match = _LABEL_RE.match(stripped)
    if not match:
        return stripped, "", stripped
    name = match.group("name").strip()
    unit = (match.group("unit") or "").strip()
    pretty = f"{name}[{unit}]" if unit else name
    return name, unit, pretty


def _unique_key(name: str, mapping: Mapping[str, Any]) -> str:
    if name not in mapping:
        return name
    idx = 2
    while f"{name}_{idx}" in mapping:
        idx += 1
    return f"{name}_{idx}"


def _resolve_name(name: str | None, mapping: Mapping[str, Any]) -> str:
    if name is None:
        raise ValueError(f"A name must be provided. Available names: {list(mapping)}")
    if name in mapping:
        return name
    lowered = name.lower()
    exact = [key for key in mapping if key.lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    contains = [key for key in mapping if lowered in key.lower()]
    if len(contains) == 1:
        return contains[0]
    if not contains:
        raise KeyError(f"'{name}' was not found. Available names: {list(mapping)}")
    raise KeyError(f"'{name}' is ambiguous. Matches: {contains}")


def sanitize_run_component(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    clean = clean.strip("._-")
    return clean or "run"


def make_timestamp_tag(
    timestamp: datetime | str | None = None,
    *,
    fmt: str = "%Y%m%d_%H%M%S",
) -> str:
    if isinstance(timestamp, str):
        return sanitize_run_component(timestamp)
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(fmt)


def build_run_name(
    input_path_or_stem: str | Path,
    *,
    tag: str | None = None,
    add_timestamp: bool = True,
    timestamp: datetime | str | None = None,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
    separator: str = "__",
) -> str:
    base = sanitize_run_component(_coerce_path(input_path_or_stem).stem)
    parts = [base]
    if add_timestamp:
        parts.append(make_timestamp_tag(timestamp, fmt=timestamp_fmt))
    if tag:
        parts.append(sanitize_run_component(tag))
    return separator.join(parts)


def _looks_like_run_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "simulation_info.txt").exists():
        return True
    if (path / "job_done.txt").exists():
        return True
    return any(child.is_dir() and child.name.startswith("bias_") for child in path.iterdir())


def resolve_run_root(run_root: str | Path) -> Path:
    path = _coerce_path(run_root)
    if not path.exists() or not path.is_dir():
        return path
    if _looks_like_run_root(path):
        return path

    children = [child for child in path.iterdir() if child.is_dir()]
    if len(children) == 1 and children[0].name == path.name and _looks_like_run_root(children[0]):
        return children[0]
    return path


def _ensure_unique_run_root(run_root: Path, *, separator: str = "__") -> Path:
    if not run_root.exists():
        return run_root
    index = 2
    while True:
        candidate = run_root.with_name(f"{run_root.name}{separator}{index:02d}")
        if not candidate.exists():
            return candidate
        index += 1


def find_runs_for_input(output_root: str | Path, input_path_or_stem: str | Path) -> list[Path]:
    root = _coerce_path(output_root)
    stem = sanitize_run_component(_coerce_path(input_path_or_stem).stem)
    if not root.exists():
        return []

    runs: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        resolved = resolve_run_root(path)
        name = resolved.name
        if name == stem or name.startswith(f"{stem}__"):
            runs.append(resolved)
    return sorted(set(runs), key=lambda item: item.name)


def find_latest_run(output_root: str | Path, input_path_or_stem: str | Path) -> Path:
    matches = find_runs_for_input(output_root, input_path_or_stem)
    if not matches:
        raise FileNotFoundError(
            f"No run folders matching '{_coerce_path(input_path_or_stem).stem}' were found under {output_root}"
        )
    return matches[-1]


def _iter_named_records(container: Any):
    if hasattr(container, "items"):
        for _, value in container.items():
            yield value
        return
    if hasattr(container, "values"):
        for value in container.values():
            yield value
        return
    for value in container:
        yield value


def _get_variable_data(dataset: OutputDataset, variable: str | None) -> VariableData:
    if variable is None:
        if len(dataset.variable_names) == 1:
            return dataset.variables[dataset.variable_names[0]]
        raise ValueError(
            f"Multiple variables are available in {dataset.path.name}: {dataset.variable_names}. "
            "Please specify variable='...'."
        )
    return dataset.get_variable(variable)


def _display_plotly_figure(fig):
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        fig.show()
    return fig


def _guess_coord_names(path: str | Path, ndim: int) -> list[str]:
    stem = _coerce_path(path).stem
    if "_1d_x_" in stem or stem.startswith("grid_x"):
        return ["x"]
    if "_1d_y_" in stem or stem.startswith("grid_y"):
        return ["y"]
    if "_1d_z_" in stem or stem.startswith("grid_z"):
        return ["z"]
    if "_2d_xy_" in stem:
        return ["x", "y"]
    if "_2d_xz_" in stem:
        return ["x", "z"]
    if "_2d_yz_" in stem:
        return ["y", "z"]
    if ndim == 1:
        return ["x"]
    if ndim == 2:
        return ["x", "z"]
    if ndim == 3:
        return ["x", "y", "z"]
    return [f"coord_{i}" for i in range(ndim)]


def _nearest_index(values: np.ndarray, target: float | None = None, index: int | None = None) -> int:
    if index is not None:
        if not 0 <= index < len(values):
            raise IndexError(f"Index {index} is outside the valid range 0..{len(values) - 1}")
        return int(index)
    if target is None:
        return int(len(values) // 2)
    return int(np.argmin(np.abs(values - target)))


def _as_2d_plot_array(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if values.shape == (len(x), len(y)):
        return values.T
    if values.shape == (len(y), len(x)):
        return values
    raise ValueError(
        f"Cannot interpret 2D variable shape {values.shape} against coordinate lengths "
        f"{len(x)} and {len(y)}."
    )


def _orient_nd_array(values: np.ndarray, coord_arrays: Sequence[np.ndarray]) -> np.ndarray:
    desired_shape = tuple(len(np.asarray(coord)) for coord in coord_arrays)
    if values.shape == desired_shape:
        return values
    if values.ndim != len(coord_arrays):
        raise ValueError(
            f"Variable has ndim={values.ndim}, but {len(coord_arrays)} coordinates were supplied."
        )

    matching_permutations = [
        perm for perm in permutations(range(values.ndim)) if tuple(values.shape[idx] for idx in perm) == desired_shape
    ]
    if not matching_permutations:
        raise ValueError(f"Cannot orient shape {values.shape} to coordinate shape {desired_shape}.")
    if len(matching_permutations) > 1:
        raise ValueError(
            f"Ambiguous axis orientation for shape {values.shape} and coordinates {desired_shape}: "
            f"{matching_permutations}"
        )
    permutation = matching_permutations[0]
    if permutation == tuple(range(values.ndim)):
        return values
    return np.transpose(values, axes=permutation)


def _maybe_log10(values: np.ndarray, log10: bool) -> np.ndarray:
    if not log10:
        return values
    clipped = np.where(values > 0, values, np.nan)
    return np.log10(clipped)


def get_run_paths(run_root: str | Path) -> RunPaths:
    return RunPaths(resolve_run_root(run_root))


def get_bias_dirs(run_root: str | Path) -> list[Path]:
    root = resolve_run_root(run_root)
    if root.name.startswith("bias_") and root.is_dir():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("bias_"))


def get_bias_dir(run_root: str | Path, bias: str | int | None = None) -> Path:
    root = resolve_run_root(run_root)
    if root.name.startswith("bias_") and root.is_dir():
        return root

    bias_dirs = get_bias_dirs(root)
    if not bias_dirs:
        raise FileNotFoundError(f"No bias_XXXXX folders were found in {root}")

    if bias is None or bias == "latest":
        return bias_dirs[-1]
    if bias == "first":
        return bias_dirs[0]
    if isinstance(bias, int):
        exact = root / f"bias_{bias:05d}"
        if exact.exists():
            return exact
        if 0 <= bias < len(bias_dirs):
            return bias_dirs[bias]
        raise FileNotFoundError(f"Could not resolve bias {bias!r} under {root}")

    exact = root / bias
    if exact.exists():
        return exact
    if bias.isdigit():
        numbered = root / f"bias_{int(bias):05d}"
        if numbered.exists():
            return numbered
    raise FileNotFoundError(f"Could not resolve bias {bias!r} under {root}")


def find_run_files(run_root: str | Path, keyword: str, deep: bool = True) -> list[Path]:
    root = resolve_run_root(run_root)
    matcher = root.rglob if deep else root.glob
    keyword_lower = keyword.lower()
    return sorted(path for path in matcher("*") if keyword_lower in path.name.lower())


def describe_output_file(
    path: str | Path,
    *,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
) -> dict[str, Any]:
    dataset = load_output_file(path, product=product, prefer_nextnanopy=prefer_nextnanopy)
    return {
        "path": str(dataset.path),
        "source": dataset.source,
        "ndim": dataset.ndim,
        "coords": {
            name: {
                "shape": tuple(np.asarray(axis.value).shape),
                "unit": axis.unit,
                "label": axis.label,
            }
            for name, axis in dataset.coords.items()
        },
        "variables": {
            name: {
                "shape": tuple(np.asarray(var.value).shape),
                "unit": var.unit,
                "label": var.label,
            }
            for name, var in dataset.variables.items()
        },
    }


def read_dat_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(_coerce_path(path), sep=r"\s+", comment="#")


def read_index_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(_coerce_path(path), sep=r"\s+", comment="#")


def _load_with_nextnanopy(path: Path, product: str) -> OutputDataset:
    nn = _import_nextnanopy()
    datafile = nn.DataFile(str(path), product=product)
    guessed_coord_names = _guess_coord_names(path, len(list(_iter_named_records(datafile.coords))))

    coords: dict[str, AxisData] = {}
    for idx, coord in enumerate(_iter_named_records(datafile.coords)):
        guessed_name = guessed_coord_names[idx] if idx < len(guessed_coord_names) else str(coord.name)
        unit = str(getattr(coord, "unit", "") or "")
        original_name = str(coord.name)
        original_label = str(getattr(coord, "label", "") or coord.name)
        label = f"{guessed_name}[{unit}]" if unit else guessed_name
        key = _unique_key(guessed_name, coords)
        coords[key] = AxisData(
            name=guessed_name,
            value=np.asarray(coord.value),
            unit=unit,
            label=label,
            metadata={
                **dict(getattr(coord, "metadata", {}) or {}),
                "original_name": original_name,
                "original_label": original_label,
            },
        )

    variables: dict[str, VariableData] = {}
    for variable in _iter_named_records(datafile.variables):
        key = _unique_key(str(variable.name), variables)
        variables[key] = VariableData(
            name=str(variable.name),
            value=np.asarray(variable.value),
            unit=str(getattr(variable, "unit", "") or ""),
            label=str(getattr(variable, "label", "") or variable.name),
            metadata=dict(getattr(variable, "metadata", {}) or {}),
        )

    return OutputDataset(path=path, coords=coords, variables=variables, source="nextnanopy")


def _fld_read_floats(lines: list[str], start_line: int, count: int) -> np.ndarray:
    values: list[float] = []
    line_idx = start_line
    while line_idx < len(lines) and len(values) < count:
        stripped = lines[line_idx].strip()
        if stripped and not stripped.startswith("#"):
            values.extend(float(token) for token in stripped.split())
        line_idx += 1
    if len(values) < count:
        raise ValueError(f"Expected {count} floats starting at line {start_line}, got {len(values)}")
    return np.asarray(values[:count], dtype=float)


def parse_fld(path: str | Path, max_lines: int = 2500) -> FldMeta:
    path = _coerce_path(path)
    lines = path.read_text(errors="ignore").splitlines()
    header = "\n".join(lines[:max_lines])

    def get_int(key: str) -> int:
        match = re.search(rf"^{key}\s*=\s*(\d+)\s*$", header, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing '{key} = ...' in {path.name}")
        return int(match.group(1))

    ndim = get_int("ndim")
    field_match = re.search(r"^field\s*=\s*(\w+)\s*$", header, flags=re.MULTILINE)
    if not field_match or field_match.group(1) != "rectilinear":
        got = field_match.group(1) if field_match else None
        raise ValueError(f"Only field=rectilinear is supported, got {got!r} in {path.name}")

    dims = tuple(get_int(f"dim{i}") for i in range(1, ndim + 1))
    veclen = get_int("veclen")
    labels = re.findall(r"^label\s*=\s*(.+)\s*$", header, flags=re.MULTILINE)

    coord_skips: dict[int, int] = {}
    for idx in range(1, ndim + 1):
        match = re.search(rf"^coord\s+{idx}\s+.*?\bskip=(\d+)\b", header, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing 'coord {idx} ... skip=' in {path.name}")
        coord_skips[idx] = int(match.group(1))

    var_skips: dict[int, int] = {}
    for idx in range(1, veclen + 1):
        match = re.search(rf"^variable\s+{idx}\s+.*?\bskip=(\d+)\b", header, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing 'variable {idx} ... skip=' in {path.name}")
        var_skips[idx] = int(match.group(1))

    return FldMeta(
        ndim=ndim,
        dims=dims,
        veclen=veclen,
        labels=labels,
        coord_skips=coord_skips,
        var_skips=var_skips,
        lines=lines,
    )


def _load_dat_dataset(path: Path) -> OutputDataset:
    df = read_dat_table(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    columns = list(df.columns)
    coord_name, coord_unit, coord_label = _parse_label(columns[0])
    coords = {
        coord_name: AxisData(
            name=coord_name,
            value=df.iloc[:, 0].to_numpy(dtype=float),
            unit=coord_unit,
            label=coord_label,
        )
    }

    variables: dict[str, VariableData] = {}
    for col_idx, column in enumerate(columns[1:], start=1):
        name, unit, label = _parse_label(column)
        key = _unique_key(name or f"var_{col_idx}", variables)
        variables[key] = VariableData(
            name=name or key,
            value=df.iloc[:, col_idx].to_numpy(dtype=float),
            unit=unit,
            label=label,
        )

    return OutputDataset(path=path, coords=coords, variables=variables, source="custom_dat")


def _load_fld_dataset(path: Path) -> OutputDataset:
    meta = parse_fld(path)
    coord_names = _guess_coord_names(path, meta.ndim)

    coords: dict[str, AxisData] = {}
    for idx, (coord_name, dim) in enumerate(zip(coord_names, meta.dims), start=1):
        key = _unique_key(coord_name, coords)
        values = _fld_read_floats(meta.lines, meta.coord_skips[idx], dim)
        coords[key] = AxisData(
            name=coord_name,
            value=values,
            unit="nm",
            label=f"{coord_name}[nm]",
        )

    variables: dict[str, VariableData] = {}
    value_count = int(np.prod(meta.dims))
    for idx in range(1, meta.veclen + 1):
        label = meta.labels[idx - 1] if idx - 1 < len(meta.labels) else f"var_{idx}"
        name, unit, pretty = _parse_label(label)
        key = _unique_key(name or f"var_{idx}", variables)
        raw = _fld_read_floats(meta.lines, meta.var_skips[idx], value_count)
        variables[key] = VariableData(
            name=name or key,
            value=raw.reshape(meta.dims, order="F"),
            unit=unit,
            label=pretty,
        )

    return OutputDataset(path=path, coords=coords, variables=variables, source="custom_fld")


def _parse_vtr_extent(extent_text: str) -> tuple[int, ...]:
    values = [int(token) for token in str(extent_text).split()]
    if len(values) % 2 != 0 or not values:
        raise ValueError(f"Invalid VTR extent: {extent_text!r}")
    return tuple(values[idx + 1] - values[idx] + 1 for idx in range(0, len(values), 2))


def _match_vtr_variable_name(raw_name: str, requests: set[str] | None) -> bool:
    if requests is None:
        return True
    parsed_name, _, _ = _parse_label(raw_name)
    candidates = {
        raw_name.strip().lower(),
        parsed_name.strip().lower(),
    }
    return any(candidate in requests for candidate in candidates if candidate)


def _is_vtr_coordinate_name(name: str) -> bool:
    return name.strip().upper() in {"X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"}


def _iter_vtr_dataarray_names_ascii(path: str | Path) -> list[str]:
    file_path = _coerce_path(path)
    pattern = re.compile(rb"<DataArray\b[^>]*\bName\s*=\s*['\"]([^'\"]+)['\"][^>]*>")
    with file_path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            return [match.group(1).decode("utf-8", errors="replace").strip() for match in pattern.finditer(mm)]


def _resolve_vtr_dataarray_name(path: str | Path, requested_name: str | None) -> str:
    names = [name for name in _iter_vtr_dataarray_names_ascii(path) if not _is_vtr_coordinate_name(name)]
    if not names:
        raise ValueError(f"No PointData variables were found in {_coerce_path(path)}")
    if requested_name is None:
        return names[0]

    requested = str(requested_name).strip().lower()
    exact = [name for name in names if name.strip().lower() == requested]
    if len(exact) == 1:
        return exact[0]

    parsed_matches = []
    contains_matches = []
    for name in names:
        parsed, _, _ = _parse_label(name)
        parsed_lower = parsed.strip().lower()
        raw_lower = name.strip().lower()
        if parsed_lower == requested:
            parsed_matches.append(name)
        elif requested in parsed_lower or requested in raw_lower:
            contains_matches.append(name)

    if len(parsed_matches) == 1:
        return parsed_matches[0]
    if len(parsed_matches) > 1:
        raise KeyError(f"'{requested_name}' is ambiguous. Matches: {parsed_matches}")
    if len(contains_matches) == 1:
        return contains_matches[0]
    if contains_matches:
        raise KeyError(f"'{requested_name}' is ambiguous. Matches: {contains_matches}")
    raise KeyError(f"'{requested_name}' was not found in {_coerce_path(path).name}. Available variables: {names}")


def _find_vtr_dataarray_body_ascii(mm: mmap.mmap, name: str) -> tuple[int, int]:
    escaped = re.escape(name.encode("utf-8"))
    pattern = re.compile(rb"<DataArray\b[^>]*\bName\s*=\s*['\"]" + escaped + rb"['\"][^>]*>")
    match = pattern.search(mm)
    if match is None:
        raise KeyError(f"DataArray named {name!r} was not found.")
    body_start = match.end()
    body_end = mm.find(b"</DataArray>", body_start)
    if body_end < 0:
        raise ValueError(f"DataArray named {name!r} is missing its closing tag.")
    return body_start, body_end


def _read_vtr_coord_arrays_ascii(path: str | Path) -> dict[str, np.ndarray]:
    file_path = _coerce_path(path)
    coords: dict[str, np.ndarray] = {}
    with file_path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            for coord_name, key in (
                ("X_COORDINATES", "x"),
                ("Y_COORDINATES", "y"),
                ("Z_COORDINATES", "z"),
            ):
                body_start, body_end = _find_vtr_dataarray_body_ascii(mm, coord_name)
                coords[key] = np.fromstring(mm[body_start:body_end], sep=" ", dtype=float)
    missing = [name for name in ("x", "y", "z") if name not in coords or coords[name].size == 0]
    if missing:
        raise ValueError(f"Missing coordinate arrays {missing} in {file_path}")
    return coords


def _read_mmap_float_window(
    mm: mmap.mmap,
    *,
    body_start: int,
    body_end: int,
    start_value: int,
    count: int,
) -> np.ndarray:
    if start_value < 0:
        raise ValueError("start_value must be non-negative.")
    if count <= 0:
        raise ValueError("count must be positive.")

    mm.seek(body_start)
    skip_remaining = int(start_value)
    values: list[float] = []

    while mm.tell() < body_end and len(values) < count:
        line = mm.readline()
        if not line:
            break
        line_end = mm.tell()
        if line_end > body_end:
            line = line[: len(line) - (line_end - body_end)]
        tokens = line.split()
        if not tokens:
            continue

        if skip_remaining >= len(tokens):
            skip_remaining -= len(tokens)
            continue
        if skip_remaining:
            tokens = tokens[skip_remaining:]
            skip_remaining = 0

        needed = count - len(values)
        values.extend(float(token) for token in tokens[:needed])

    if len(values) != count:
        raise ValueError(
            f"Expected {count} values from VTR DataArray window, got {len(values)} "
            f"after skipping {start_value} values."
        )
    return np.asarray(values, dtype=float)


def load_vtr_plane(
    path: str | Path,
    *,
    variable: str | None,
    slice_axis: str = "z",
    slice_value: float | None = None,
    slice_index: int | None = None,
) -> dict[str, Any]:
    """
    Load one plane from an ASCII rectilinear-grid ``.vtr`` file.

    This avoids loading the full 3D volume when slicing along the last coordinate
    axis, which is the common nextnano++ use case for xy planes at a fixed z.
    For non-contiguous x/y slices it falls back to the full selected-variable
    loader and then extracts the plane.
    """
    file_path = _coerce_path(path)
    raw_variable_name = _resolve_vtr_dataarray_name(file_path, variable)
    coords = _read_vtr_coord_arrays_ascii(file_path)
    coord_names = [name for name in ("x", "y", "z") if name in coords]
    slice_name = _resolve_name(slice_axis, {name: coords[name] for name in coord_names})

    if slice_name != coord_names[-1]:
        dataset = _load_vtr_dataset(file_path, variable_names=[raw_variable_name])
        return extract_plane(
            dataset,
            variable=raw_variable_name,
            slice_axis=slice_name,
            slice_value=slice_value,
            slice_index=slice_index,
            prefer_nextnanopy=False,
        )

    slice_coord = coords[slice_name]
    slice_idx = _nearest_index(slice_coord, target=slice_value, index=slice_index)
    remaining_names = [name for name in coord_names if name != slice_name]
    remaining_shape = tuple(len(coords[name]) for name in remaining_names)
    value_count = int(np.prod(remaining_shape))
    start_value = int(slice_idx * value_count)

    with file_path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            body_start, body_end = _find_vtr_dataarray_body_ascii(mm, raw_variable_name)
            raw_plane = _read_mmap_float_window(
                mm,
                body_start=body_start,
                body_end=body_end,
                start_value=start_value,
                count=value_count,
            )

    name, unit, pretty = _parse_label(raw_variable_name)
    variable_data = VariableData(
        name=name or raw_variable_name,
        value=raw_plane.reshape(remaining_shape, order="F"),
        unit=unit,
        label=pretty,
    )
    x_name, y_name = remaining_names
    return {
        "path": file_path,
        "variable": variable_data,
        "slice_axis": slice_name,
        "slice_index": int(slice_idx),
        "slice_coordinate": float(slice_coord[slice_idx]),
        "requested_slice_coordinate": slice_value,
        "x_name": x_name,
        "y_name": y_name,
        "x": coords[x_name],
        "y": coords[y_name],
        "x_label": f"{x_name}[nm]",
        "y_label": f"{y_name}[nm]",
        "values": variable_data.value,
    }


def load_vtr_linecut(
    path: str | Path,
    *,
    variable: str | None,
    axis: str = "x",
    fixed_coords: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Load one line from an ASCII VTR file without reading the full volume.

    nextnano writes rectilinear VTR point data with the first coordinate
    (normally ``x``) varying fastest. Lines along that coordinate are one
    contiguous window in the ASCII DataArray and are read with mmap. Other
    line orientations fall back to the general full-volume extractor.
    """
    file_path = _coerce_path(path)
    raw_variable_name = _resolve_vtr_dataarray_name(file_path, variable)
    coords = _read_vtr_coord_arrays_ascii(file_path)
    coord_names = [name for name in ("x", "y", "z") if name in coords]
    axis_name = _resolve_name(axis, {name: coords[name] for name in coord_names})

    if axis_name != coord_names[0]:
        dataset = _load_vtr_dataset(file_path, variable_names=[raw_variable_name])
        return extract_linecut(
            dataset,
            variable=raw_variable_name,
            axis=axis_name,
            fixed_coords=fixed_coords,
            prefer_nextnanopy=False,
        )

    chosen_coords: dict[str, float] = {}
    chosen_indices: dict[str, int] = {}
    start_value = 0
    stride = len(coords[axis_name])
    for coord_name in coord_names[1:]:
        target = None if fixed_coords is None else fixed_coords.get(coord_name)
        coord_idx = _nearest_index(coords[coord_name], target=target)
        chosen_coords[coord_name] = float(coords[coord_name][coord_idx])
        chosen_indices[coord_name] = int(coord_idx)
        start_value += int(coord_idx * stride)
        stride *= len(coords[coord_name])

    with file_path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            body_start, body_end = _find_vtr_dataarray_body_ascii(mm, raw_variable_name)
            values = _read_mmap_float_window(
                mm,
                body_start=body_start,
                body_end=body_end,
                start_value=start_value,
                count=len(coords[axis_name]),
            )

    name, unit, pretty = _parse_label(raw_variable_name)
    variable_data = VariableData(
        name=name or raw_variable_name,
        value=values,
        unit=unit,
        label=pretty,
    )
    return {
        "path": file_path,
        "axis_name": axis_name,
        "axis": coords[axis_name],
        "axis_label": f"{axis_name}[nm]",
        "requested_coords": dict(fixed_coords or {}),
        "chosen_coords": chosen_coords,
        "chosen_indices": chosen_indices,
        "variable": variable_data,
        "values": values,
    }


def _as_xy_value_array(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if values.shape == (len(x), len(y)):
        return values
    if values.shape == (len(y), len(x)):
        return values.T
    raise ValueError(
        f"Cannot interpret 2D variable shape {values.shape} against coordinate lengths "
        f"{len(x)} and {len(y)}."
    )


def _iter_xy_point_records(
    points: pd.DataFrame | Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    x_col: str,
    y_col: str,
) -> Iterable[tuple[int, Any, float, float]]:
    if isinstance(points, pd.DataFrame):
        records = points.reset_index(drop=True).to_dict("records")
    else:
        records = list(points)

    for index, point in enumerate(records, start=1):
        if isinstance(point, Mapping):
            label = point.get("peak", point.get("point", index))
            try:
                x_nm = float(point[x_col])
                y_nm = float(point[y_col])
            except KeyError as exc:
                raise KeyError(f"Point record is missing coordinate column '{exc.args[0]}'") from exc
        else:
            values = list(point)
            if len(values) < 2:
                raise ValueError("Point sequences must contain at least x and y coordinates.")
            label = index
            x_nm = float(values[0])
            y_nm = float(values[1])
        yield index, label, x_nm, y_nm


def sample_vtr_plane_at_points(
    path: str | Path,
    *,
    variable: str | None,
    slice_axis: str = "z",
    slice_value: float | None = None,
    slice_index: int | None = None,
    points: pd.DataFrame | Sequence[Mapping[str, Any] | Sequence[Any]],
    x_col: str = "x_nm",
    y_col: str = "y_nm",
) -> pd.DataFrame:
    """
    Sample the nearest grid value of one VTR plane at requested x-y points.

    This is useful for comparing a 2D profile, such as the HH band edge at a
    fixed depth, with peak coordinates extracted from another output file.
    """
    plane = load_vtr_plane(
        path,
        variable=variable,
        slice_axis=slice_axis,
        slice_value=slice_value,
        slice_index=slice_index,
    )
    x = np.asarray(plane["x"], dtype=float)
    y = np.asarray(plane["y"], dtype=float)
    values = _as_xy_value_array(np.asarray(plane["values"], dtype=float), x, y)
    variable_data = plane["variable"]
    variable_label = variable_data.label or variable_data.name
    unit_suffix = f"[{variable_data.unit}]" if variable_data.unit else ""
    value_column = f"{variable_data.name}{unit_suffix}" if variable_data.name else "value"
    slice_name = str(plane["slice_axis"])
    slice_coordinate = float(plane["slice_coordinate"])

    rows: list[dict[str, Any]] = []
    for _, label, requested_x_nm, requested_y_nm in _iter_xy_point_records(points, x_col=x_col, y_col=y_col):
        ix = _nearest_index(x, target=requested_x_nm)
        iy = _nearest_index(y, target=requested_y_nm)
        rows.append(
            {
                "point": label,
                "requested_x_nm": requested_x_nm,
                "requested_y_nm": requested_y_nm,
                "nearest_x_nm": float(x[ix]),
                "nearest_y_nm": float(y[iy]),
                f"{slice_name}_nm": slice_coordinate,
                value_column: float(values[ix, iy]),
                "value": float(values[ix, iy]),
                "value_label": variable_label,
                "ix": int(ix),
                "iy": int(iy),
                "slice_axis": slice_name,
                "slice_index": int(plane["slice_index"]),
                "path": str(plane["path"]),
            }
        )
    return pd.DataFrame(rows)


def find_vtr_plane_extrema(
    path: str | Path,
    *,
    variable: str | None,
    slice_axis: str = "z",
    slice_value: float | None = None,
    slice_index: int | None = None,
    kind: str = "max",
    n_extrema: int = 2,
    min_lateral_separation_nm: float = 50.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Find separated minima or maxima on one VTR plane.

    Extrema are selected by value order with lateral non-maximum suppression in
    the plotted x-y plane, which prevents neighboring mesh points on one feature
    from being reported as independent extrema.
    """
    if n_extrema <= 0:
        raise ValueError("n_extrema must be positive.")
    if min_lateral_separation_nm < 0:
        raise ValueError("min_lateral_separation_nm must be non-negative.")

    normalized_kind = kind.lower()
    if normalized_kind in {"maximum", "maxima"}:
        normalized_kind = "max"
    elif normalized_kind in {"minimum", "minima"}:
        normalized_kind = "min"
    if normalized_kind not in {"min", "max"}:
        raise ValueError("kind must be 'min' or 'max'.")

    plane = load_vtr_plane(
        path,
        variable=variable,
        slice_axis=slice_axis,
        slice_value=slice_value,
        slice_index=slice_index,
    )
    x = np.asarray(plane["x"], dtype=float)
    y = np.asarray(plane["y"], dtype=float)
    values = _as_xy_value_array(np.asarray(plane["values"], dtype=float), x, y)
    variable_data = plane["variable"]
    variable_label = variable_data.label or variable_data.name
    unit_suffix = f"[{variable_data.unit}]" if variable_data.unit else ""
    value_column = f"{variable_data.name}{unit_suffix}" if variable_data.name else "value"
    slice_name = str(plane["slice_axis"])
    slice_coordinate = float(plane["slice_coordinate"])

    mask = np.isfinite(values)
    if xlim is not None:
        x_min, x_max = sorted((float(xlim[0]), float(xlim[1])))
        mask &= (x[:, None] >= x_min) & (x[:, None] <= x_max)
    if ylim is not None:
        y_min, y_max = sorted((float(ylim[0]), float(ylim[1])))
        mask &= (y[None, :] >= y_min) & (y[None, :] <= y_max)

    ranked_values = values.copy()
    ranked_values[~mask] = np.nan
    flat_order = np.argsort(ranked_values.ravel())
    if normalized_kind == "max":
        flat_order = flat_order[::-1]

    selected: list[dict[str, Any]] = []
    for flat_index in flat_order:
        value = float(ranked_values.ravel()[flat_index])
        if not np.isfinite(value):
            continue
        ix, iy = np.unravel_index(int(flat_index), values.shape)
        x_nm = float(x[ix])
        y_nm = float(y[iy])

        if any(
            np.hypot(x_nm - extremum["x_nm"], y_nm - extremum["y_nm"]) < min_lateral_separation_nm
            for extremum in selected
        ):
            continue

        selected.append(
            {
                "extremum": len(selected) + 1,
                "kind": normalized_kind,
                "x_nm": x_nm,
                "y_nm": y_nm,
                f"{slice_name}_nm": slice_coordinate,
                value_column: value,
                "value": value,
                "value_label": variable_label,
                "ix": int(ix),
                "iy": int(iy),
                "slice_axis": slice_name,
                "slice_index": int(plane["slice_index"]),
                "path": str(plane["path"]),
            }
        )
        if len(selected) >= n_extrema:
            break

    if len(selected) < n_extrema:
        raise ValueError(
            f"Found only {len(selected)} finite separated extrema; requested {n_extrema}. "
            "Try lowering min_lateral_separation_nm or widening xlim/ylim."
        )

    return pd.DataFrame(selected)


def _load_vtr_dataset(
    path: str | Path,
    *,
    variable_names: Sequence[str] | None = None,
) -> OutputDataset:
    from xml.etree import ElementTree as ET

    file_path = _coerce_path(path)
    requested = {str(name).strip().lower() for name in variable_names} if variable_names is not None else None

    grid_dims: tuple[int, ...] | None = None
    coord_arrays: dict[str, np.ndarray] = {}
    point_arrays: list[tuple[str, np.ndarray]] = []
    section: str | None = None

    try:
        for event, elem in ET.iterparse(file_path, events=("start", "end")):
            tag = elem.tag.rsplit("}", 1)[-1]
            if event == "start":
                if tag == "RectilinearGrid" and grid_dims is None:
                    extent_text = elem.attrib.get("WholeExtent")
                    if extent_text:
                        grid_dims = _parse_vtr_extent(extent_text)
                elif tag == "Piece" and grid_dims is None:
                    extent_text = elem.attrib.get("Extent")
                    if extent_text:
                        grid_dims = _parse_vtr_extent(extent_text)
                elif tag == "Coordinates":
                    section = "Coordinates"
                elif tag == "PointData":
                    section = "PointData"
            else:
                if tag == "DataArray" and section == "Coordinates":
                    raw_name = str(elem.attrib.get("Name", "")).strip().upper()
                    if raw_name.endswith("X_COORDINATES"):
                        coord_key = "x"
                    elif raw_name.endswith("Y_COORDINATES"):
                        coord_key = "y"
                    elif raw_name.endswith("Z_COORDINATES"):
                        coord_key = "z"
                    else:
                        coord_key = f"coord_{len(coord_arrays)}"
                    coord_arrays[coord_key] = np.fromstring(elem.text or "", sep=" ", dtype=float)
                elif tag == "DataArray" and section == "PointData":
                    raw_name = str(elem.attrib.get("Name", "")).strip()
                    if _match_vtr_variable_name(raw_name, requested):
                        values = np.fromstring(elem.text or "", sep=" ", dtype=float)
                        point_arrays.append((raw_name, values))
                elif tag in {"Coordinates", "PointData"}:
                    section = None
                elem.clear()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse ASCII VTR file {file_path}") from exc

    ordered_coord_names = [name for name in ("x", "y", "z") if name in coord_arrays]
    ordered_coord_names.extend(name for name in coord_arrays if name not in ordered_coord_names)
    if not ordered_coord_names:
        raise ValueError(f"No coordinate arrays were found in {file_path}")

    coord_dims = tuple(len(coord_arrays[name]) for name in ordered_coord_names)
    if grid_dims is not None and tuple(grid_dims[: len(coord_dims)]) != coord_dims:
        raise ValueError(
            f"VTR extent {grid_dims} does not match coordinate lengths {coord_dims} in {file_path}"
        )

    coords: dict[str, AxisData] = {}
    for coord_name in ordered_coord_names:
        key = _unique_key(coord_name, coords)
        coords[key] = AxisData(
            name=coord_name,
            value=coord_arrays[coord_name],
            unit="nm",
            label=f"{coord_name}[nm]",
        )

    expected_value_count = int(np.prod(coord_dims))
    variables: dict[str, VariableData] = {}
    for raw_name, raw_values in point_arrays:
        if raw_values.size != expected_value_count:
            raise ValueError(
                f"Variable {raw_name!r} in {file_path.name} has {raw_values.size} values, "
                f"expected {expected_value_count} from coordinates {coord_dims}"
            )
        name, unit, pretty = _parse_label(raw_name)
        key = _unique_key(name or raw_name, variables)
        variables[key] = VariableData(
            name=name or key,
            value=raw_values.reshape(coord_dims, order="F"),
            unit=unit,
            label=pretty,
        )

    if requested is not None and not variables:
        available = _list_vtr_variable_names(file_path)
        raise KeyError(
            f"Requested VTR variables {list(variable_names or [])} were not found in {file_path.name}. "
            f"Available variables: {available}"
        )
    if not variables:
        raise ValueError(f"No PointData arrays were loaded from {file_path}")

    return OutputDataset(path=file_path, coords=coords, variables=variables, source="custom_vtr")


def _list_vtr_variable_names(path: str | Path) -> list[str]:
    from xml.etree import ElementTree as ET

    file_path = _coerce_path(path)
    variables: dict[str, None] = {}
    inside_point_data = False

    try:
        for event, elem in ET.iterparse(file_path, events=("start", "end")):
            tag = elem.tag.rsplit("}", 1)[-1]
            if event == "start":
                if tag == "PointData":
                    inside_point_data = True
                elif inside_point_data and tag == "DataArray":
                    raw_name = str(elem.attrib.get("Name", "")).strip()
                    if raw_name:
                        name, _, _ = _parse_label(raw_name)
                        key = _unique_key(name or raw_name, variables)
                        variables[key] = None
            else:
                if tag == "PointData":
                    inside_point_data = False
                elem.clear()
    except ET.ParseError as exc:
        if variables:
            return list(variables)
        raise ValueError(f"Could not parse VTR variable metadata from {file_path}") from exc

    if not variables:
        raise ValueError(f"No PointData variables were found in {file_path}")
    return list(variables)


def load_output_file(
    path: str | Path,
    *,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
) -> OutputDataset:
    file_path = _coerce_path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".vtr":
        try:
            return _load_vtr_dataset(file_path)
        except Exception:
            if prefer_nextnanopy:
                try:
                    return _load_with_nextnanopy(file_path, product=product)
                except ModuleNotFoundError:
                    pass
            raise

    if prefer_nextnanopy:
        try:
            return _load_with_nextnanopy(file_path, product=product)
        except ModuleNotFoundError:
            pass
        except Exception:
            pass

    if suffix == ".dat":
        return _load_dat_dataset(file_path)
    if suffix == ".fld":
        return _load_fld_dataset(file_path)
    if suffix == ".vtr":
        raise ModuleNotFoundError(
            "Loading .vtr files requires nextnanopy in the current Python environment. "
            "Install nextnanopy and call load_output_file(..., prefer_nextnanopy=True)."
        )
    raise ValueError(f"Unsupported file type '{suffix}' for {file_path}")


def list_variables(path: str | Path, *, product: str = DEFAULT_PRODUCT) -> list[str]:
    file_path = _coerce_path(path)
    if file_path.suffix.lower() == ".vtr":
        return _list_vtr_variable_names(file_path)
    return load_output_file(file_path, product=product).variable_names


def resolve_bias_output_file(
    run_root: str | Path,
    quantity: str,
    *,
    cut: str | None = None,
    bias: str | int | None = None,
    preferred_extensions: Sequence[str] = ("vtr", "fld", "dat"),
) -> Path:
    bias_dir = get_bias_dir(run_root, bias=bias)
    stem = f"{quantity}_{cut}" if cut else quantity
    for extension in preferred_extensions:
        candidate = bias_dir / f"{stem}.{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find output for '{stem}' in {bias_dir}")


def resolve_structure_file(
    run_root: str | Path,
    quantity: str,
    *,
    cut: str | None = None,
    preferred_extensions: Sequence[str] = ("fld", "dat", "vtr"),
) -> Path:
    structure_dir = get_run_paths(run_root).structure
    stem = f"{quantity}_{cut}" if cut else quantity
    for extension in preferred_extensions:
        candidate = structure_dir / f"{stem}.{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find structure output for '{stem}' in {structure_dir}")


def resolve_quantum_output_file(
    run_root: str | Path,
    quantity: str,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    cut: str | None = None,
    kpoint: str | None = None,
    bias: str | int | None = None,
    preferred_extensions: Sequence[str] = ("fld", "dat", "vtr"),
) -> Path:
    stem = f"{quantity}_{cut}" if cut else quantity
    bias_dir = get_bias_dir(run_root, bias=bias)
    quantum_root = bias_dir / "Quantum"
    nested_dir = quantum_root / region / band

    candidate_stems = [stem]
    flat_stem = f"{quantity}_{region}_{band}"
    if cut:
        flat_stem = f"{flat_stem}_{cut}"
    candidate_stems.append(flat_stem)

    if kpoint:
        kpoint_stem = f"{quantity}_{kpoint}"
        if cut:
            kpoint_stem = f"{kpoint_stem}_{cut}"
        candidate_stems.append(kpoint_stem)
        candidate_stems.append(f"{quantity}_{region}_{band}_{kpoint}")
        if cut:
            candidate_stems.append(f"{quantity}_{region}_{band}_{kpoint}_{cut}")

    kpoint_match = re.match(r"(?P<base>.+)_k(?P<digits>\d+)$", quantity)
    if kpoint_match:
        base = kpoint_match.group("base")
        digits = kpoint_match.group("digits")
        flat_without_kpoint = f"{base}_{region}_{band}"
        if cut:
            flat_without_kpoint = f"{flat_without_kpoint}_{cut}"
        candidate_stems.append(flat_without_kpoint)
        candidate_stems.append(f"{base}_{region}_{band}_{digits}")

    candidates: list[Path] = []
    for extension in preferred_extensions:
        candidates.append(nested_dir / f"{stem}.{extension}")
        for candidate_stem in candidate_stems:
            candidates.append(quantum_root / f"{candidate_stem}.{extension}")
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find quantum output for '{stem}'. Searched:\n  {searched}")


def resolve_quantum_probability_state_file(
    run_root: str | Path,
    *,
    state: int = 1,
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str | None = "k00000",
    shifted: bool = True,
    bias: str | int | None = None,
    preferred_extensions: Sequence[str] = ("vtr",),
) -> Path:
    if state < 1:
        raise ValueError("state must be a positive 1-based state index.")

    bias_dir = get_bias_dir(run_root, bias=bias)
    quantum_root = bias_dir / "Quantum"
    nested_dir = quantum_root / region / band
    prefix = "probability_shift" if shifted else "probability"
    state_token = f"{state:04d}"
    nested_stems = [f"{prefix}_{state_token}"]
    if kpoint:
        nested_stems.insert(0, f"{prefix}_{kpoint}_{state_token}")
    flat_stems = [f"{prefix}_{region}_{band}_{state_token}"]
    if kpoint:
        flat_stems.append(f"{prefix}_{region}_{band}_{kpoint}_{state_token}")

    candidates: list[Path] = []
    for extension in preferred_extensions:
        candidates.extend(nested_dir / f"{stem}.{extension}" for stem in nested_stems)
        candidates.extend(quantum_root / f"{stem}.{extension}" for stem in flat_stems)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find quantum probability output for state {state}. Searched:\n  {searched}")


def load_input_file(input_path: str | Path, *, configpath: str | Path | None = None):
    nn = _import_nextnanopy()
    kwargs: dict[str, Any] = {}
    if configpath is not None:
        kwargs["configpath"] = str(configpath)
    return nn.InputFile(str(_coerce_path(input_path)), **kwargs)


def set_input_variables(input_file_or_path: Any, variables: Mapping[str, Any]):
    input_file = input_file_or_path
    if not hasattr(input_file, "set_variable"):
        input_file = load_input_file(input_file_or_path)
    for name, value in variables.items():
        input_file.set_variable(name, value=value)
    return input_file


def save_input_file(
    input_file_or_path: Any,
    *,
    fullpath: str | Path | None = None,
    overwrite: bool = False,
    automkdir: bool = True,
    content: bool = False,
    temp: bool = False,
):
    input_file = input_file_or_path
    if not hasattr(input_file, "save"):
        input_file = load_input_file(input_file_or_path)
    kwargs: dict[str, Any] = {
        "overwrite": overwrite,
        "automkdir": automkdir,
        "content": content,
        "temp": temp,
    }
    if fullpath is not None:
        kwargs["fullpath"] = str(_coerce_path(fullpath))
    return input_file.save(**kwargs)


def get_output_directory(input_file: Any) -> Path:
    execute_info = getattr(input_file, "execute_info", None)
    if isinstance(execute_info, Mapping) and "outputdirectory" in execute_info:
        return resolve_run_root(execute_info["outputdirectory"])
    folder_output = getattr(input_file, "folder_output", None)
    if callable(folder_output):
        return resolve_run_root(folder_output())
    if folder_output is not None:
        return resolve_run_root(folder_output)
    raise AttributeError("The supplied InputFile object does not expose folder_output().")


def run_input_file(
    input_file_or_path: Any,
    *,
    outputdirectory: str | Path | None = None,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    tag: str | None = None,
    add_timestamp: bool = True,
    timestamp: datetime | str | None = None,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
    separator: str = "__",
    staging_root: str | Path | None = None,
    keep_staged_input: bool = False,
    variables: Mapping[str, Any] | None = None,
    show_log: bool = True,
    convergenceCheck: bool = True,
    save_fullpath: str | Path | None = None,
    save_overwrite: bool = True,
    execute_kwargs: Mapping[str, Any] | None = None,
):
    input_file = input_file_or_path
    if not hasattr(input_file, "execute"):
        input_file = load_input_file(input_file_or_path)

    if output_root is not None and outputdirectory is not None:
        root_a = _coerce_path(output_root)
        root_b = _coerce_path(outputdirectory)
        if root_a != root_b:
            raise ValueError("Pass only one of output_root or outputdirectory, or make them identical.")

    original_fullpath = _coerce_path(getattr(input_file, "fullpath"))
    if variables:
        set_input_variables(input_file, variables)
    if save_fullpath is not None:
        save_input_file(input_file, fullpath=save_fullpath, overwrite=save_overwrite)
        original_fullpath = _coerce_path(getattr(input_file, "fullpath"))

    if output_root is None:
        output_root = outputdirectory
    if output_root is None:
        config = getattr(input_file, "config", None)
        product = getattr(input_file, "product", None)
        if config is None or product is None:
            raise ValueError("Could not infer output_root from the input file. Pass output_root explicitly.")
        output_root = config.get(section=product, option="outputdirectory")

    output_root_path = _coerce_path(output_root).expanduser().resolve()
    desired_run_name = run_name or build_run_name(
        original_fullpath,
        tag=tag,
        add_timestamp=add_timestamp,
        timestamp=timestamp,
        timestamp_fmt=timestamp_fmt,
        separator=separator,
    )
    desired_run_name = sanitize_run_component(desired_run_name)
    planned_run_root = _ensure_unique_run_root(output_root_path / desired_run_name, separator=separator)
    final_run_name = planned_run_root.name

    cleanup_staging_dir = False
    if staging_root is None:
        staged_dir = Path(tempfile.mkdtemp(prefix="nextnanopp_input_"))
        cleanup_staging_dir = True
    else:
        staged_dir = _coerce_path(staging_root)
        staged_dir.mkdir(parents=True, exist_ok=True)
    staged_input_path = staged_dir / f"{final_run_name}{original_fullpath.suffix}"
    save_input_file(input_file, fullpath=staged_input_path, overwrite=True)

    kwargs = dict(execute_kwargs or {})
    kwargs.setdefault("show_log", show_log)
    kwargs.setdefault("convergenceCheck", convergenceCheck)
    kwargs["outputdirectory"] = str(output_root_path)

    execution_failed = False
    try:
        input_file.execute(**kwargs)
        if isinstance(getattr(input_file, "execute_info", None), Mapping):
            input_file.execute_info["source_inputfile"] = str(original_fullpath)
            input_file.execute_info["staged_inputfile"] = str(staged_input_path)
            input_file.execute_info["run_name"] = final_run_name
            input_file.execute_info["output_root"] = str(output_root_path)
    except Exception:
        execution_failed = True
        raise
    finally:
        input_file.fullpath = str(original_fullpath)
        if cleanup_staging_dir and not keep_staged_input and not execution_failed:
            shutil.rmtree(staged_dir, ignore_errors=True)

    return input_file


def build_sweep(input_path: str | Path, sweep_variables: Mapping[str, Sequence[Any]]):
    nn = _import_nextnanopy()
    return nn.Sweep({key: list(values) for key, values in sweep_variables.items()}, str(_coerce_path(input_path)))


def run_sweep(
    input_path: str | Path,
    sweep_variables: Mapping[str, Sequence[Any]],
    *,
    delete_old_files: bool = True,
    delete_input_files: bool = False,
    overwrite: bool = True,
    show_log: bool = True,
    convergenceCheck: bool = True,
    parallel_limit: int = 1,
    execute_kwargs: Mapping[str, Any] | None = None,
):
    sweep = build_sweep(input_path, sweep_variables)
    sweep.save_sweep(delete_old_files=delete_old_files)

    kwargs = dict(execute_kwargs or {})
    kwargs.update(
        delete_input_files=delete_input_files,
        overwrite=overwrite,
        show_log=show_log,
        convergenceCheck=convergenceCheck,
        parallel_limit=parallel_limit,
    )
    sweep.execute_sweep(**kwargs)
    return sweep


def find_sweep_subrun(sweep_root: str | Path, **variable_values: Any) -> Path:
    root = _coerce_path(sweep_root)
    matches = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        if all(f"__{name}_{value}_" in folder.name for name, value in variable_values.items()):
            matches.append(folder)
    if not matches:
        raise FileNotFoundError(f"No sweep subrun matched {variable_values} under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple sweep subruns matched {variable_values}: {[m.name for m in matches]}")
    return matches[0]


def _metadata_sweep_value(run_root: Path, sweep_variable: str) -> tuple[float | None, str | None]:
    metadata_path = run_root / "variables_input.txt"
    if not metadata_path.is_file():
        return None, None

    variable = re.escape(sweep_variable)
    variable_reference = re.compile(
        rf"^\s*\$\s*{variable}(?![A-Za-z0-9_])"
    )
    assignment = re.compile(
        rf"^\s*\$\s*{variable}\s*=\s*(?P<value>{_SWEEP_NUMBER_TOKEN})\s*(?:#.*)?$"
    )

    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return None, f"Could not read {metadata_path}: {exc}"

    values: list[float] = []
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if variable_reference.match(line) is None:
            continue
        match = assignment.match(line)
        if match is None:
            errors.append(
                f"Malformed assignment for '{sweep_variable}' in {metadata_path} "
                f"at line {line_number}: {line.strip()!r}"
            )
            continue

        value = float(match.group("value"))
        if not np.isfinite(value):
            errors.append(
                f"Non-finite assignment for '{sweep_variable}' in {metadata_path} "
                f"at line {line_number}: {line.strip()!r}"
            )
            continue
        values.append(value)

    unique_values = set(values)
    if len(unique_values) > 1:
        rendered = ", ".join(f"{value:g}" for value in sorted(unique_values))
        errors.append(
            f"Conflicting duplicate assignments for '{sweep_variable}' in "
            f"{metadata_path}: {rendered}"
        )
        return None, "; ".join(errors)
    if unique_values:
        return values[0], "; ".join(errors) or None
    return None, "; ".join(errors) or None


def _folder_sweep_value(folder_name: str, sweep_variable: str) -> float | None:
    if "__" not in folder_name:
        return None
    sweep_suffix = folder_name.rsplit("__", maxsplit=1)[1]

    token = re.compile(
        rf"{re.escape(sweep_variable)}_(?P<value>{_SWEEP_NUMBER_TOKEN})(?=_|$)"
    )
    preceding_number = re.compile(rf"(?:^|_){_SWEEP_NUMBER_TOKEN}$")

    values: list[float] = []
    for match in token.finditer(sweep_suffix):
        start = match.start()
        has_primary_boundary = start == 0
        has_later_variable_boundary = (
            start > 0
            and sweep_suffix[start - 1] == "_"
            and preceding_number.search(sweep_suffix[: start - 1]) is not None
        )
        if not has_primary_boundary and not has_later_variable_boundary:
            continue

        value = float(match.group("value"))
        if not np.isfinite(value):
            raise ValueError(
                f"Folder {folder_name!r} contains a non-finite value for "
                f"'{sweep_variable}'."
            )
        values.append(value)

    unique_values = set(values)
    if len(unique_values) > 1:
        rendered = ", ".join(f"{value:g}" for value in sorted(unique_values))
        raise ValueError(
            f"Folder {folder_name!r} contains conflicting values for "
            f"'{sweep_variable}': {rendered}"
        )
    return values[0] if values else None


def _find_required_outputs(
    run_root: Path,
    bias_dir: Path | None,
    required_outputs: Sequence[str | Path],
) -> dict[str, Path | None]:
    resolved_outputs: dict[str, Path | None] = {}
    for requested_output in required_outputs:
        key = str(requested_output)
        requested_path = _coerce_path(requested_output).expanduser()
        if requested_path.is_absolute():
            candidates = [requested_path]
        else:
            candidates = []
            if bias_dir is not None:
                candidates.append(bias_dir / requested_path)
            candidates.append(run_root / requested_path)

        resolved_outputs[key] = next(
            (
                candidate.resolve()
                for candidate in candidates
                if candidate.resolve().is_file()
            ),
            None,
        )
    return resolved_outputs


def discover_sweep_runs(
    sweep_root: str | Path,
    sweep_variable: str,
    *,
    bias: str | int | None = "first",
    required_outputs: Sequence[str | Path] = (),
    strict: bool = True,
) -> pd.DataFrame:
    """Discover and describe the immediate run directories in a nextnano sweep.

    Exact assignments in ``variables_input.txt`` take precedence over the
    strict ``<variable>_<number>`` folder token used by nextnanopy. Relative
    required outputs are checked beneath the selected bias first and then the
    run root; absolute outputs are checked directly. Strict mode raises for
    malformed, missing, or duplicate values and missing biases, while
    non-strict mode retains rows and records those problems in ``error``.
    """
    if not isinstance(sweep_variable, str) or not sweep_variable:
        raise ValueError("sweep_variable must be a non-empty string.")

    root = _coerce_path(sweep_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sweep root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Sweep root is not a directory: {root}")

    rows: list[dict[str, Any]] = []
    candidates = sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ),
        key=lambda child: child.name,
    )

    for candidate in candidates:
        run_root = resolve_run_root(candidate).resolve()
        errors: list[str] = []
        sweep_value: float = np.nan
        value_source: str | None = None

        metadata_value, metadata_error = _metadata_sweep_value(run_root, sweep_variable)
        if metadata_error is not None:
            contextual_error = f"Run '{run_root.name}': {metadata_error}"
            if strict:
                raise ValueError(contextual_error)
            errors.append(contextual_error)

        if metadata_value is not None:
            sweep_value = metadata_value
            value_source = "variables_input"
        else:
            try:
                folder_value = _folder_sweep_value(candidate.name, sweep_variable)
            except ValueError as exc:
                contextual_error = f"Run '{run_root.name}': {exc}"
                if strict:
                    raise ValueError(contextual_error) from exc
                errors.append(contextual_error)
                folder_value = None

            if folder_value is not None:
                sweep_value = folder_value
                value_source = "folder_name"
            elif not errors:
                contextual_error = (
                    f"Run '{run_root.name}': could not determine sweep variable "
                    f"'{sweep_variable}' from variables_input.txt or the folder name."
                )
                if strict:
                    raise ValueError(contextual_error)
                errors.append(contextual_error)

        try:
            bias_dir = get_bias_dir(run_root, bias=bias).resolve()
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            contextual_error = (
                f"Run '{run_root.name}': could not select bias {bias!r}: {exc}"
            )
            if strict:
                raise FileNotFoundError(contextual_error) from exc
            errors.append(contextual_error)
            bias_dir = None

        output_paths = _find_required_outputs(run_root, bias_dir, required_outputs)
        rows.append(
            {
                "sweep_variable": sweep_variable,
                "sweep_value": sweep_value,
                "run_root": run_root,
                "run_name": run_root.name,
                "value_source": value_source,
                "bias_dir": bias_dir,
                "complete": (run_root / "job_done.txt").is_file(),
                "required_outputs": output_paths,
                "outputs_available": all(path is not None for path in output_paths.values()),
                "error": "; ".join(errors) or None,
            }
        )

    frame = pd.DataFrame(rows, columns=_SWEEP_RUN_COLUMNS)
    frame["sweep_value"] = pd.to_numeric(frame["sweep_value"], errors="coerce")
    frame["complete"] = frame["complete"].astype(bool)
    frame["outputs_available"] = frame["outputs_available"].astype(bool)

    duplicate_mask = frame["sweep_value"].notna() & frame["sweep_value"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_groups = frame.loc[duplicate_mask].groupby("sweep_value", sort=True)
        details = []
        for value, group in duplicate_groups:
            run_names = ", ".join(group["run_name"].tolist())
            details.append(f"{value:g}: {run_names}")
        duplicate_error = (
            f"Duplicate sweep values for '{sweep_variable}' under {root}: "
            + "; ".join(details)
        )
        if strict:
            raise ValueError(duplicate_error)
        for index in frame.index[duplicate_mask]:
            existing_error = frame.at[index, "error"]
            frame.at[index, "error"] = (
                f"{existing_error}; {duplicate_error}"
                if existing_error
                else duplicate_error
            )

    return frame.sort_values(
        "sweep_value",
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def load_sweep_outputs(
    runs: pd.DataFrame,
    output: str | Path,
    *,
    variable: str | None = None,
    prefer_nextnanopy: bool = True,
) -> pd.DataFrame:
    """Load one exact output path for every row in a sweep manifest.

    Relative outputs resolve beneath ``bias_dir`` first and then ``run_root``;
    absolute paths are checked directly. Every input row is preserved when an
    output is missing or fails to load. When supplied, ``variable`` is
    validated against the normalized dataset returned by ``load_output_file``.
    The returned copy adds
    ``requested_output``, ``output_path``, ``output_available``, ``dataset``,
    and ``load_error`` columns.
    """
    if not isinstance(runs, pd.DataFrame):
        raise TypeError("runs must be a pandas DataFrame.")
    if "run_root" not in runs.columns:
        raise ValueError("runs must contain a 'run_root' column.")
    if not str(output).strip():
        raise ValueError("output must be a non-empty path.")

    requested_output = _coerce_path(output).expanduser()
    if requested_output == Path("."):
        raise ValueError("output must be a non-empty file path.")

    result = runs.copy(deep=True)
    if result.empty:
        result["requested_output"] = pd.Series(index=result.index, dtype=object)
        result["output_path"] = pd.Series(index=result.index, dtype=object)
        result["output_available"] = pd.Series(index=result.index, dtype=bool)
        result["dataset"] = pd.Series(index=result.index, dtype=object)
        result["load_error"] = pd.Series(index=result.index, dtype=object)
        return result

    requested_outputs: list[Path] = []
    output_paths: list[Path | None] = []
    output_availability: list[bool] = []
    datasets: list[OutputDataset | None] = []
    load_errors: list[str | None] = []
    has_bias_dir = "bias_dir" in result.columns

    for _, row in result.iterrows():
        requested_outputs.append(requested_output)
        found_path: Path | None = None

        try:
            if requested_output.is_absolute():
                absolute_path = requested_output.resolve()
                found_path = absolute_path if absolute_path.is_file() else None
            else:
                run_root = _coerce_path(row["run_root"]).expanduser().resolve()
                bias_dir: Path | None = None
                if has_bias_dir:
                    raw_bias_dir = row["bias_dir"]
                    is_missing_bias = raw_bias_dir is None or raw_bias_dir is pd.NA
                    if not is_missing_bias:
                        try:
                            is_missing_bias = bool(pd.isna(raw_bias_dir))
                        except (TypeError, ValueError):
                            is_missing_bias = False
                    if not is_missing_bias:
                        bias_dir = _coerce_path(raw_bias_dir).expanduser().resolve()

                found_path = _find_required_outputs(
                    run_root,
                    bias_dir,
                    (requested_output,),
                )[str(requested_output)]
        except Exception as exc:
            output_paths.append(None)
            output_availability.append(False)
            datasets.append(None)
            load_errors.append(
                f"Could not resolve output {requested_output}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        output_paths.append(found_path)
        if found_path is None:
            output_availability.append(False)
            datasets.append(None)
            load_errors.append(f"Output not found: {requested_output}")
            continue

        output_availability.append(True)
        try:
            dataset = load_output_file(
                found_path,
                prefer_nextnanopy=prefer_nextnanopy,
            )
            if variable is not None:
                dataset.get_variable(variable)
        except Exception as exc:
            datasets.append(None)
            load_errors.append(
                f"Failed to load {found_path}: {type(exc).__name__}: {exc}"
            )
            continue

        datasets.append(dataset)
        load_errors.append(None)

    result["requested_output"] = requested_outputs
    result["output_path"] = output_paths
    result["output_available"] = output_availability
    result["dataset"] = datasets
    result["load_error"] = load_errors
    return result


def extract_linecut(
    dataset_or_path: OutputDataset | str | Path,
    *,
    variable: str | None = None,
    axis: str,
    fixed_coords: Mapping[str, float] | None = None,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
) -> dict[str, Any]:
    dataset = (
        dataset_or_path
        if isinstance(dataset_or_path, OutputDataset)
        else load_output_file(dataset_or_path, product=product, prefer_nextnanopy=prefer_nextnanopy)
    )
    variable_data = _get_variable_data(dataset, variable)
    coord_names = dataset.coord_names
    axis_name = _resolve_name(axis, dataset.coords)
    oriented_values = _orient_nd_array(
        np.asarray(variable_data.value),
        [np.asarray(dataset.coords[name].value) for name in coord_names],
    )

    indexer: list[Any] = []
    chosen_coords: dict[str, float] = {}
    chosen_indices: dict[str, int] = {}
    for coord_name in coord_names:
        coord_values = np.asarray(dataset.coords[coord_name].value)
        if coord_name == axis_name:
            indexer.append(slice(None))
            continue
        target = None if fixed_coords is None else fixed_coords.get(coord_name)
        coord_idx = _nearest_index(coord_values, target=target)
        chosen_coords[coord_name] = float(coord_values[coord_idx])
        chosen_indices[coord_name] = int(coord_idx)
        indexer.append(coord_idx)

    line = oriented_values[tuple(indexer)]
    if line.ndim != 1:
        raise ValueError(
            f"Expected a 1D line cut after fixing all other coordinates, got shape {line.shape}"
        )

    axis_data = dataset.coords[axis_name]
    return {
        "path": dataset.path,
        "axis_name": axis_name,
        "axis": np.asarray(axis_data.value),
        "axis_label": axis_data.label or axis_name,
        "requested_coords": dict(fixed_coords or {}),
        "chosen_coords": chosen_coords,
        "chosen_indices": chosen_indices,
        "variable": variable_data,
        "values": line,
    }


def extract_plane(
    dataset_or_path: OutputDataset | str | Path,
    *,
    variable: str | None = None,
    slice_axis: str,
    slice_value: float | None = None,
    slice_index: int | None = None,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
) -> dict[str, Any]:
    dataset = (
        dataset_or_path
        if isinstance(dataset_or_path, OutputDataset)
        else load_output_file(dataset_or_path, product=product, prefer_nextnanopy=prefer_nextnanopy)
    )
    if dataset.ndim != 3:
        raise ValueError(f"extract_plane requires a 3D dataset, got ndim={dataset.ndim}")

    variable_data = _get_variable_data(dataset, variable)
    coord_names = dataset.coord_names
    oriented_values = _orient_nd_array(
        np.asarray(variable_data.value),
        [np.asarray(dataset.coords[name].value) for name in coord_names],
    )
    slice_name = _resolve_name(slice_axis, dataset.coords)
    slice_axis_idx = coord_names.index(slice_name)
    slice_coord = dataset.coords[slice_name]
    slice_idx = _nearest_index(np.asarray(slice_coord.value), target=slice_value, index=slice_index)

    plane = np.take(oriented_values, slice_idx, axis=slice_axis_idx)
    remaining_names = [name for name in coord_names if name != slice_name]
    if len(remaining_names) != 2:
        raise ValueError(f"Expected 2 remaining axes after slicing, got {remaining_names}")

    x_axis = dataset.coords[remaining_names[0]]
    y_axis = dataset.coords[remaining_names[1]]
    return {
        "path": dataset.path,
        "variable": variable_data,
        "slice_axis": slice_name,
        "slice_index": slice_idx,
        "slice_coordinate": float(np.asarray(slice_coord.value)[slice_idx]),
        "x_name": remaining_names[0],
        "y_name": remaining_names[1],
        "x": np.asarray(x_axis.value),
        "y": np.asarray(y_axis.value),
        "x_label": x_axis.label or remaining_names[0],
        "y_label": y_axis.label or remaining_names[1],
        "values": plane,
    }


def _line_title(path: Path, provided: str | None, variable_names: Sequence[str]) -> str:
    if provided:
        return provided
    if len(variable_names) == 1:
        return f"{path.name}: {variable_names[0]}"
    return path.name


def _heatmap_title(path: Path, provided: str | None, variable_name: str) -> str:
    return provided or f"{path.name}: {variable_name}"


def _normalize_plot_markers(markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for marker in markers or []:
        if isinstance(marker, Mapping):
            value = marker.get("value")
            label = marker.get("label", "")
            color = marker.get("color", "rgba(80,80,80,0.85)")
            linestyle = marker.get("linestyle", marker.get("dash", "dash"))
            linewidth = marker.get("linewidth", marker.get("width", 1.3))
            alpha = marker.get("alpha", 0.85)
        else:
            values = list(marker)
            value = values[0] if values else None
            label = values[1] if len(values) > 1 else ""
            color = values[2] if len(values) > 2 else "rgba(80,80,80,0.85)"
            linestyle = "dash"
            linewidth = 1.3
            alpha = 0.85
        if value is None:
            continue
        normalized.append(
            {
                "value": float(value),
                "label": str(label),
                "color": color,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
            }
        )
    return normalized


def _matplotlib_color(color: str) -> str:
    if color.startswith("rgba("):
        channels = color.removeprefix("rgba(").removesuffix(")").split(",")
        if len(channels) >= 3:
            return tuple(float(channel) / 255 for channel in channels[:3])  # type: ignore[return-value]
    return color


def _matplotlib_linestyle(linestyle: str) -> str:
    return {
        "solid": "-",
        "dash": "--",
        "dashed": "--",
        "dot": ":",
        "dotted": ":",
        "dashdot": "-.",
    }.get(linestyle, linestyle)


def _add_matplotlib_x_markers(ax: Any, markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None) -> None:
    for marker in _normalize_plot_markers(markers):
        color = _matplotlib_color(str(marker["color"]))
        ax.axvline(
            marker["value"],
            color=color,
            linestyle=_matplotlib_linestyle(str(marker["linestyle"])),
            linewidth=marker["linewidth"],
            alpha=marker["alpha"],
        )
        if marker["label"]:
            ax.text(
                marker["value"],
                0.98,
                marker["label"],
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                color=color,
                fontsize=8,
                alpha=marker["alpha"],
            )


def _add_matplotlib_y_markers(ax: Any, markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None) -> None:
    for marker in _normalize_plot_markers(markers):
        color = _matplotlib_color(str(marker["color"]))
        ax.axhline(
            marker["value"],
            color=color,
            linestyle=_matplotlib_linestyle(str(marker["linestyle"])),
            linewidth=marker["linewidth"],
            alpha=marker["alpha"],
        )
        if marker["label"]:
            ax.text(
                0.01,
                marker["value"],
                marker["label"],
                transform=ax.get_yaxis_transform(),
                va="bottom",
                ha="left",
                color=color,
                fontsize=8,
                alpha=marker["alpha"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.5),
            )


def _add_plotly_x_markers(fig: Any, markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None) -> None:
    for marker in _normalize_plot_markers(markers):
        fig.add_vline(
            x=marker["value"],
            line_dash=marker["linestyle"],
            line_color=marker["color"],
            line_width=marker["linewidth"],
            opacity=marker["alpha"],
            annotation_text=marker["label"] or None,
            annotation_position="top left",
        )


def _add_plotly_y_markers(fig: Any, markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None) -> None:
    for marker in _normalize_plot_markers(markers):
        fig.add_hline(
            y=marker["value"],
            line_dash=marker["linestyle"],
            line_color=marker["color"],
            line_width=marker["linewidth"],
            opacity=marker["alpha"],
            annotation_text=marker["label"] or None,
            annotation_position="bottom right",
        )


def plot_dat(
    path: str | Path,
    *,
    variables: Sequence[str] | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    yscale: str = "linear",
    markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
):
    dataset = load_output_file(path, product=product, prefer_nextnanopy=prefer_nextnanopy)
    if dataset.ndim != 1:
        raise ValueError(f"plot_dat expects a 1D dataset, got ndim={dataset.ndim}")

    plt = _import_matplotlib_pyplot()
    x_name = dataset.coord_names[0]
    x_axis = dataset.coords[x_name]
    selected = list(variables) if variables is not None else dataset.variable_names

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for variable_name in selected:
        variable_data = dataset.get_variable(variable_name)
        ax.plot(x_axis.value, variable_data.value, label=variable_data.name)

    ax.set_xlabel(x_axis.label or x_name)
    y_label = dataset.get_variable(selected[0]).label if len(selected) == 1 else "Value"
    ax.set_ylabel(y_label)
    ax.set_yscale(yscale)
    ax.set_title(_line_title(dataset.path, title, selected))
    ax.grid(alpha=0.25)
    _add_matplotlib_x_markers(ax, markers)
    if len(selected) > 1:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def plot_dat_interactive(
    path: str | Path,
    *,
    variables: Sequence[str] | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    yscale: str = "linear",
    markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
):
    dataset = load_output_file(path, product=product, prefer_nextnanopy=prefer_nextnanopy)
    if dataset.ndim != 1:
        raise ValueError(f"plot_dat_interactive expects a 1D dataset, got ndim={dataset.ndim}")

    go = _import_plotly_go()
    x_name = dataset.coord_names[0]
    x_axis = dataset.coords[x_name]
    selected = list(variables) if variables is not None else dataset.variable_names

    fig = go.Figure()
    for variable_name in selected:
        variable_data = dataset.get_variable(variable_name)
        fig.add_trace(
            go.Scatter(
                x=x_axis.value,
                y=variable_data.value,
                mode="lines",
                name=variable_data.name,
            )
        )

    y_label = dataset.get_variable(selected[0]).label if len(selected) == 1 else "Value"
    fig.update_layout(
        title=_line_title(dataset.path, title, selected),
        xaxis_title=x_axis.label or x_name,
        yaxis_title=y_label,
        template="plotly_white",
        height=500,
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    if yscale == "log":
        fig.update_yaxes(type="log")
    _add_plotly_x_markers(fig, markers)
    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    if ylim is not None:
        fig.update_yaxes(range=[ylim[0], ylim[1]])
    return _display_plotly_figure(fig)


def plot_fld(
    path: str | Path,
    *,
    variable: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    log10: bool = False,
    cmap: str = "magma",
    x_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    y_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
):
    dataset = load_output_file(path, product=product, prefer_nextnanopy=prefer_nextnanopy)
    if dataset.ndim != 2:
        raise ValueError(f"plot_fld expects a 2D dataset, got ndim={dataset.ndim}")

    plt = _import_matplotlib_pyplot()
    x_name, y_name = dataset.coord_names
    x_axis = dataset.coords[x_name]
    y_axis = dataset.coords[y_name]
    variable_data = _get_variable_data(dataset, variable)
    z = _as_2d_plot_array(_maybe_log10(np.asarray(variable_data.value), log10), x_axis.value, y_axis.value)

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(x_axis.value, y_axis.value, z, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(variable_data.label if not log10 else f"log10({variable_data.label or variable_data.name})")
    ax.set_xlabel(x_axis.label or x_name)
    ax.set_ylabel(y_axis.label or y_name)
    ax.set_title(_heatmap_title(dataset.path, title, variable_data.name))
    _add_matplotlib_x_markers(ax, x_markers)
    _add_matplotlib_y_markers(ax, y_markers)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def plot_fld_interactive(
    path: str | Path,
    *,
    variable: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    log10: bool = False,
    colorscale: str = "Turbo",
    x_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    y_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    product: str = DEFAULT_PRODUCT,
    prefer_nextnanopy: bool = True,
):
    dataset = load_output_file(path, product=product, prefer_nextnanopy=prefer_nextnanopy)
    if dataset.ndim != 2:
        raise ValueError(f"plot_fld_interactive expects a 2D dataset, got ndim={dataset.ndim}")

    go = _import_plotly_go()
    x_name, y_name = dataset.coord_names
    x_axis = dataset.coords[x_name]
    y_axis = dataset.coords[y_name]
    variable_data = _get_variable_data(dataset, variable)
    z = _as_2d_plot_array(_maybe_log10(np.asarray(variable_data.value), log10), x_axis.value, y_axis.value)

    fig = go.Figure(
        go.Heatmap(
            x=x_axis.value,
            y=y_axis.value,
            z=z,
            colorscale=colorscale,
            colorbar=dict(
                title=variable_data.label if not log10 else f"log10({variable_data.label or variable_data.name})"
            ),
        )
    )
    fig.update_layout(
        title=_heatmap_title(dataset.path, title, variable_data.name),
        xaxis_title=x_axis.label or x_name,
        yaxis_title=y_axis.label or y_name,
        template="plotly_white",
        height=650,
    )
    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    if ylim is not None:
        fig.update_yaxes(range=[ylim[0], ylim[1]])
    _add_plotly_x_markers(fig, x_markers)
    _add_plotly_y_markers(fig, y_markers)
    return _display_plotly_figure(fig)


def plot_vtr_slice(
    path: str | Path,
    *,
    variable: str | None = None,
    slice_axis: str = "y",
    slice_value: float | None = None,
    slice_index: int | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    log10: bool = False,
    cmap: str = "magma",
    product: str = DEFAULT_PRODUCT,
):
    selected = [variable] if variable is not None else None
    dataset = _load_vtr_dataset(path, variable_names=selected)
    variable_data = _get_variable_data(dataset, variable)
    plane = extract_plane(
        dataset,
        variable=variable_data.name,
        slice_axis=slice_axis,
        slice_value=slice_value,
        slice_index=slice_index,
        product=product,
        prefer_nextnanopy=True,
    )
    plt = _import_matplotlib_pyplot()
    z = _as_2d_plot_array(_maybe_log10(np.asarray(plane["values"]), log10), plane["x"], plane["y"])

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(plane["x"], plane["y"], z, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    variable_data = plane["variable"]
    cbar.set_label(variable_data.label if not log10 else f"log10({variable_data.label or variable_data.name})")

    plot_title = title or (
        f"{_coerce_path(path).name}: {variable_data.name} @ "
        f"{plane['slice_axis']}={plane['slice_coordinate']:.3f}"
    )
    ax.set_title(plot_title)
    ax.set_xlabel(plane["x_label"])
    ax.set_ylabel(plane["y_label"])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def plot_vtr_slice_interactive(
    path: str | Path,
    *,
    variable: str | None = None,
    slice_axis: str = "y",
    slice_value: float | None = None,
    slice_index: int | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    log10: bool = False,
    colorscale: str = "Turbo",
    product: str = DEFAULT_PRODUCT,
):
    selected = [variable] if variable is not None else None
    dataset = _load_vtr_dataset(path, variable_names=selected)
    variable_data = _get_variable_data(dataset, variable)
    plane = extract_plane(
        dataset,
        variable=variable_data.name,
        slice_axis=slice_axis,
        slice_value=slice_value,
        slice_index=slice_index,
        product=product,
        prefer_nextnanopy=True,
    )
    go = _import_plotly_go()
    z = _as_2d_plot_array(_maybe_log10(np.asarray(plane["values"]), log10), plane["x"], plane["y"])
    variable_data = plane["variable"]

    fig = go.Figure(
        go.Heatmap(
            x=plane["x"],
            y=plane["y"],
            z=z,
            colorscale=colorscale,
            colorbar=dict(
                title=variable_data.label if not log10 else f"log10({variable_data.label or variable_data.name})"
            ),
        )
    )
    fig.update_layout(
        title=title
        or f"{_coerce_path(path).name}: {variable_data.name} @ {plane['slice_axis']}={plane['slice_coordinate']:.3f}",
        xaxis_title=plane["x_label"],
        yaxis_title=plane["y_label"],
        template="plotly_white",
        height=650,
    )
    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    if ylim is not None:
        fig.update_yaxes(range=[ylim[0], ylim[1]])
    return _display_plotly_figure(fig)


def plot_vtr_linecut(
    path: str | Path,
    *,
    variable: str | None = None,
    axis: str,
    fixed_coords: Mapping[str, float] | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    yscale: str = "linear",
    interactive: bool = False,
    product: str = DEFAULT_PRODUCT,
):
    selected = [variable] if variable is not None else None
    dataset = _load_vtr_dataset(path, variable_names=selected)
    variable_data = _get_variable_data(dataset, variable)
    line = extract_linecut(
        dataset,
        variable=variable_data.name,
        axis=axis,
        fixed_coords=fixed_coords,
        product=product,
        prefer_nextnanopy=True,
    )

    if interactive:
        go = _import_plotly_go()
        fig = go.Figure(
            go.Scatter(
                x=line["axis"],
                y=line["values"],
                mode="lines",
                name=line["variable"].name,
            )
        )
        fig.update_layout(
            title=title or f"{_coerce_path(path).name}: {line['variable'].name}",
            xaxis_title=line["axis_label"],
            yaxis_title=line["variable"].label or line["variable"].name,
            template="plotly_white",
            height=500,
        )
        if yscale == "log":
            fig.update_yaxes(type="log")
        if xlim is not None:
            fig.update_xaxes(range=[xlim[0], xlim[1]])
        if ylim is not None:
            fig.update_yaxes(range=[ylim[0], ylim[1]])
        return _display_plotly_figure(fig)

    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(line["axis"], line["values"], label=line["variable"].name)
    ax.set_xlabel(line["axis_label"])
    ax.set_ylabel(line["variable"].label or line["variable"].name)
    ax.set_yscale(yscale)
    ax.set_title(title or f"{_coerce_path(path).name}: {line['variable'].name}")
    ax.grid(alpha=0.25)
    if line["chosen_coords"]:
        subtitle = ", ".join(f"{name}={value:.3f}" for name, value in line["chosen_coords"].items())
        ax.text(0.99, 0.98, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def _range_slice(values: np.ndarray, value_range: Sequence[float] | None) -> slice:
    if value_range is None:
        return slice(None)
    if len(value_range) != 2:
        raise ValueError(f"Coordinate ranges must have two values, got {value_range!r}")
    lo, hi = sorted(float(value) for value in value_range)
    indices = np.flatnonzero((values >= lo) & (values <= hi))
    if indices.size == 0:
        raise ValueError(f"No coordinate values are inside range [{lo}, {hi}]")
    return slice(int(indices[0]), int(indices[-1]) + 1)


def _volume_stride_for_shape(
    shape: Sequence[int],
    coord_names: Sequence[str],
    *,
    max_points: int,
    strides: int | Mapping[str, int] | Sequence[int] | None,
) -> tuple[int, ...]:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    if strides is None:
        total_points = int(np.prod(shape))
        if total_points <= max_points:
            return tuple(1 for _ in shape)
        step = int(np.ceil((total_points / max_points) ** (1.0 / len(shape))))
        return tuple(max(step, 1) for _ in shape)

    if isinstance(strides, int):
        if strides <= 0:
            raise ValueError("strides must be positive.")
        return tuple(strides for _ in shape)

    if isinstance(strides, Mapping):
        return tuple(max(int(strides.get(name, 1)), 1) for name in coord_names)

    normalized = tuple(max(int(value), 1) for value in strides)
    if len(normalized) != len(shape):
        raise ValueError(f"Expected {len(shape)} stride values, got {len(normalized)}")
    return normalized


def plot_vtr_volume_interactive(
    path: str | Path,
    *,
    variable: str | None = None,
    title: str | None = None,
    log10: bool = False,
    coord_ranges: Mapping[str, Sequence[float] | None] | None = None,
    max_points: int = 200_000,
    strides: int | Mapping[str, int] | Sequence[int] | None = None,
    percentile_range: tuple[float, float] = (1.0, 99.8),
    value_range: tuple[float, float] | None = None,
    mode: str = "volume",
    opacity: float = 0.16,
    surface_count: int = 12,
    colorscale: str = "Turbo",
    product: str = DEFAULT_PRODUCT,
):
    """Plot a downsampled 3D view of a rectilinear-grid `.vtr` output.

    This is intended for large nextnano++ ASCII VTK outputs where sending every
    grid point to Plotly would make the notebook sluggish. Use `coord_ranges` to
    focus on a region such as the quantum well, and `strides` or `max_points` to
    control downsampling.
    """
    selected = [variable] if variable is not None else None
    dataset = _load_vtr_dataset(path, variable_names=selected)
    if dataset.ndim != 3:
        raise ValueError(f"plot_vtr_volume_interactive expects a 3D dataset, got ndim={dataset.ndim}")

    variable_data = _get_variable_data(dataset, variable)
    coord_names = dataset.coord_names
    coords = [np.asarray(dataset.coords[name].value) for name in coord_names]
    values = _orient_nd_array(np.asarray(variable_data.value), coords)

    coord_ranges = coord_ranges or {}
    crop_slices = tuple(_range_slice(coord, coord_ranges.get(name)) for name, coord in zip(coord_names, coords))
    coords = [coord[slicer] for coord, slicer in zip(coords, crop_slices)]
    values = values[crop_slices]

    stride_values = _volume_stride_for_shape(
        values.shape,
        coord_names,
        max_points=max_points,
        strides=strides,
    )
    downsample_slices = tuple(slice(None, None, stride) for stride in stride_values)
    coords = [coord[slicer] for coord, slicer in zip(coords, downsample_slices)]
    values = values[downsample_slices]

    plot_values = _maybe_log10(values.astype(float, copy=False), log10)
    finite = plot_values[np.isfinite(plot_values)]
    if finite.size == 0:
        raise ValueError(f"No finite values are available for {variable_data.name} after filtering.")

    if value_range is None:
        isomin, isomax = np.nanpercentile(finite, percentile_range)
    else:
        isomin, isomax = value_range
    isomin = float(isomin)
    isomax = float(isomax)
    if not np.isfinite(isomin) or not np.isfinite(isomax) or isomin == isomax:
        isomin = float(np.nanmin(finite))
        isomax = float(np.nanmax(finite))
    if isomin == isomax:
        raise ValueError(f"Cannot plot volume with a single finite value: {isomin}")

    plot_values = np.nan_to_num(plot_values, nan=isomin, posinf=isomax, neginf=isomin)
    meshes = np.meshgrid(*coords, indexing="ij")
    flat_coords = [mesh.ravel() for mesh in meshes]
    flat_values = plot_values.ravel()

    go = _import_plotly_go()
    value_label = variable_data.label if not log10 else f"log10({variable_data.label or variable_data.name})"
    mode = mode.lower()
    if mode == "volume":
        trace = go.Volume(
            x=flat_coords[0],
            y=flat_coords[1],
            z=flat_coords[2],
            value=flat_values,
            isomin=isomin,
            isomax=isomax,
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            colorbar=dict(title=value_label),
        )
    elif mode == "isosurface":
        trace = go.Isosurface(
            x=flat_coords[0],
            y=flat_coords[1],
            z=flat_coords[2],
            value=flat_values,
            isomin=isomin,
            isomax=isomax,
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title=value_label),
        )
    else:
        raise ValueError("mode must be either 'volume' or 'isosurface'.")

    fig = go.Figure(trace)
    range_bits = []
    for name, coord in zip(coord_names, coords):
        if coord.size:
            range_bits.append(f"{name}=[{float(coord.min()):.3g}, {float(coord.max()):.3g}]")
    stride_bits = ", ".join(f"{name}:{stride}" for name, stride in zip(coord_names, stride_values))
    fig.update_layout(
        title=title
        or (
            f"{_coerce_path(path).name}: {variable_data.name} 3D {mode} "
            f"({'; '.join(range_bits)}, strides {stride_bits})"
        ),
        scene={
            "xaxis_title": dataset.coords[coord_names[0]].label or coord_names[0],
            "yaxis_title": dataset.coords[coord_names[1]].label or coord_names[1],
            "zaxis_title": dataset.coords[coord_names[2]].label or coord_names[2],
            "aspectmode": "data",
        },
        template="plotly_white",
        height=760,
        margin=dict(l=0, r=0, t=54, b=0),
    )
    fig.update_traces(
        hovertemplate=(
            f"{value_label}: %{{value:.4g}}<br>"
            f"{coord_names[0]}=%{{x:.3f}}<br>"
            f"{coord_names[1]}=%{{y:.3f}}<br>"
            f"{coord_names[2]}=%{{z:.3f}}<extra></extra>"
        )
    )
    return _display_plotly_figure(fig)


def plot_bias_volume_3d(
    run_root: str | Path,
    quantity: str,
    *,
    bias: str | int | None = None,
    variable: str | None = None,
    title: str | None = None,
    log10: bool = False,
    coord_ranges: Mapping[str, Sequence[float] | None] | None = None,
    max_points: int = 200_000,
    strides: int | Mapping[str, int] | Sequence[int] | None = None,
    percentile_range: tuple[float, float] = (1.0, 99.8),
    value_range: tuple[float, float] | None = None,
    mode: str = "volume",
    opacity: float = 0.16,
    surface_count: int = 12,
    colorscale: str = "Turbo",
):
    path = resolve_bias_output_file(run_root, quantity, bias=bias, preferred_extensions=("vtr",))
    return plot_vtr_volume_interactive(
        path,
        variable=variable,
        title=title,
        log10=log10,
        coord_ranges=coord_ranges,
        max_points=max_points,
        strides=strides,
        percentile_range=percentile_range,
        value_range=value_range,
        mode=mode,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
    )


def load_simulation_layout_json(path: str | Path) -> dict[str, Any]:
    """Read a qd_design simulation-layout JSON file."""
    return json.loads(_coerce_path(path).read_text())


def _gate_mesh_trace(
    region: Mapping[str, Any],
    *,
    color: str,
    opacity: float,
    name_prefix: str = "gate",
):
    polygons = region.get("polygon_xy_nm") or []
    if not polygons:
        polygon = [
            (float(region["x_min_nm"]), float(region["y_min_nm"])),
            (float(region["x_max_nm"]), float(region["y_min_nm"])),
            (float(region["x_max_nm"]), float(region["y_max_nm"])),
            (float(region["x_min_nm"]), float(region["y_max_nm"])),
        ]
    else:
        polygon = [(float(x), float(y)) for x, y in polygons[0]]

    if len(polygon) < 3:
        return None

    z_min = float(region.get("z_min_nm", 0.0))
    z_max = float(region.get("z_max_nm", z_min))
    n = len(polygon)

    xs = [point[0] for point in polygon] + [point[0] for point in polygon]
    ys = [point[1] for point in polygon] + [point[1] for point in polygon]
    zs = [z_min] * n + [z_max] * n

    i: list[int] = []
    j: list[int] = []
    k: list[int] = []

    top0 = n
    for idx in range(1, n - 1):
        i.append(top0)
        j.append(n + idx)
        k.append(n + idx + 1)

    for idx in range(n):
        nxt = (idx + 1) % n
        i.extend([idx, idx])
        j.extend([nxt, n + nxt])
        k.extend([n + nxt, n + idx])

    go = _import_plotly_go()
    gate_name = str(region.get("name", name_prefix))
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        name=gate_name,
        color=color,
        opacity=opacity,
        flatshading=True,
        hovertemplate=(
            f"{gate_name}<br>"
            "x=%{x:.1f} nm<br>"
            "y=%{y:.1f} nm<br>"
            "z=%{z:.1f} nm<extra></extra>"
        ),
        showscale=False,
    )


def _gate_polygon_surface_trace(
    region: Mapping[str, Any],
    *,
    z_value: float,
    color: str,
    opacity: float,
    name_prefix: str = "gate",
):
    polygons = region.get("polygon_xy_nm") or []
    if not polygons:
        polygon = [
            (float(region["x_min_nm"]), float(region["y_min_nm"])),
            (float(region["x_max_nm"]), float(region["y_min_nm"])),
            (float(region["x_max_nm"]), float(region["y_max_nm"])),
            (float(region["x_min_nm"]), float(region["y_max_nm"])),
        ]
    else:
        polygon = [(float(x), float(y)) for x, y in polygons[0]]

    if len(polygon) < 3:
        return None

    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    zs = [float(z_value)] * len(polygon)
    i = [0] * max(len(polygon) - 2, 0)
    j = list(range(1, len(polygon) - 1))
    k = list(range(2, len(polygon)))

    go = _import_plotly_go()
    gate_name = str(region.get("name", name_prefix))
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        name=gate_name,
        color=color,
        opacity=opacity,
        flatshading=True,
        hovertemplate=(
            f"{gate_name}<br>"
            "x=%{x:.1f} nm<br>"
            "y=%{y:.1f} nm<extra></extra>"
        ),
        showscale=False,
    )


def _read_contact_index_map(path: str | Path) -> dict[int, str]:
    table = read_index_table(path)
    if table.empty:
        return {}
    index_col = table.columns[0]
    name_col = table.columns[1] if len(table.columns) > 1 else table.columns[0]
    return {int(row[index_col]): str(row[name_col]) for _, row in table.iterrows()}


def _contact_projection_surface_traces(
    run_root: str | Path,
    gate_regions: Sequence[Mapping[str, Any]],
    *,
    gate_z: float,
    color: str,
    opacity: float,
    contact_path: str | Path | None = None,
) -> list[Any]:
    root = resolve_run_root(run_root)
    contact_file = _coerce_path(contact_path) if contact_path is not None else root / "Structure" / "contacts.vtr"
    contact_index_file = root / "Structure" / "contact_indices.txt"
    if not contact_file.exists() or not contact_index_file.exists():
        return []

    contact_names = _read_contact_index_map(contact_index_file)
    gate_names = {str(region.get("name", "")) for region in gate_regions}
    gate_indices = {index: name for index, name in contact_names.items() if name in gate_names}
    if not gate_indices:
        return []

    dataset = _load_vtr_dataset(contact_file, variable_names=["Contact_index"])
    variable_data = _get_variable_data(dataset, "Contact_index")
    coord_names = dataset.coord_names
    if coord_names[:3] != ["x", "y", "z"]:
        return []

    x = np.asarray(dataset.coords["x"].value, dtype=float)
    y = np.asarray(dataset.coords["y"].value, dtype=float)
    z = np.asarray(dataset.coords["z"].value, dtype=float)
    contact_values = _orient_nd_array(np.asarray(variable_data.value), [x, y, z])
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    go = _import_plotly_go()
    traces: list[Any] = []
    ordered_gate_names = [str(region.get("name", "")) for region in gate_regions]
    ordered_indices = sorted(gate_indices, key=lambda index: ordered_gate_names.index(gate_indices[index]))
    for contact_index in ordered_indices:
        contact_name = gate_indices[contact_index]
        mask = np.any(np.isclose(contact_values, float(contact_index)), axis=2)
        if not np.any(mask):
            continue
        mask_plot = _as_2d_plot_array(mask.astype(float), x, y)
        traces.append(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=np.where(mask_plot > 0, gate_z, np.nan),
                surfacecolor=np.where(mask_plot > 0, float(contact_index), np.nan),
                colorscale=[[0.0, color], [1.0, color]],
                cmin=float(contact_index) - 0.5,
                cmax=float(contact_index) + 0.5,
                name=contact_name,
                opacity=opacity,
                showscale=False,
                hovertemplate=(
                    f"{contact_name}<br>"
                    "x=%{x:.1f} nm<br>"
                    "y=%{y:.1f} nm<extra></extra>"
                ),
            )
        )
    return traces


def _normal_axis_range(values: np.ndarray, pad_fraction: float = 0.02) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    pad = pad_fraction * (hi - lo if hi > lo else max(abs(hi), 1.0))
    return lo - pad, hi + pad


def plot_gate_bandedge_density_3d(
    run_root: str | Path,
    simulation_layout_path: str | Path,
    *,
    z_nm: float = -4.0,
    bias: str | int | None = 0,
    bandedge_variable: str = "HH",
    density_quantity: str = "density_hole",
    density_variable: str | None = None,
    density_log10: bool = False,
    density_threshold_percentile: float | None = None,
    gate_lift_fraction: float = 0.16,
    gate_color: str = "#0b37bf",
    gate_opacity: float = 0.92,
    gate_source: str = "contacts_projection",
    contact_path: str | Path | None = None,
    bandedge_colorscale: Sequence[Sequence[Any]] | str = "Greens",
    density_colorscale: Sequence[Sequence[Any]] | str | None = "Reds",
    title: str | None = None,
):
    """
    Compose a 3D presentation view like the nextnano band/density examples.

    The bandedge and density planes are loaded from full 3D ``.vtr`` outputs, so
    this can target depths that were not pre-exported as ``section2D`` files.

    The plotted vertical coordinate is a profile height, not the physical z
    position: HH band edge values and hole-density values are each used directly
    as the third coordinate of their own surface. The gates are drawn as a flat
    2D top-view layer above both profile surfaces. By default that layer is a
    projection of the simulated ``Structure/contacts.vtr`` contact index volume.
    """
    go = _import_plotly_go()
    run_root = resolve_run_root(run_root)
    bandedge_path = resolve_bias_output_file(run_root, "bandedges", bias=bias, preferred_extensions=("vtr",))
    density_path = resolve_bias_output_file(run_root, density_quantity, bias=bias, preferred_extensions=("vtr",))

    bandedge_plane = load_vtr_plane(
        bandedge_path,
        variable=bandedge_variable,
        slice_axis="z",
        slice_value=z_nm,
    )
    density_plane = load_vtr_plane(
        density_path,
        variable=density_variable,
        slice_axis="z",
        slice_value=z_nm,
    )

    x = np.asarray(bandedge_plane["x"], dtype=float)
    y = np.asarray(bandedge_plane["y"], dtype=float)
    if not np.allclose(x, np.asarray(density_plane["x"], dtype=float)) or not np.allclose(
        y, np.asarray(density_plane["y"], dtype=float)
    ):
        raise ValueError("Bandedge and density planes use different x/y grids.")

    bandedge_values = _as_2d_plot_array(np.asarray(bandedge_plane["values"], dtype=float), x, y)
    density_values = _as_2d_plot_array(np.asarray(density_plane["values"], dtype=float), x, y)
    density_display = _maybe_log10(density_values, density_log10)

    actual_z = float(bandedge_plane["slice_coordinate"])
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    z_bandedge = bandedge_values

    density_mask = np.isfinite(density_display)
    if density_threshold_percentile is not None and np.any(density_mask):
        threshold = float(np.nanpercentile(density_display[density_mask], density_threshold_percentile))
        density_mask &= density_display >= threshold
    z_density = np.where(density_mask, density_display, np.nan)

    bandedge_label = bandedge_plane["variable"].label or bandedge_plane["variable"].name
    density_label = density_plane["variable"].label or density_plane["variable"].name
    if density_log10:
        density_label = f"log10({density_label})"

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_bandedge,
            surfacecolor=bandedge_values,
            colorscale=bandedge_colorscale,
            colorbar=dict(title=bandedge_label, x=0.86, y=0.72, yanchor="middle", len=0.36, thickness=18),
            name=bandedge_label,
            opacity=0.96,
            hovertemplate=(
                "x=%{x:.1f} nm<br>"
                "y=%{y:.1f} nm<br>"
                f"{bandedge_label}=%{{surfacecolor:.4g}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_density,
            surfacecolor=density_display,
            colorscale=density_colorscale,
            colorbar=dict(title=density_label, x=0.86, y=0.28, yanchor="middle", len=0.36, thickness=18),
            name=density_label,
            opacity=0.68,
            showscale=True,
            hovertemplate=(
                "x=%{x:.1f} nm<br>"
                "y=%{y:.1f} nm<br>"
                f"{density_label}=%{{surfacecolor:.4g}}<extra></extra>"
            ),
        )
    )

    layout = load_simulation_layout_json(simulation_layout_path)
    gate_regions = layout.get("patterned_regions", [])

    profile_values = np.concatenate(
        [
            bandedge_values[np.isfinite(bandedge_values)].ravel(),
            density_display[np.isfinite(density_display)].ravel(),
        ]
    )
    profile_min = float(np.nanmin(profile_values)) if profile_values.size else 0.0
    profile_max = float(np.nanmax(profile_values)) if profile_values.size else 1.0
    profile_span = profile_max - profile_min
    if not np.isfinite(profile_span) or profile_span <= 0:
        profile_span = max(abs(profile_max), 1.0)
    gate_z = profile_max + gate_lift_fraction * profile_span

    gate_source = gate_source.lower()
    gate_traces: list[Any] = []
    if gate_source in {"contacts", "contacts_projection", "contact_projection"}:
        gate_traces = _contact_projection_surface_traces(
            run_root,
            gate_regions,
            gate_z=gate_z,
            color=gate_color,
            opacity=gate_opacity,
            contact_path=contact_path,
        )
    if not gate_traces:
        for region in gate_regions:
            trace = _gate_polygon_surface_trace(
                region,
                z_value=gate_z,
                color=gate_color,
                opacity=gate_opacity,
            )
            if trace is not None:
                gate_traces.append(trace)
    for trace in gate_traces:
        fig.add_trace(trace)

    for region in gate_regions:
        fig.add_trace(
            go.Scatter3d(
                x=[0.5 * (float(region["x_min_nm"]) + float(region["x_max_nm"]))],
                y=[0.5 * (float(region["y_min_nm"]) + float(region["y_max_nm"]))],
                z=[gate_z + 0.04 * profile_span],
                text=[str(region.get("name", ""))],
                mode="text",
                textfont=dict(color=gate_color, size=11),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    x_range = _normal_axis_range(x)
    y_range = _normal_axis_range(y)
    z_values = np.concatenate(
        [
            profile_values,
            np.asarray([gate_z + 0.04 * profile_span], dtype=float),
        ]
    )
    z_range = _normal_axis_range(np.asarray(z_values, dtype=float), pad_fraction=0.06)

    requested = bandedge_plane.get("requested_slice_coordinate")
    if requested is not None:
        requested_text = f"requested z={requested:.2f} nm, actual z={actual_z:.2f} nm"
    else:
        requested_text = f"z={actual_z:.2f} nm"
    fig.update_layout(
        title=title or f"{bandedge_label} and hole density under gates ({requested_text})",
        scene=dict(
            domain=dict(x=[0.0, 0.76], y=[0.0, 1.0]),
            xaxis=dict(title="x (nm)", range=x_range, backgroundcolor="rgba(245,247,250,0.65)"),
            yaxis=dict(title="y (nm)", range=y_range, backgroundcolor="rgba(245,247,250,0.65)"),
            zaxis=dict(
                title="",
                range=z_range,
                backgroundcolor="rgba(245,247,250,0.65)",
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.35, y=0.9, z=0.85),
            camera=dict(eye=dict(x=1.55, y=-1.7, z=1.05)),
        ),
        template="plotly_white",
        height=760,
        margin=dict(l=0, r=20, t=64, b=0),
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        meta=dict(requested_z_nm=z_nm, actual_z_nm=actual_z),
    )
    return fig


def read_convergence_table(bias_dir: str | Path) -> pd.DataFrame:
    path = _coerce_path(bias_dir)
    if path.is_dir():
        path = path / "iteration_quantum_poisson.dat"
    return read_dat_table(path)


def convergence_summary(
    bias_dir: str | Path,
    *,
    residual_targets: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    df = read_convergence_table(bias_dir)
    residual_columns = [column for column in df.columns if "Residual" in column]
    last_row = df.iloc[-1]
    summary = {
        "n_iterations": int(len(df)),
        "final_iteration": int(last_row[df.columns[0]]),
        "final_residuals": {column: float(last_row[column]) for column in residual_columns},
        "min_residuals": {column: float(df[column].min()) for column in residual_columns},
    }
    if residual_targets:
        summary["targets"] = dict(residual_targets)
        summary["meets_targets"] = {
            column: float(last_row[column]) <= target for column, target in residual_targets.items() if column in df.columns
        }
    return summary


def plot_convergence(
    bias_dir: str | Path,
    *,
    logy: bool = True,
    interactive: bool = False,
    residual_targets: Mapping[str, float] | None = None,
):
    df = read_convergence_table(bias_dir)
    x_column = df.columns[0]
    residual_columns = [column for column in df.columns if "Residual" in column]

    if interactive:
        go = _import_plotly_go()
        fig = go.Figure()
        for column in residual_columns:
            fig.add_trace(go.Scatter(x=df[x_column], y=df[column], mode="lines+markers", name=column))
        if residual_targets:
            for column, target in residual_targets.items():
                fig.add_hline(y=target, line_dash="dot", annotation_text=f"{column} target")
        fig.update_layout(
            title="Quantum-Poisson convergence",
            xaxis_title=x_column,
            yaxis_title="Residual",
            template="plotly_white",
            height=500,
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        )
        if logy:
            fig.update_yaxes(type="log")
        return _display_plotly_figure(fig)

    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for column in residual_columns:
        ax.plot(df[x_column], df[column], marker="o", label=column)
    if residual_targets:
        for column, target in residual_targets.items():
            ax.axhline(target, linestyle=":", alpha=0.7)
    ax.set_title("Quantum-Poisson convergence")
    ax.set_xlabel(x_column)
    ax.set_ylabel("Residual")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def build_region_material_map(run_root: str | Path, *, cut: str = "1d_z_BG2") -> pd.DataFrame:
    run_paths = get_run_paths(run_root)
    region_file = run_paths.structure / f"regions_material_{cut}.dat"
    material_file = run_paths.structure / f"materials_{cut}.dat"
    material_index_file = run_paths.structure / "material_indices.txt"

    region_df = read_dat_table(region_file)
    material_df = read_dat_table(material_file)
    material_names = read_index_table(material_index_file)

    coord_col = region_df.columns[0]
    region_col = region_df.columns[-1]
    material_col = material_df.columns[-1]

    merged = pd.DataFrame(
        {
            coord_col: region_df[coord_col].to_numpy(dtype=float),
            "structure_region_index": region_df[region_col].to_numpy(dtype=int),
            "material_index": material_df[material_col].to_numpy(dtype=int),
        }
    )
    summary = (
        merged.groupby(["structure_region_index", "material_index"], as_index=False)
        .agg(coord_min=(coord_col, "min"), coord_max=(coord_col, "max"), n_points=(coord_col, "size"))
        .sort_values(["coord_min", "structure_region_index"])
        .reset_index(drop=True)
    )

    material_lookup = dict(zip(material_names.iloc[:, 0], material_names.iloc[:, 1]))
    summary["material_name"] = summary["material_index"].map(material_lookup).fillna("unknown")
    summary["thickness_nm"] = summary["coord_max"] - summary["coord_min"]
    summary["cut"] = cut
    summary["coordinate"] = coord_col
    return summary


def plot_structure_linecut(
    run_root: str | Path,
    *,
    cut: str = "1d_z_BG2",
    quantities: Sequence[str] = (
        "materials",
        "regions_material",
        "regions_all",
        "contacts",
        "density_acceptor",
        "density_fixed_charge",
    ),
):
    plt = _import_matplotlib_pyplot()
    run_paths = get_run_paths(run_root)
    existing = []
    for quantity in quantities:
        candidate = run_paths.structure / f"{quantity}_{cut}.dat"
        if candidate.exists():
            existing.append((quantity, candidate))
    if not existing:
        raise FileNotFoundError(f"No structure line-cut files matching '*_{cut}.dat' were found")

    fig, axes = plt.subplots(len(existing), 1, figsize=(10, 2.6 * len(existing)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (quantity, path) in zip(axes, existing):
        dataset = load_output_file(path, prefer_nextnanopy=False)
        x_name = dataset.coord_names[0]
        x_axis = dataset.coords[x_name]
        variable_name = dataset.variable_names[0]
        variable = dataset.variables[variable_name]
        ax.plot(x_axis.value, variable.value, drawstyle="steps-mid", linewidth=1.6)
        ax.set_ylabel(variable.label or variable.name)
        ax.set_title(quantity)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(dataset.coords[dataset.coord_names[0]].label)
    fig.suptitle(f"Structure inspection: {Path(run_root).name} ({cut})", y=1.01)
    fig.tight_layout()
    return fig


def plot_structure_plane(
    run_root: str | Path,
    *,
    quantity: str = "regions_material",
    interactive: bool = False,
    log10: bool = False,
):
    path = resolve_structure_file(run_root, quantity)
    variables = list_variables(path, product=DEFAULT_PRODUCT) if path.suffix.lower() != ".dat" else None
    variable = variables[0] if variables else None
    if path.suffix.lower() == ".fld":
        if interactive:
            return plot_fld_interactive(path, variable=variable, log10=log10)
        return plot_fld(path, variable=variable, log10=log10)
    if path.suffix.lower() == ".vtr":
        if interactive:
            return plot_vtr_slice_interactive(path, variable=variable, log10=log10)
        return plot_vtr_slice(path, variable=variable, log10=log10)
    raise ValueError(f"plot_structure_plane expected .fld or .vtr, got {path}")


def read_integrated_density_hole(run_root_or_path: str | Path) -> pd.DataFrame:
    path = _coerce_path(run_root_or_path)
    if path.is_dir():
        path = path / "integrated_density_hole.dat"
    return read_dat_table(path)


def integrated_density_region_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if _REGION_COLUMN_RE.match(column)]


def map_integrated_density_regions(
    run_root: str | Path,
    *,
    cut: str = "1d_z_BG2",
    region_index_offset: int = 1,
) -> pd.DataFrame:
    integrated = read_integrated_density_hole(run_root)
    structure_map = build_region_material_map(run_root, cut=cut)

    rows: list[dict[str, Any]] = []
    for column in integrated_density_region_columns(integrated):
        match = _REGION_COLUMN_RE.match(column)
        if match is None:
            continue
        integrated_region = int(match.group("index"))
        structure_region = integrated_region - region_index_offset
        match_rows = structure_map[structure_map["structure_region_index"] == structure_region]

        row: dict[str, Any] = {
            "integrated_column": column,
            "integrated_region_index": integrated_region,
            "structure_region_index": structure_region,
            "region_index_offset": region_index_offset,
        }
        if not match_rows.empty:
            best = match_rows.iloc[0]
            row.update(
                material_index=int(best["material_index"]),
                material_name=str(best["material_name"]),
                coord_min=float(best["coord_min"]),
                coord_max=float(best["coord_max"]),
                thickness_nm=float(best["thickness_nm"]),
                n_points=int(best["n_points"]),
                cut=str(best["cut"]),
            )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_integrated_density_hole(
    run_root_or_path: str | Path,
    *,
    x: str | None = None,
    region_columns: Sequence[str] | None = None,
    interactive: bool = True,
    label_with_materials: bool = False,
    cut: str = "1d_z_BG2",
    region_index_offset: int = 1,
):
    path = _coerce_path(run_root_or_path)
    run_root = path if path.is_dir() else path.parent
    if path.is_dir():
        path = path / "integrated_density_hole.dat"

    df = read_integrated_density_hole(path)
    regions = list(region_columns) if region_columns is not None else integrated_density_region_columns(df)
    if not regions:
        raise ValueError("No region_* columns were found in integrated_density_hole.dat")

    non_region_columns = [column for column in df.columns if column not in regions]
    x_column = x or (next((col for col in non_region_columns if "bias" in col.lower()), None) or non_region_columns[0])
    trace_names = {region: region for region in regions}
    if label_with_materials and run_root.is_dir():
        mapping = map_integrated_density_regions(run_root, cut=cut, region_index_offset=region_index_offset)
        for _, row in mapping.iterrows():
            region = row["integrated_column"]
            if region not in trace_names:
                continue
            material = row.get("material_name")
            structure_region = row.get("structure_region_index")
            if pd.notna(material):
                trace_names[region] = f"{region} -> {material} (struct {int(structure_region)})"

    if interactive:
        go = _import_plotly_go()
        fig = go.Figure()
        for region in regions:
            fig.add_trace(
                go.Scatter(
                    x=df[x_column],
                    y=df[region],
                    mode="lines+markers",
                    name=trace_names[region],
                )
            )
        fig.update_layout(
            title=f"Integrated hole density vs {x_column}",
            xaxis_title=x_column,
            yaxis_title="Integrated hole density",
            template="plotly_white",
            height=500,
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        )
        return _display_plotly_figure(fig)

    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for region in regions:
        ax.plot(df[x_column], df[region], marker="o", label=trace_names[region])
    ax.set_title(f"Integrated hole density vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel("Integrated hole density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def read_total_charges(bias_dir_or_path: str | Path) -> pd.DataFrame:
    path = _coerce_path(bias_dir_or_path)
    if path.is_dir():
        path = path / "total_charges.txt"

    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        name, rest = stripped.split(":", maxsplit=1)
        parts = rest.strip().split()
        if not parts:
            continue
        value = float(parts[0])
        unit = " ".join(parts[1:])
        rows.append({"quantity": name.strip(), "value": value, "unit": unit})
    return pd.DataFrame(rows)


def plot_total_charges(bias_dir_or_path: str | Path, *, interactive: bool = False):
    df = read_total_charges(bias_dir_or_path)
    title = f"Total charges: {_coerce_path(bias_dir_or_path).name}"

    if interactive:
        go = _import_plotly_go()
        fig = go.Figure(go.Bar(x=df["quantity"], y=df["value"], text=df["unit"]))
        fig.update_layout(
            title=title,
            xaxis_title="Quantity",
            yaxis_title="Charge",
            template="plotly_white",
            height=500,
        )
        return _display_plotly_figure(fig)

    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(df["quantity"], df["value"])
    ax.set_title(title)
    ax.set_ylabel("Charge")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_bias_linecut(
    run_root: str | Path,
    quantity: str,
    *,
    cut: str = "1d_x_QD",
    bias: str | int | None = None,
    variables: Sequence[str] | None = None,
    interactive: bool = True,
    yscale: str = "linear",
    markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
):
    path = resolve_bias_output_file(run_root, quantity, cut=cut, bias=bias, preferred_extensions=("dat",))
    if interactive:
        return plot_dat_interactive(path, variables=variables, yscale=yscale, markers=markers)
    return plot_dat(path, variables=variables, yscale=yscale, markers=markers)


def plot_bias_plane(
    run_root: str | Path,
    quantity: str,
    *,
    cut: str | None = None,
    bias: str | int | None = None,
    variable: str | None = None,
    interactive: bool = False,
    log10: bool = False,
    x_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    y_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
):
    path = resolve_bias_output_file(run_root, quantity, cut=cut, bias=bias, preferred_extensions=("fld",))
    if interactive:
        return plot_fld_interactive(path, variable=variable, log10=log10, x_markers=x_markers, y_markers=y_markers)
    return plot_fld(path, variable=variable, log10=log10, x_markers=x_markers, y_markers=y_markers)


def plot_bias_volume_slice(
    run_root: str | Path,
    quantity: str,
    *,
    bias: str | int | None = None,
    variable: str | None = None,
    slice_axis: str = "y",
    slice_value: float | None = None,
    slice_index: int | None = None,
    title: str | None = None,
    interactive: bool = False,
    log10: bool = False,
):
    path = resolve_bias_output_file(run_root, quantity, bias=bias, preferred_extensions=("vtr",))
    if interactive:
        return plot_vtr_slice_interactive(
            path,
            variable=variable,
            slice_axis=slice_axis,
            slice_value=slice_value,
            slice_index=slice_index,
            title=title,
            log10=log10,
        )
    return plot_vtr_slice(
        path,
        variable=variable,
        slice_axis=slice_axis,
        slice_value=slice_value,
        slice_index=slice_index,
        title=title,
        log10=log10,
    )


def plot_bias_volume_linecut(
    run_root: str | Path,
    quantity: str,
    *,
    bias: str | int | None = None,
    variable: str | None = None,
    axis: str,
    fixed_coords: Mapping[str, float] | None = None,
    interactive: bool = False,
    yscale: str = "linear",
):
    path = resolve_bias_output_file(run_root, quantity, bias=bias, preferred_extensions=("vtr",))
    return plot_vtr_linecut(
        path,
        variable=variable,
        axis=axis,
        fixed_coords=fixed_coords,
        interactive=interactive,
        yscale=yscale,
    )


def plot_quantum_density_volume_3d(
    run_root: str | Path,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias: str | int | None = None,
    variable: str | None = None,
    title: str | None = None,
    log10: bool = False,
    coord_ranges: Mapping[str, Sequence[float] | None] | None = None,
    max_points: int = 200_000,
    strides: int | Mapping[str, int] | Sequence[int] | None = None,
    percentile_range: tuple[float, float] = (1.0, 99.8),
    value_range: tuple[float, float] | None = None,
    mode: str = "volume",
    opacity: float = 0.16,
    surface_count: int = 12,
    colorscale: str = "Turbo",
):
    path = resolve_quantum_output_file(
        run_root,
        "density",
        region=region,
        band=band,
        bias=bias,
        preferred_extensions=("vtr",),
    )
    return plot_vtr_volume_interactive(
        path,
        variable=variable,
        title=title,
        log10=log10,
        coord_ranges=coord_ranges,
        max_points=max_points,
        strides=strides,
        percentile_range=percentile_range,
        value_range=value_range,
        mode=mode,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
    )


def plot_quantum_density_volume_slice(
    run_root: str | Path,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias: str | int | None = None,
    variable: str | None = None,
    slice_axis: str = "z",
    slice_value: float | None = None,
    slice_index: int | None = None,
    title: str | None = None,
    interactive: bool = False,
    log10: bool = False,
):
    path = resolve_quantum_output_file(
        run_root,
        "density",
        region=region,
        band=band,
        bias=bias,
        preferred_extensions=("vtr",),
    )
    if interactive:
        return plot_vtr_slice_interactive(
            path,
            variable=variable,
            slice_axis=slice_axis,
            slice_value=slice_value,
            slice_index=slice_index,
            title=title,
            log10=log10,
        )
    return plot_vtr_slice(
        path,
        variable=variable,
        slice_axis=slice_axis,
        slice_value=slice_value,
        slice_index=slice_index,
        title=title,
        log10=log10,
    )


def plot_quantum_density_volume_linecut(
    run_root: str | Path,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias: str | int | None = None,
    variable: str | None = None,
    axis: str,
    fixed_coords: Mapping[str, float] | None = None,
    title: str | None = None,
    interactive: bool = False,
    yscale: str = "linear",
):
    path = resolve_quantum_output_file(
        run_root,
        "density",
        region=region,
        band=band,
        bias=bias,
        preferred_extensions=("vtr",),
    )
    return plot_vtr_linecut(
        path,
        variable=variable,
        axis=axis,
        fixed_coords=fixed_coords,
        title=title,
        interactive=interactive,
        yscale=yscale,
    )


def plot_quantum_probability_volume_3d(
    run_root: str | Path,
    *,
    state: int = 1,
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str | None = "k00000",
    shifted: bool = True,
    bias: str | int | None = None,
    variable: str | None = None,
    title: str | None = None,
    log10: bool = False,
    coord_ranges: Mapping[str, Sequence[float] | None] | None = None,
    max_points: int = 200_000,
    strides: int | Mapping[str, int] | Sequence[int] | None = None,
    percentile_range: tuple[float, float] = (1.0, 99.8),
    value_range: tuple[float, float] | None = None,
    mode: str = "volume",
    opacity: float = 0.16,
    surface_count: int = 12,
    colorscale: str = "Turbo",
):
    path = resolve_quantum_probability_state_file(
        run_root,
        state=state,
        region=region,
        band=band,
        kpoint=kpoint,
        shifted=shifted,
        bias=bias,
        preferred_extensions=("vtr",),
    )
    variable = f"Psi^2_{state}" if variable is None else variable
    return plot_vtr_volume_interactive(
        path,
        variable=variable,
        title=title,
        log10=log10,
        coord_ranges=coord_ranges,
        max_points=max_points,
        strides=strides,
        percentile_range=percentile_range,
        value_range=value_range,
        mode=mode,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
    )


def plot_quantum_probability_volume_slice(
    run_root: str | Path,
    *,
    state: int = 1,
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str | None = "k00000",
    shifted: bool = True,
    bias: str | int | None = None,
    variable: str | None = None,
    slice_axis: str = "z",
    slice_value: float | None = None,
    slice_index: int | None = None,
    title: str | None = None,
    interactive: bool = False,
    log10: bool = False,
):
    path = resolve_quantum_probability_state_file(
        run_root,
        state=state,
        region=region,
        band=band,
        kpoint=kpoint,
        shifted=shifted,
        bias=bias,
        preferred_extensions=("vtr",),
    )
    variable = f"Psi^2_{state}" if variable is None else variable
    if interactive:
        return plot_vtr_slice_interactive(
            path,
            variable=variable,
            slice_axis=slice_axis,
            slice_value=slice_value,
            slice_index=slice_index,
            title=title,
            log10=log10,
        )
    return plot_vtr_slice(
        path,
        variable=variable,
        slice_axis=slice_axis,
        slice_value=slice_value,
        slice_index=slice_index,
        title=title,
        log10=log10,
    )


def plot_quantum_probability_volume_linecut(
    run_root: str | Path,
    *,
    state: int = 1,
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str | None = "k00000",
    shifted: bool = True,
    bias: str | int | None = None,
    variable: str | None = None,
    axis: str,
    fixed_coords: Mapping[str, float] | None = None,
    title: str | None = None,
    interactive: bool = False,
    yscale: str = "linear",
):
    path = resolve_quantum_probability_state_file(
        run_root,
        state=state,
        region=region,
        band=band,
        kpoint=kpoint,
        shifted=shifted,
        bias=bias,
        preferred_extensions=("vtr",),
    )
    variable = f"Psi^2_{state}" if variable is None else variable
    return plot_vtr_linecut(
        path,
        variable=variable,
        axis=axis,
        fixed_coords=fixed_coords,
        title=title,
        interactive=interactive,
        yscale=yscale,
    )


def find_probability_peaks(
    path: str | Path,
    *,
    variable: str | None = None,
    n_peaks: int = 2,
    min_lateral_separation_nm: float = 50.0,
) -> pd.DataFrame:
    """
    Find separated maxima in a 3D quantum probability-density VTR file.

    Peaks are selected by descending probability value with lateral non-maximum
    suppression in the x-y plane. This keeps neighboring grid points on one dot
    from being reported as separate peaks.
    """
    if n_peaks <= 0:
        raise ValueError("n_peaks must be positive.")
    if min_lateral_separation_nm < 0:
        raise ValueError("min_lateral_separation_nm must be non-negative.")

    dataset = load_output_file(path, prefer_nextnanopy=False)
    if dataset.ndim != 3:
        raise ValueError(f"Expected a 3D probability dataset, got ndim={dataset.ndim}")

    if variable is None:
        psi_candidates = [name for name in dataset.variable_names if "psi" in name.lower()]
        if not psi_candidates:
            raise ValueError(f"No Psi^2-like variable found. Available variables: {dataset.variable_names}")
        variable = psi_candidates[0]

    variable_data = _get_variable_data(dataset, variable)
    coord_names = dataset.coord_names
    x_name = _resolve_name("x", dataset.coords)
    y_name = _resolve_name("y", dataset.coords)
    z_name = _resolve_name("z", dataset.coords)
    if coord_names != [x_name, y_name, z_name]:
        raise ValueError(f"Expected coordinate order [x, y, z], got {coord_names}")

    x = np.asarray(dataset.coords[x_name].value, dtype=float)
    y = np.asarray(dataset.coords[y_name].value, dtype=float)
    z = np.asarray(dataset.coords[z_name].value, dtype=float)
    values = _orient_nd_array(np.asarray(variable_data.value, dtype=float), [x, y, z])

    selected: list[dict[str, Any]] = []
    flat_order = np.argsort(values.ravel())[::-1]
    for flat_index in flat_order:
        value = float(values.ravel()[flat_index])
        if not np.isfinite(value):
            continue
        ix, iy, iz = np.unravel_index(int(flat_index), values.shape)
        x_nm = float(x[ix])
        y_nm = float(y[iy])
        z_nm = float(z[iz])

        if any(
            np.hypot(x_nm - peak["x_nm"], y_nm - peak["y_nm"]) < min_lateral_separation_nm
            for peak in selected
        ):
            continue

        selected.append(
            {
                "peak": len(selected) + 1,
                "x_nm": x_nm,
                "y_nm": y_nm,
                "z_nm": z_nm,
                "probability": value,
                "ix": int(ix),
                "iy": int(iy),
                "iz": int(iz),
                "variable": variable_data.label or variable_data.name,
                "path": str(dataset.path),
            }
        )
        if len(selected) >= n_peaks:
            break

    if len(selected) < n_peaks:
        raise ValueError(
            f"Found only {len(selected)} separated peaks; requested {n_peaks}. "
            "Try lowering min_lateral_separation_nm."
        )

    return pd.DataFrame(selected)


def read_quantum_occupation(
    run_root: str | Path,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias: str | int | None = None,
) -> pd.DataFrame:
    path = resolve_quantum_output_file(run_root, "occupation", region=region, band=band, bias=bias, preferred_extensions=("dat",))
    return read_dat_table(path)


def plot_quantum_occupation(
    run_root: str | Path,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias: str | int | None = None,
    interactive: bool = True,
    yscale: str = "linear",
):
    df = read_quantum_occupation(run_root, region=region, band=band, bias=bias)
    x_column = df.columns[0]
    y_column = df.columns[-1]
    title = f"Quantum occupation ({band}, {region})"

    if interactive:
        go = _import_plotly_go()
        fig = go.Figure(go.Bar(x=df[x_column], y=df[y_column], name=y_column))
        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white",
            height=500,
        )
        if yscale == "log":
            fig.update_yaxes(type="log")
        return _display_plotly_figure(fig)

    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df[x_column], df[y_column])
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_yscale(yscale)
    fig.tight_layout()
    return fig


def read_quantum_energy_spectrum(
    run_root: str | Path,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str = "k00000",
    bias: str | int | None = None,
) -> pd.DataFrame:
    path = resolve_quantum_output_file(
        run_root,
        f"energy_spectrum_{kpoint}",
        region=region,
        band=band,
        bias=bias,
        preferred_extensions=("dat",),
    )
    return read_dat_table(path)


def plot_quantum_energy_spectrum(
    run_root: str | Path,
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str = "k00000",
    bias: str | int | None = None,
    interactive: bool = True,
    markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
):
    df = read_quantum_energy_spectrum(run_root, region=region, band=band, kpoint=kpoint, bias=bias)
    x_column = df.columns[0]
    y_column = df.columns[-1]
    title = f"Quantum energy spectrum ({band}, {region}, {kpoint})"

    if interactive:
        go = _import_plotly_go()
        fig = go.Figure(go.Scatter(x=df[x_column], y=df[y_column], mode="markers+lines", name=y_column))
        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white",
            height=500,
        )
        return _display_plotly_figure(fig)

    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(df[x_column], df[y_column], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_quantum_linecut(
    run_root: str | Path,
    quantity: str = "density",
    *,
    cut: str = "1d_x_QD",
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias: str | int | None = None,
    variables: Sequence[str] | None = None,
    interactive: bool = True,
    yscale: str = "linear",
    markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
):
    path = resolve_quantum_output_file(
        run_root,
        quantity,
        region=region,
        band=band,
        cut=cut,
        bias=bias,
        preferred_extensions=("dat",),
    )
    if interactive:
        return plot_dat_interactive(path, variables=variables, yscale=yscale, markers=markers)
    return plot_dat(path, variables=variables, yscale=yscale, markers=markers)


def plot_quantum_plane(
    run_root: str | Path,
    quantity: str = "density",
    *,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias: str | int | None = None,
    variable: str | None = None,
    interactive: bool = False,
    log10: bool = False,
    x_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    y_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
):
    path = resolve_quantum_output_file(
        run_root,
        quantity,
        region=region,
        band=band,
        bias=bias,
        preferred_extensions=("fld",),
    )
    if interactive:
        return plot_fld_interactive(path, variable=variable, log10=log10, x_markers=x_markers, y_markers=y_markers)
    return plot_fld(path, variable=variable, log10=log10, x_markers=x_markers, y_markers=y_markers)


def plot_quantum_probabilities_linecut(
    run_root: str | Path,
    *,
    cut: str = "1d_x_QD",
    states: Sequence[int] = (1, 2, 3, 4),
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str = "k00000",
    bias: str | int | None = None,
    interactive: bool = True,
    markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
):
    path = resolve_quantum_output_file(
        run_root,
        f"probabilities_shift_{kpoint}",
        region=region,
        band=band,
        cut=cut,
        bias=bias,
        preferred_extensions=("dat",),
    )
    df = read_dat_table(path)
    x_column = df.columns[0]

    selected_columns: list[str] = []
    for state in states:
        energy_col = f"E_{state}[eV]"
        prob_candidates = [
            f"Psi^2_{state}[nm^-2]",
            f"Psi^2_{state}[nm^-1]",
            f"Psi^2_{state}",
        ]
        if energy_col in df.columns:
            selected_columns.append(energy_col)
        selected_columns.extend(column for column in prob_candidates if column in df.columns)

    title = f"Probabilities shift ({band}, {region}, {cut})"
    if interactive:
        go = _import_plotly_go()
        fig = go.Figure()
        for column in selected_columns:
            fig.add_trace(go.Scatter(x=df[x_column], y=df[column], mode="lines", name=column))
        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title="Mixed units",
            template="plotly_white",
            height=600,
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        )
        _add_plotly_x_markers(fig, markers)
        return _display_plotly_figure(fig)

    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for column in selected_columns:
        ax.plot(df[x_column], df[column], label=column)
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel("Mixed units")
    ax.grid(alpha=0.25)
    _add_matplotlib_x_markers(ax, markers)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_quantum_probability_plane(
    run_root: str | Path,
    *,
    state: int = 1,
    region: str = "c-Ge_QW",
    band: str = "HH",
    kpoint: str = "k00000",
    bias: str | int | None = None,
    interactive: bool = False,
    log10: bool = False,
    x_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
    y_markers: Sequence[Mapping[str, Any] | Sequence[Any]] | None = None,
):
    path = resolve_quantum_output_file(
        run_root,
        f"probabilities_shift_{kpoint}",
        region=region,
        band=band,
        bias=bias,
        preferred_extensions=("fld",),
    )
    dataset = load_output_file(path, prefer_nextnanopy=False)
    psi_name = dataset.get_variable(f"Psi^2_{state}").name
    if interactive:
        return plot_fld_interactive(
            path,
            variable=psi_name,
            log10=log10,
            prefer_nextnanopy=False,
            x_markers=x_markers,
            y_markers=y_markers,
        )
    return plot_fld(
        path,
        variable=psi_name,
        log10=log10,
        prefer_nextnanopy=False,
        x_markers=x_markers,
        y_markers=y_markers,
    )


__all__ = [
    "AxisData",
    "OutputDataset",
    "RunPaths",
    "VariableData",
    "build_run_name",
    "build_region_material_map",
    "build_sweep",
    "convergence_summary",
    "describe_output_file",
    "discover_sweep_runs",
    "extract_linecut",
    "extract_plane",
    "find_latest_run",
    "find_probability_peaks",
    "find_run_files",
    "find_runs_for_input",
    "find_sweep_subrun",
    "find_vtr_plane_extrema",
    "get_bias_dir",
    "get_bias_dirs",
    "get_output_directory",
    "get_run_paths",
    "integrated_density_region_columns",
    "list_variables",
    "load_input_file",
    "load_output_file",
    "load_sweep_outputs",
    "load_vtr_linecut",
    "load_vtr_plane",
    "make_timestamp_tag",
    "map_integrated_density_regions",
    "parse_fld",
    "plot_bias_linecut",
    "plot_bias_plane",
    "plot_bias_volume_3d",
    "plot_bias_volume_linecut",
    "plot_bias_volume_slice",
    "plot_convergence",
    "plot_dat",
    "plot_dat_interactive",
    "plot_fld",
    "plot_fld_interactive",
    "plot_integrated_density_hole",
    "plot_quantum_energy_spectrum",
    "plot_quantum_density_volume_3d",
    "plot_quantum_density_volume_linecut",
    "plot_quantum_density_volume_slice",
    "plot_quantum_linecut",
    "plot_quantum_occupation",
    "plot_quantum_plane",
    "plot_quantum_probabilities_linecut",
    "plot_quantum_probability_volume_3d",
    "plot_quantum_probability_volume_linecut",
    "plot_quantum_probability_plane",
    "plot_quantum_probability_volume_slice",
    "plot_structure_linecut",
    "plot_structure_plane",
    "plot_total_charges",
    "plot_vtr_linecut",
    "plot_vtr_slice",
    "plot_vtr_slice_interactive",
    "plot_vtr_volume_interactive",
    "read_convergence_table",
    "read_dat_table",
    "read_index_table",
    "read_integrated_density_hole",
    "read_quantum_energy_spectrum",
    "read_quantum_occupation",
    "read_total_charges",
    "resolve_run_root",
    "resolve_bias_output_file",
    "resolve_quantum_output_file",
    "resolve_quantum_probability_state_file",
    "resolve_structure_file",
    "run_input_file",
    "run_sweep",
    "sample_vtr_plane_at_points",
    "sanitize_run_component",
    "save_input_file",
    "set_input_variables",
]
