# nextnano_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =============================================================================
# Run folder utilities
# =============================================================================

@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def structure(self) -> Path:
        return self.root / "Structure"

    @property
    def strain(self) -> Path:
        return self.root / "Strain"

    def bias_dirs(self) -> list[Path]:
        return sorted([p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("bias_")])

    def bias0(self) -> Path:
        dirs = self.bias_dirs()
        if not dirs:
            raise FileNotFoundError(f"No bias_XXXXX folders found in {self.root}")
        return dirs[0]


def get_run_paths(folder_output: str | Path) -> RunPaths:
    return RunPaths(Path(folder_output))


# =============================================================================
# Basic readers
# =============================================================================

def read_dat(path: str | Path) -> pd.DataFrame:
    """Read nextnano *.dat whitespace tables."""
    return pd.read_csv(Path(path), delim_whitespace=True, comment="#")


# =============================================================================
# AVS/Express .fld (rectilinear, ASCII) support
# =============================================================================

@dataclass(frozen=True)
class FldMeta:
    nx: int
    nz: int
    veclen: int
    labels: list[str]
    skip_x: int
    skip_z: int
    var_skips: dict[int, int]   # 1-based variable index -> skip line
    lines: list[str]


def _fld_read_floats(lines: list[str], start_line: int, count: int) -> np.ndarray:
    vals: list[float] = []
    i = start_line
    while i < len(lines) and len(vals) < count:
        s = lines[i].strip()
        if s and not s.startswith("#"):
            vals.extend([float(v) for v in s.split()])
        i += 1
    if len(vals) < count:
        raise ValueError(f"Expected {count} floats from line {start_line}, got {len(vals)}")
    return np.asarray(vals[:count], dtype=float)


def parse_fld(path: str | Path, max_lines: int = 1500) -> FldMeta:
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()
    header = "\n".join(lines[:max_lines])

    def get_int(key: str) -> int:
        m = re.search(rf"^{key}\s*=\s*(\d+)\s*$", header, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Missing '{key} = ...' in {path.name}")
        return int(m.group(1))

    ndim = get_int("ndim")
    if ndim != 2:
        raise ValueError(f"Only ndim=2 supported, got ndim={ndim} in {path.name}")

    field_m = re.search(r"^field\s*=\s*(\w+)\s*$", header, flags=re.MULTILINE)
    if not field_m or field_m.group(1) != "rectilinear":
        raise ValueError(f"Only field=rectilinear supported, got {field_m.group(1) if field_m else None}")

    nx = get_int("dim1")
    nz = get_int("dim2")
    veclen = get_int("veclen")

    labels = re.findall(r'^label\s*=\s*(.+)\s*$', header, flags=re.MULTILINE)

    def get_skip(kind: str, idx: int) -> int:
        m = re.search(rf"^{kind}\s+{idx}\s+.*?\bskip=(\d+)\b", header, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Missing '{kind} {idx} ... skip=' in {path.name}")
        return int(m.group(1))

    skip_x = get_skip("coord", 1)
    skip_z = get_skip("coord", 2)

    var_skips: dict[int, int] = {}
    for i in range(1, veclen + 1):
        m = re.search(rf"^variable\s+{i}\s+.*?\bskip=(\d+)\b", header, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Missing 'variable {i} ... skip=' in {path.name}")
        var_skips[i] = int(m.group(1))

    return FldMeta(nx=nx, nz=nz, veclen=veclen, labels=labels, skip_x=skip_x, skip_z=skip_z,
                   var_skips=var_skips, lines=lines)


def load_fld_scalar(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load veclen=1 scalar fld -> x[nx], z[nz], Z[nz,nx]."""
    meta = parse_fld(path)
    if meta.veclen != 1:
        raise ValueError(f"{Path(path).name} has veclen={meta.veclen}, use load_fld_variable(...)")
    x = _fld_read_floats(meta.lines, meta.skip_x, meta.nx)
    z = _fld_read_floats(meta.lines, meta.skip_z, meta.nz)
    raw = _fld_read_floats(meta.lines, meta.var_skips[1], meta.nx * meta.nz)
    Z = raw.reshape((meta.nz, meta.nx))
    return x, z, Z


def load_fld_variable(path: str | Path, var_index_1based: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load one variable i from a multi-var fld (bandedges/probabilities) -> x,z,Z,label."""
    meta = parse_fld(path)
    if not (1 <= var_index_1based <= meta.veclen):
        raise ValueError(f"var_index out of range: 1..{meta.veclen}")
    x = _fld_read_floats(meta.lines, meta.skip_x, meta.nx)
    z = _fld_read_floats(meta.lines, meta.skip_z, meta.nz)
    raw = _fld_read_floats(meta.lines, meta.var_skips[var_index_1based], meta.nx * meta.nz)
    Z = raw.reshape((meta.nz, meta.nx))
    label = meta.labels[var_index_1based - 1] if meta.labels else f"var_{var_index_1based}"
    return x, z, Z, label


def find_fld_var_index_by_label_substring(path: str | Path, needle: str) -> int:
    """Return 1-based variable index whose label contains `needle`."""
    meta = parse_fld(path)
    if not meta.labels:
        raise ValueError(f"No labels found in {Path(path).name}")
    for i, lab in enumerate(meta.labels, start=1):
        if needle in lab:
            return i
    raise ValueError(f"Label containing '{needle}' not found in {Path(path).name}")


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_heatmap(x: np.ndarray, z: np.ndarray, Z: np.ndarray, title: str, cbar: str,
                 xlim=None, zlim=None, log10: bool = False) -> go.Figure:
    if log10:
        Zp = np.where(Z > 0, Z, np.nan)
        Z = np.log10(Zp)

    fig = go.Figure(go.Heatmap(x=x, y=z, z=Z, colorbar=dict(title=cbar)))
    fig.update_layout(
        title=title,
        xaxis_title="x (nm)",
        yaxis_title="z (nm)",
        template="plotly_white",
        height=700,
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    if zlim is not None:
        fig.update_yaxes(range=[zlim[0], zlim[1]])
    fig.show()
    return fig


def plot_lines_df(df: pd.DataFrame, xcol: str, ycols: list[str], title: str,
                  y_label: str = "", xlim=None) -> go.Figure:
    fig = go.Figure()
    for c in ycols:
        fig.add_trace(go.Scatter(x=df[xcol], y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        xaxis_title=xcol,
        yaxis_title=y_label,
        template="plotly_white",
        height=520,
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    fig.show()
    return fig


# =============================================================================
# Non-negotiable plots
# =============================================================================

# --- 1D ---
def plot_1d_potential(bias_dir: str | Path, xlim=None) -> go.Figure:
    df = read_dat(Path(bias_dir) / "potential.dat")
    return plot_lines_df(df, xcol=df.columns[0], ycols=[df.columns[-1]],
                         title="Potential (1D)", y_label="Potential (V)", xlim=xlim)


def plot_1d_bandedges(bias_dir: str | Path,
                      include=("HH[eV]", "LH[eV]", "SO[eV]", "electron_Fermi_level[eV]", "hole_Fermi_level[eV]"),
                      xlim=None) -> go.Figure:
    df = read_dat(Path(bias_dir) / "bandedges.dat")
    xcol = "x[nm]" if "x[nm]" in df.columns else df.columns[0]
    ycols = [c for c in include if c in df.columns]
    return plot_lines_df(df, xcol=xcol, ycols=ycols, title="Band edges + Fermi (1D)", y_label="Energy (eV)", xlim=xlim)


def plot_1d_carrier_density(bias_dir: str | Path, which: str = "density_hole.dat", xlim=None) -> go.Figure:
    df = read_dat(Path(bias_dir) / which)
    xcol = df.columns[0]
    ycol = df.columns[-1]
    return plot_lines_df(df, xcol=xcol, ycols=[ycol], title=which, y_label=ycol, xlim=xlim)


# --- 2D (fld) ---
def plot_2d_potential(bias_dir: str | Path, xlim=None, zlim=None) -> go.Figure:
    x, z, Phi = load_fld_scalar(Path(bias_dir) / "potential.fld")
    return plot_heatmap(x, z, Phi, title="Potential (2D)", cbar="Potential (V)", xlim=xlim, zlim=zlim)


def plot_2d_efield_norm(bias_dir: str | Path, xlim=None, zlim=None) -> go.Figure:
    x, z, En = load_fld_scalar(Path(bias_dir) / "electric_field_norm.fld")
    return plot_heatmap(x, z, En, title="|E| (2D)", cbar="E_norm (kV/cm)", xlim=xlim, zlim=zlim)


def plot_2d_density_hole(bias_dir: str | Path, log10: bool = True, xlim=None, zlim=None) -> go.Figure:
    x, z, p = load_fld_scalar(Path(bias_dir) / "density_hole.fld")
    return plot_heatmap(x, z, p, title="Hole density (2D)", cbar="log10(p)" if log10 else "p",
                        log10=log10, xlim=xlim, zlim=zlim)


def plot_2d_bandedges_component(bias_dir: str | Path, label_substring: str,
                                xlim=None, zlim=None) -> go.Figure:
    fld = Path(bias_dir) / "bandedges.fld"
    idx = find_fld_var_index_by_label_substring(fld, label_substring)
    x, z, Z, lab = load_fld_variable(fld, idx)
    return plot_heatmap(x, z, Z, title=f"Bandedge: {lab}", cbar=lab, xlim=xlim, zlim=zlim)


# =============================================================================
# Quantum plots (density + probabilities)
# =============================================================================

def plot_quantum_density_2d(run_root: str | Path, region: str = "c-Ge_QW", band: str = "HH",
                            xlim=None, zlim=None, log10: bool = True) -> go.Figure:
    run_root = Path(run_root)
    bias0 = RunPaths(run_root).bias0()
    fld = bias0 / "Quantum" / region / band / "density.fld"
    x, z, rho = load_fld_scalar(fld)
    return plot_heatmap(x, z, rho, title=f"Quantum density {band} ({region})",
                        cbar="log10(density)" if log10 else "density",
                        log10=log10, xlim=xlim, zlim=zlim)


def plot_probabilities_1d_mixed(prob_1d_path: str | Path, n_states: int = 5,
                               title: str = "", xlim=None) -> go.Figure:
    """
    For probabilities_shift_k00000_1d_x_QD.dat / _1d_z_QD.dat
    plots E_n (as horizontal lines) + Psi^2_n on the SAME axis (mixed units).
    Supports Psi^2_* columns in nm^-2.
    """
    df = read_dat(prob_1d_path)
    pos_col = df.columns[0]
    pos = df[pos_col]

    fig = go.Figure()

    for i in range(1, n_states + 1):
        ecol = f"E_{i}[eV]"
        if ecol in df.columns:
            Ei = float(df.loc[0, ecol])
            fig.add_trace(go.Scatter(
                x=[pos.min(), pos.max()],
                y=[Ei, Ei],
                mode="lines",
                name=ecol,
                line=dict(dash="dot"),
            ))

    for i in range(1, n_states + 1):
        # accept nm^-2 or fallback
        candidates = [f"Psi^2_{i}[nm^-2]", f"Psi^2_{i}[nm^-1]", f"Psi^2_{i}"]
        pcol = next((c for c in candidates if c in df.columns), None)
        if pcol is None:
            continue
        fig.add_trace(go.Scatter(x=pos, y=df[pcol], mode="lines", name=pcol))

    fig.update_layout(
        title=title or Path(prob_1d_path).name,
        xaxis_title=pos_col,
        yaxis_title="Mixed units (E in eV, Psi^2 in nm^-2)",
        template="plotly_white",
        height=600,
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    fig.show()
    return fig


def plot_probability_2d_state(prob_fld_path: str | Path, state: int = 1,
                              xlim=None, zlim=None, log10: bool = False) -> float:
    """
    For probabilities_shift_k00000.fld with veclen=100:
      variables 1..50  : E_1..E_50 (constant fields)
      variables 51..100: Psi^2_1..Psi^2_50
    Returns E_state (eV).
    """
    prob_fld_path = Path(prob_fld_path)
    if not (1 <= state <= 50):
        raise ValueError("state must be in 1..50")

    # E_n is variable n
    _, _, Efield, _ = load_fld_variable(prob_fld_path, state)
    En = float(Efield.flat[0])

    # Psi^2_n is variable 50 + n
    x, z, P, Plab = load_fld_variable(prob_fld_path, 50 + state)
    plot_heatmap(x, z, P, title=f"{Plab}  (E{state}={En:.6g} eV)", cbar=Plab, xlim=xlim, zlim=zlim, log10=log10)

    return En


# =============================================================================
# Sweeps (nextnanopy)
# =============================================================================

def run_input(input_path: str | Path):
    """Execute one input file and return nextnanopy InputFile object."""
    import nextnanopy as nn
    inp = nn.InputFile(str(input_path))
    inp.execute()
    return inp


def sweep_single_var(input_path: str | Path, var: str, values: list[float | int],
                     name: str | None = None,
                     delete_old_files: bool = True,
                     delete_input_files: bool = False,
                     overwrite: bool = True,
                     show_log: bool = True,
                     convergenceCheck: bool = True,
                     parallel_limit: int = 1):
    """
    Run a nextnanopy Sweep for a single variable using your working API:
      sw = nn.Sweep({var: values}, inp_path)
      sw.save_sweep(...)
      sw.execute_sweep(...)

    Returns: the Sweep object (so you can inspect paths/attributes).
    """
    import nextnanopy as nn
    input_path = str(input_path)

    sweep_variables = {var: list(values)}
    sw = nn.Sweep(sweep_variables, input_path)

    # generate sweep input files
    sw.save_sweep(delete_old_files=delete_old_files)

    # run sweep
    sw.execute_sweep(
        delete_input_files=delete_input_files,
        overwrite=overwrite,
        show_log=show_log,
        convergenceCheck=convergenceCheck,
        parallel_limit=parallel_limit
    )

    return sw


def sweep_multi_var(input_path: str | Path, variables: dict[str, list[float | int]],
                    name: str | None = None,
                    delete_old_files: bool = True,
                    delete_input_files: bool = False,
                    overwrite: bool = True,
                    show_log: bool = True,
                    convergenceCheck: bool = True,
                    parallel_limit: int = 1):
    """
    Multi-var sweep using the same nn.Sweep API.
    variables = {"V_PG":[...], "V_BG":[...], ...}

    Returns: the Sweep object.
    """
    import nextnanopy as nn
    input_path = str(input_path)

    sw = nn.Sweep({k: list(v) for k, v in variables.items()}, input_path)
    sw.save_sweep(delete_old_files=delete_old_files)
    sw.execute_sweep(
        delete_input_files=delete_input_files,
        overwrite=overwrite,
        show_log=show_log,
        convergenceCheck=convergenceCheck,
        parallel_limit=parallel_limit
    )
    return sw