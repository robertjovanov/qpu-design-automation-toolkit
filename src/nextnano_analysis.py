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
    return pd.read_csv(Path(path), sep=r"\s+", comment="#")


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

def sweep_with_nn_sweep(
    input_path: str | Path,
    sweep_variables: dict[str, list[float | int]],
    delete_old_files: bool = True,
    delete_input_files: bool = False,
    overwrite: bool = True,
    show_log: bool = True,
    convergenceCheck: bool = True,
    parallel_limit: int = 1,
):
    """
    Run a sweep using nextnanopy's nn.Sweep API (the one you already use):
      sw = nn.Sweep(sweep_variables, inp_path)
      sw.save_sweep(...)
      sw.execute_sweep(...)
    Returns the Sweep object.
    """
    import nextnanopy as nn

    inp_path = str(input_path)
    sw = nn.Sweep(sweep_variables, inp_path)

    sw.save_sweep(delete_old_files=delete_old_files)
    sw.execute_sweep(
        delete_input_files=delete_input_files,
        overwrite=overwrite,
        show_log=show_log,
        convergenceCheck=convergenceCheck,
        parallel_limit=parallel_limit,
    )
    return sw

def sweep_subrun_dir(sweep_root: str | Path, var: str, value: float) -> Path:
    """
    Return the subrun folder in a nextnanopy sweep directory matching __{var}_{value}_.
    Example folder name: ...__V_PG_-3.0_
    """
    sweep_root = Path(sweep_root)
    target = f"__{var}_{value}_"
    matches = [p for p in sweep_root.iterdir() if p.is_dir() and target in p.name]
    if not matches:
        raise FileNotFoundError(f"No subrun folder containing '{target}' under {sweep_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for '{target}': {[m.name for m in matches]}")
    return matches[0]

def compare_qw_sheet_density(
    run_dir: str | Path,
    zmin_nm: float = -15.0,
    zmax_nm: float = 0.0,
    bias_folder: str = "bias_00000",
    integrated_file: str = "integrated_density_hole.dat",
    integrated_region_col: str = "region_3[carriers/cm^2]",
    density_file: str = "density_hole.dat",
    verbose: bool = True,
) -> dict:
    """
    Compare:
      (A) nextnano-reported integrated sheet density (carriers/cm^2) from run_root/integrated_density_hole.dat
    vs
      (B) numerical integral of volumetric hole density p(z) from run_root/bias_00000/density_hole.dat over z in [zmin_nm, zmax_nm].

    Units:
      - integrated_density_hole.dat: carriers/cm^2
      - density_hole.dat column Hole_density[1e18_cm^-3] means:
            p_true[cm^-3] = p_file * 1e18
      - 1 nm = 1e-7 cm
      - sheet density: p_s[cm^-2] = ∫ p(z)[cm^-3] dz[cm]
    """
    from pathlib import Path
    import numpy as np

    run_dir = Path(run_dir)
    bias_dir = run_dir / bias_folder

    # --- read integrated sheet density from RUN ROOT ---
    df_int = read_dat(run_dir / integrated_file)
    if integrated_region_col not in df_int.columns:
        raise ValueError(
            f"Column '{integrated_region_col}' not found in {run_dir/integrated_file}.\n"
            f"Available columns: {df_int.columns.tolist()}"
        )
    sheet_int = float(df_int.loc[0, integrated_region_col])  # carriers/cm^2

    # --- read volumetric density from BIAS folder and integrate ---
    df_den = read_dat(bias_dir / density_file)
    zcol = df_den.columns[0]

    hole_col = next((c for c in df_den.columns if "Hole_density" in c), None)
    if hole_col is None:
        raise ValueError(
            f"No 'Hole_density' column found in {bias_dir/density_file}.\n"
            f"Available columns: {df_den.columns.tolist()}"
        )

    z_nm = df_den[zcol].to_numpy(dtype=float)
    p_file = df_den[hole_col].to_numpy(dtype=float)  # in units of 1e18 cm^-3

    lo, hi = (zmin_nm, zmax_nm) if zmin_nm <= zmax_nm else (zmax_nm, zmin_nm)
    mask = (z_nm >= lo) & (z_nm <= hi)
    if not np.any(mask):
        raise ValueError(f"No points found in z-window [{lo}, {hi}] nm. Check your z range / file.")

    z_nm_w = z_nm[mask]
    p_file_w = p_file[mask]

    p_cm3 = p_file_w * 1e18
    z_cm = z_nm_w * 1e-7
    sheet_num = float(np.trapz(p_cm3, z_cm))  # carriers/cm^2

    abs_err = sheet_num - sheet_int
    rel_err = abs_err / sheet_int if sheet_int != 0 else np.nan

    out = dict(
        run_dir=str(run_dir),
        bias_dir=str(bias_dir),
        z_window_nm=(lo, hi),
        integrated_region_col=integrated_region_col,
        sheet_from_integrated_cm2=sheet_int,
        sheet_from_density_integral_cm2=sheet_num,
        abs_error_cm2=abs_err,
        rel_error=rel_err,
        density_col=hole_col,
        z_col=zcol,
        n_points=int(mask.sum()),
    )

    if verbose:
        print(f"Run: {run_dir.name}")
        print(f"QW window: z ∈ [{lo}, {hi}] nm  (N={out['n_points']})")
        print(f"Integrated file: {sheet_int:.6e} carriers/cm^2  ({integrated_region_col})")
        print(f"Numerical ∫p(z)dz: {sheet_num:.6e} carriers/cm^2  (from {bias_folder}/{density_file})")
        print(f"Abs error: {abs_err:.6e}  |  Rel error: {rel_err:.6e}")

    return out

def plot_quantum_occupation(
    run_dir: str | Path,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias_folder: str = "bias_00000",
    yscale: str = "linear",
):
    """
    Plot occupation.dat for a given quantum region/band.
    Returns a Plotly figure.
    """
    run_dir = Path(run_dir)
    occ_path = run_dir / bias_folder / "Quantum" / region / band / "occupation.dat"
    df = read_dat(occ_path)

    state_col = df.columns[0]
    occ_col = next((c for c in df.columns if "Occupation" in c), df.columns[-1])

    fig = go.Figure(go.Bar(
        x=df[state_col],
        y=df[occ_col],
        name=occ_col
    ))

    fig.update_layout(
        title=f"Quantum occupation ({band}, {region})",
        xaxis_title=state_col,
        yaxis_title=occ_col,
        template="plotly_white",
        height=500,
    )
    if yscale == "log":
        fig.update_yaxes(type="log")

    return fig


def plot_quantum_density_1d(
    run_dir: str | Path,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias_folder: str = "bias_00000",
    xlim=None,
):
    """
    Plot 1D quantum density from density.dat for a given quantum region/band.
    Returns a Plotly figure.
    """
    run_dir = Path(run_dir)
    den_path = run_dir / bias_folder / "Quantum" / region / band / "density.dat"
    df = read_dat(den_path)

    xcol = df.columns[0]
    ycol = df.columns[-1]

    return plot_lines_df(
        df,
        xcol=xcol,
        ycols=[ycol],
        title=f"Quantum density ({band}, {region})",
        y_label=ycol,
        xlim=xlim,
    )

def plot_1d_electric_field(bias_dir: str | Path, xlim=None):
    """
    Plot electric_field.dat from a 1D run.
    """
    df = read_dat(Path(bias_dir) / "electric_field.dat")
    return plot_lines_df(
        df,
        xcol=df.columns[0],
        ycols=[df.columns[-1]],
        title="Electric field (1D)",
        y_label=df.columns[-1],
        xlim=xlim,
    )

def plot_1d_summary_presentation(
    run_dir: str | Path,
    region: str = "c-Ge_QW",
    band: str = "HH",
    bias_folder: str = "bias_00000",
    n_states: int = 2,
    xlim=None,
    density_zero_offset_ev: float = 0.1,
    density_x_shift_nm: float = 3.5,
):
    """
    Plot raw 1D QW quantities directly from nextnano output files, with presentation colors:
      - HH: black
      - LH: black dashed
      - electron / hole Fermi levels: cyan
      - E_1 / Psi^2_1: red
      - E_2 / Psi^2_2: blue
      - quantum hole density: green (right axis)

    Left y-axis: energies + raw Psi^2_i
    Right y-axis: hole density, with density=0 aligned to (hole_Fermi_level - density_zero_offset_ev)
    The density x-coordinate is shifted by density_x_shift_nm for this presentation plot only.
    """
    run_dir = Path(run_dir)
    b0 = run_dir / bias_folder

    # --- band edges ---
    df_be = read_dat(b0 / "bandedges.dat")
    xcol_be = df_be.columns[0]

    # --- quantum hole density ---
    df_den = read_dat(b0 / "Quantum" / region / band / "density.dat")
    xcol_den = df_den.columns[0]
    hole_col = df_den.columns[-1]
    hole_label = "Hole_density[1e18_cm^-3]" if hole_col == "Density[1e18_cm^-3]" else hole_col
    density_x = df_den[xcol_den].to_numpy(dtype=float) + density_x_shift_nm

    # --- probabilities / eigenenergies ---
    prob_path = b0 / "Quantum" / region / band / "probabilities_shift_k00000.dat"
    df_prob = read_dat(prob_path)
    xcol_prob = df_prob.columns[0]

    fig = go.Figure()

    y1_vals = []

    # HH and LH
    if "HH[eV]" in df_be.columns:
        y = df_be["HH[eV]"].to_numpy()
        y1_vals.append(y)
        fig.add_trace(go.Scatter(
            x=df_be[xcol_be], y=y,
            mode="lines", name="HH[eV]",
            line=dict(color="black", width=2),
            yaxis="y1"
        ))

    if "LH[eV]" in df_be.columns:
        y = df_be["LH[eV]"].to_numpy()
        y1_vals.append(y)
        fig.add_trace(go.Scatter(
            x=df_be[xcol_be], y=y,
            mode="lines", name="LH[eV]",
            line=dict(color="black", width=2, dash="dash"),
            yaxis="y1"
        ))

    # Fermi levels (cyan)
    for col, dash in [("electron_Fermi_level[eV]", "dot"), ("hole_Fermi_level[eV]", "solid")]:
        if col in df_be.columns:
            y = df_be[col].to_numpy()
            y1_vals.append(y)
            fig.add_trace(go.Scatter(
                x=df_be[xcol_be], y=y,
                mode="lines", name=col,
                line=dict(color="cyan", width=2, dash=dash),
                yaxis="y1"
            ))

    # Hole density on right axis (green)
    density_vals = df_den[hole_col].to_numpy()
    fig.add_trace(go.Scatter(
        x=density_x, y=density_vals,
        mode="lines", name=hole_label,
        line=dict(color="green", width=2),
        yaxis="y2"
    ))

    # State colors
    state_colors = {
        1: "red",
        2: "blue",
    }

    # Raw eigenenergies + raw probabilities on same left axis
    for i in range(1, n_states + 1):
        color = state_colors.get(i, None)

        ecol = f"E_{i}[eV]"
        if ecol in df_prob.columns:
            y = df_prob[ecol].to_numpy()
            y1_vals.append(y)
            fig.add_trace(go.Scatter(
                x=df_prob[xcol_prob],
                y=y,
                mode="lines",
                name=ecol,
                line=dict(color=color, width=2, dash="dot") if color else dict(width=2, dash="dot"),
                yaxis="y1"
            ))

        candidates = [
            f"Psi^2_{i}[nm^-2]",
            f"Psi^2_{i}[nm^-1]",
            f"Psi^2_{i}",
        ]
        pcol = next((c for c in candidates if c in df_prob.columns), None)
        if pcol is not None:
            y = df_prob[pcol].to_numpy()
            y1_vals.append(y)
            fig.add_trace(go.Scatter(
                x=df_prob[xcol_prob],
                y=y,
                mode="lines",
                name=pcol,
                line=dict(color=color, width=2) if color else dict(width=2),
                yaxis="y1"
            ))

    # QW boundaries
    for z0 in (-15.0, 0.0):
        fig.add_vline(x=z0, line_width=2, line_dash="dot", line_color="gray")

    def visible_values(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if xlim is not None:
            mask = (x >= xlim[0]) & (x <= xlim[1])
            y = y[mask]
        y = y[np.isfinite(y)]
        return y

    # --- compute y1 range manually ---
    y1_visible = []
    for trace in fig.data:
        if getattr(trace, "yaxis", None) in ("y", "y1"):
            vals = visible_values(trace.x, trace.y)
            if vals.size:
                y1_visible.append(vals)
    y1_all = np.concatenate(y1_visible or [np.asarray(v, dtype=float).ravel() for v in y1_vals])
    y1_min = float(np.nanmin(y1_all))
    y1_max = float(np.nanmax(y1_all))
    y1_pad = 0.05 * (y1_max - y1_min + 1e-12)
    y1_min_plot = y1_min - y1_pad
    y1_max_plot = y1_max + y1_pad

    # --- align density=0 to (hole_Fermi_level - density_zero_offset_ev) ---
    if "hole_Fermi_level[eV]" in df_be.columns:
        efh_values = visible_values(df_be[xcol_be], df_be["hole_Fermi_level[eV]"])
        efh_ref = float(np.mean(efh_values)) if efh_values.size else float(np.mean(df_be["hole_Fermi_level[eV]"].to_numpy()))
    else:
        efh_ref = y1_max

    y_ref = efh_ref - density_zero_offset_ev
    if density_zero_offset_ev > 0:
        y1_min_plot = min(y1_min_plot, y_ref - density_zero_offset_ev)

    t = (y_ref - y1_min_plot) / (y1_max_plot - y1_min_plot)
    t = min(max(t, 1e-6), 1 - 1e-6)  # keep safe

    density_visible = visible_values(density_x, density_vals)
    density_peak = float(np.nanmax(density_visible)) if density_visible.size else float(np.nanmax(density_vals))
    y2_max = density_peak * 1.05 if density_peak > 0 else 1.0
    y2_min = -t * y2_max / (1 - t)

    fig.update_layout(
        title=f"Valence band edge profiles and hole density of 2DHG structure",
        xaxis_title="z (nm)",
        yaxis=dict(
            title="Energy (eV) + raw Psi²",
            range=[y1_min_plot, y1_max_plot]
        ),
        yaxis2=dict(
            title=hole_label,
            overlaying="y",
            side="right",
            range=[y2_min, y2_max]
        ),
        template="plotly_white",
        height=650,
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )

    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])

    return fig

def plot_dat_file(path: str | Path, ycols: list[str] | None = None, title: str | None = None, xlim=None):
    """
    Generic helper: read a nextnano .dat file and plot selected columns vs the first column.
    If ycols is None, all columns except the first are plotted.
    """
    path = Path(path)
    df = read_dat(path)
    xcol = df.columns[0]

    if ycols is None:
        ycols = list(df.columns[1:])

    return plot_lines_df(
        df,
        xcol=xcol,
        ycols=ycols,
        title=title or path.name,
        y_label="",
        xlim=xlim,
    )
