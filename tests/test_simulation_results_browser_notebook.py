import ast
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/analysis/simulation_results_browser.ipynb"
ALLOWED_CHANGES = {
    "notebooks/analysis/simulation_results_browser.ipynb",
    "tests/test_simulation_results_browser_notebook.py",
}
EXPECTED_HEADINGS = """# Nextnano Simulation Results Browser
## Setup
## User controls
## Validate and inspect run
## Structure and diagnostics
## Classical outputs
### Hole density
### Electrostatic potential
### Band edges
## Quantum outputs
### Quantum-calculated hole density
### Wavefunction probability
### Occupation and energy spectrum
## Multi-quantity line comparison
## Optional 3D exploration
## Selected figure export
## Summary""".splitlines()
CONTROL_NAMES = """RUN_DIRECTORY BIAS ANALYSE_QUANTUM_OUTPUTS SHOW_OPTIONAL_3D
EXPORT_FIGURES XY_PLANE_Z_NM XZ_PLANE_Y_NM X_LINE_FIXED_COORDINATES_NM
Z_LINE_FIXED_COORDINATES_NM""".split()
CONTROL_VALUES = (
    "bias_00000", True, False, False, -4.0, 0.0,
    {"y": 0.0, "z": -4.0}, {"x": 1150.0, "y": 0.0},
)
SLICE_APIS = {
    "plot_vtr_slice", "plot_bias_volume_slice",
    "plot_quantum_density_volume_slice",
    "plot_quantum_probability_volume_slice",
}
LINE_APIS = {
    "plot_bias_volume_linecut", "plot_quantum_density_volume_linecut",
    "plot_quantum_probability_volume_linecut",
}
THREE_D_APIS = {
    "plot_bias_volume_3d", "plot_quantum_density_volume_3d",
    "plot_quantum_probability_volume_3d",
}


def cell_source(cell):
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else value

def call_name(call):
    return (call.func.id if isinstance(call.func, ast.Name) else
            call.func.attr if isinstance(call.func, ast.Attribute) else None)

def assigned_name(node):
    target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
              else node.target if isinstance(node, ast.AnnAssign) else None)
    return target.id if isinstance(target, ast.Name) else None

assert NOTEBOOK_PATH.is_file(), f"Missing browser notebook: {NOTEBOOK_PATH}"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
CELLS = NOTEBOOK["cells"]
SOURCES = {index: cell_source(cell) for index, cell in enumerate(CELLS)}
CODE = {i: SOURCES[i] for i, cell in enumerate(CELLS) if cell["cell_type"] == "code"}
MARKDOWN = {
    i: SOURCES[i] for i, cell in enumerate(CELLS) if cell["cell_type"] == "markdown"
}
TREES = {index: ast.parse(text) for index, text in CODE.items()}
NODES = [(index, node) for index, tree in TREES.items() for node in ast.walk(tree)]
CALLS = [(index, node) for index, node in NODES if isinstance(node, ast.Call)]
TOP_ASSIGNMENTS = {}
for _index, _tree in TREES.items():
    for _node in _tree.body:
        if assigned_name(_node):
            TOP_ASSIGNMENTS.setdefault(assigned_name(_node), []).append((_index, _node))
ALL_CODE = "\n".join(CODE.values())
ALL_SOURCE = "\n".join(SOURCES.values())

def calls_to(name):
    return [(index, call) for index, call in CALLS if call_name(call) == name]

def one_assignment(name):
    matches = TOP_ASSIGNMENTS.get(name, [])
    assert len(matches) == 1, f"Expected one top-level assignment to {name}"
    return matches[0]

def heading_index(heading):
    matches = [i for i, text in MARKDOWN.items() if heading in text.splitlines()]
    assert len(matches) == 1, f"Expected one heading {heading!r}"
    return matches[0]

def section(heading):
    start = heading_index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    for index in range(start + 1, len(CELLS)):
        if any(
            (match := re.match(r"^(#{1,3})\s+", line))
            and len(match.group(1)) <= level
            for line in MARKDOWN.get(index, "").splitlines()
        ):
            return "\n".join(SOURCES[i] for i in range(start, index))
    return "\n".join(SOURCES[i] for i in range(start, len(CELLS)))

def guarded_ids(control):
    guards, body_ids, else_ids = [], set(), set()
    for index, node in NODES:
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == control):
            continue
        guards.append((index, node))
        body_ids.update(id(child) for item in node.body for child in ast.walk(item))
        else_ids.update(id(child) for item in node.orelse for child in ast.walk(item))
    return guards, body_ids, else_ids

def local_expression(index, expression, before):
    if not isinstance(expression, ast.Name):
        return expression
    matches = [
        node for node in ast.walk(TREES[index])
        if assigned_name(node) == expression.id and node.lineno < before
    ]
    assert matches, f"Cell {index} must assign {expression.id} before its call"
    return max(matches, key=lambda node: node.lineno).value

def resolved_literal(index, expression, before):
    try:
        return ast.literal_eval(local_expression(index, expression, before))
    except (TypeError, ValueError):
        return None

def argument(call, name, position=None):
    keywords = {item.arg: item.value for item in call.keywords if item.arg}
    if name in keywords:
        return keywords[name]
    if position is not None and len(call.args) > position:
        return call.args[position]
    raise AssertionError(f"{call_name(call)} must pass {name} explicitly")

def axes_for(api, quantity=None, variable=None):
    axes = set()
    for index, call in calls_to(api):
        if quantity is not None and resolved_literal(
                index, argument(call, "quantity", 1), call.lineno) != quantity:
            continue
        if variable is not None and resolved_literal(
                index, argument(call, "variable"), call.lineno) != variable:
            continue
        axis = "slice_axis" if api in SLICE_APIS else "axis"
        axes.add(resolved_literal(index, argument(call, axis), call.lineno))
    return axes

def selected_dicts():
    base = one_assignment("SELECTED_FIGURES")[1].value
    assert isinstance(base, ast.Dict)
    updates = [
        call.args[0] for _, call in calls_to("update")
        if isinstance(call.func, ast.Attribute)
        and ast.unparse(call.func.value) == "SELECTED_FIGURES"
        and call.args and isinstance(call.args[0], ast.Dict)
    ]
    return [base, *updates]


def test_outline_and_exact_single_control_cell():
    headings = [
        line.strip() for text in MARKDOWN.values() for line in text.splitlines()
        if re.match(r"^#{1,3}\s+\S", line)
    ]
    assert headings == EXPECTED_HEADINGS
    start = heading_index("## User controls")
    end = heading_index("## Validate and inspect run")
    indices = [index for index in CODE if start < index < end]
    assert len(indices) == 1
    body = TREES[indices[0]].body
    assert all(isinstance(node, ast.Assign) for node in body)
    assert [assigned_name(node) for node in body] == CONTROL_NAMES
    assert ast.unparse(body[0].value) == "Path('path/to/completed/run')"
    assert tuple(ast.literal_eval(node.value) for node in body[1:]) == CONTROL_VALUES
    controls_text = section("## User controls").casefold()
    assert "editable" in controls_text and "selected simulation" in controls_text


def test_analysis_only_validation_and_static_safety():
    banned = (
        "RUN_SIMULATION", "run_input_file", "generated_input", "latest", "mtime",
        "getmtime", "qd_design", "write_nextnano_input", "nextnanopy.config",
        "nextnano_analysis", "sys.path", "importlib.reload",
    )
    assert all(fragment not in ALL_CODE for fragment in banned)
    assert "/Users/" not in ALL_SOURCE
    assert re.search(r"[A-Za-z]:\\Users\\", ALL_SOURCE) is None
    assert all(x in ALL_CODE for x in ("Path.cwd()", ".parents", "pyproject.toml", "src"))
    validation = calls_to("validate_run_directory")
    assert len(validation) == 1
    index, call = validation[0]
    assert ast.unparse(call.args[0]) == "RUN_DIRECTORY"
    assert ast.unparse(argument(call, "bias")) == "BIAS"
    if "required_outputs" in {item.arg for item in call.keywords}:
        assert ast.literal_eval(argument(call, "required_outputs")) == ()
    assert any(
        assigned_name(node) == "RUN_DIRECTORY"
        and any(child is call for child in ast.walk(node.value))
        for node in ast.walk(TREES[index])
    )
    assert "RUN_NAME = RUN_DIRECTORY.name" in ALL_CODE
    forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    assert not any(isinstance(node, forbidden) for _, node in NODES)


def test_inventory_diagnostics_and_classical_quantum_coverage():
    inventory = (
        "job_done.txt", '"potential"', '"bandedges"', '"density_hole"', '"vtr"',
        "resolve_bias_output_file", "resolve_quantum_output_file",
        "resolve_quantum_probability_state_file", "list_variables",
    )
    assert all(token in ALL_SOURCE for token in inventory)
    assert any(
        isinstance(node, ast.ExceptHandler)
        and "FileNotFoundError" in ast.unparse(node)
        for _, node in NODES
    )
    diagnostics = section("## Structure and diagnostics")
    diagnostic_apis = (
        "resolve_structure_file", "load_vtr_plane", "plot_vtr_slice",
        "convergence_summary", "plot_convergence", "read_integrated_density_hole",
        "plot_integrated_density_hole", "read_total_charges", "plot_total_charges",
    )
    assert all(api in diagnostics for api in diagnostic_apis)
    assert {"z", "y"} <= axes_for("plot_vtr_slice")
    expected = {
        ("density_hole", "Hole_density"): ({"z", "y"}, {"x"}),
        ("potential", "Potential"): ({"z", "y"}, {"x", "z"}),
        ("bandedges", "HH"): ({"z", "y"}, {"x", "z"}),
    }
    for (quantity, variable), (planes, lines) in expected.items():
        assert planes <= axes_for("plot_bias_volume_slice", quantity, variable)
        assert lines <= axes_for("plot_bias_volume_linecut", quantity, variable)
    assert re.search(r"replace|editable", section("### Band edges").casefold())
    defaults = {
        "QUANTUM_REGION": "c-Ge_QW", "QUANTUM_BAND": "HH",
        "QUANTUM_KPOINT": "k00000", "QUANTUM_STATE": 1,
    }
    actual = {name: ast.literal_eval(one_assignment(name)[1].value) for name in defaults}
    assert actual == defaults
    assert {"z", "y"} <= axes_for("plot_quantum_density_volume_slice")
    assert "x" in axes_for("plot_quantum_density_volume_linecut")
    assert {"z", "y"} <= axes_for("plot_quantum_probability_volume_slice")
    assert "x" in axes_for("plot_quantum_probability_volume_linecut")
    table_apis = (
        "read_quantum_occupation", "plot_quantum_occupation",
        "read_quantum_energy_spectrum", "plot_quantum_energy_spectrum",
    )
    assert all(calls_to(api) for api in table_apis)


def test_spatial_cells_and_optional_paths_are_self_contained_and_guarded():
    spatial = [
        (index, call) for index, call in CALLS
        if call_name(call) in SLICE_APIS | LINE_APIS
    ]
    assert spatial
    for index, call in spatial:
        api = call_name(call)
        if api == "plot_vtr_slice":
            configured = argument(call, "path", 0)
        elif api.startswith("plot_bias_"):
            configured = argument(call, "quantity", 1)
        else:
            configured = ast.Name(id="QUANTITY")
        assert isinstance(configured, ast.Name)
        local_expression(index, configured, call.lineno)
        variable = argument(call, "variable")
        assert isinstance(variable, ast.Name)
        local_expression(index, variable, call.lineno)
        fields = (
            ("slice_axis", "slice_value") if api in SLICE_APIS
            else ("axis", "fixed_coords")
        )
        for field in fields:
            configured = argument(call, field)
            assert isinstance(configured, ast.Name)
            local_expression(index, configured, call.lineno)
        nearest = "slice_coordinate" if api in SLICE_APIS else "chosen_coords"
        assert nearest in CODE[index]
    for index, call in CALLS:
        if "interactive" in {item.arg for item in call.keywords}:
            assert resolved_literal(index, argument(call, "interactive"), call.lineno) is False
    quantum_guards, quantum_ids, quantum_else = guarded_ids("ANALYSE_QUANTUM_OUTPUTS")
    assert quantum_guards and any(node.orelse for _, node in quantum_guards)
    quantum_calls = [
        call for _, call in CALLS
        if (call_name(call) or "").startswith(
            ("resolve_quantum", "plot_quantum", "read_quantum")
        ) or call_name(call) == "find_probability_peaks"
    ]
    assert quantum_calls and all(id(call) in quantum_ids for call in quantum_calls)
    assert all(id(call) not in quantum_else for call in quantum_calls)
    guards_3d, ids_3d, _ = guarded_ids("SHOW_OPTIONAL_3D")
    assert guards_3d
    for api in THREE_D_APIS:
        calls = calls_to(api)
        assert calls and all(id(call) in ids_3d for _, call in calls)


def test_comparison_and_explicit_static_figure_selection():
    comparison = section("## Multi-quantity line comparison")
    tokens = (
        "multi_quantity_x_line_figure", "load_vtr_linecut",
        "X_LINE_FIXED_COORDINATES_NM", "HH", "Fermi", "density", "probability",
    )
    assert all(token in comparison for token in tokens)
    assert any(token in comparison for token in ("sharex", "twinx", "axes["))
    assert "np.interp" not in comparison
    assert "np.array_equal" not in comparison
    assert "COMMON_X_NM" not in comparison
    assert all(token in comparison for token in (
        "X_OVERLAP_MIN_NM", "X_OVERLAP_MAX_NM", "set_xlim",
        'HH_COMPARISON_LINE_DATA["chosen_coords"]',
        'for coordinate_name in ("y", "z"):',
    ))
    assert "max(" in comparison and "min(" in comparison
    for line_data_name in (
        "HH_COMPARISON_LINE_DATA",
        "FERMI_COMPARISON_LINE_DATA",
        "CLASSICAL_DENSITY_COMPARISON_LINE_DATA",
        "QUANTUM_DENSITY_COMPARISON_LINE_DATA",
        "PROBABILITY_COMPARISON_LINE_DATA",
    ):
        assert f'{line_data_name}["axis"]' in comparison
    mappings, values = selected_dicts(), []
    for mapping in mappings:
        for key, value in zip(mapping.keys, mapping.values):
            assert key is not None and isinstance(value, ast.Name)
            assert not any(x in value.id.upper() for x in ("3D", "PLOTLY", "TABLE"))
            values.append(value.id)
    assert len(values) >= 12 and "multi_quantity_x_line_figure" in values
    assert "None" not in "\n".join(ast.unparse(item) for item in mappings)
    assert "edit" in section("## Selected figure export").casefold()


def test_export_and_manifest_are_external_guarded_and_reproducible():
    assert ast.literal_eval(one_assignment("EXPORT_FIGURES")[1].value) is False
    root_source = ast.unparse(one_assignment("FIGURE_OUTPUT_ROOT")[1].value)
    root_tokens = (
        "os.environ.get('THESIS_FIGURE_ROOT'",
        "REPO_ROOT.parent / 'qpu-local-outputs' / 'refactoring' / 'figures'",
        ".expanduser()", ".resolve()",
    )
    assert all(token in root_source for token in root_tokens)
    directory = ast.unparse(one_assignment("FIGURE_DIRECTORY")[1].value)
    assert directory == "FIGURE_OUTPUT_ROOT / RUN_NAME / BIAS"
    assert ast.literal_eval(one_assignment("FIGURE_FORMATS")[1].value) == ("pdf", "png")
    assert ast.literal_eval(one_assignment("FIGURE_PNG_DPI")[1].value) >= 300
    assert all(token in ALL_CODE for token in (
        "FIGURE_OUTPUT_ROOT == REPO_ROOT",
        "REPO_ROOT in FIGURE_OUTPUT_ROOT.parents",
        "raise ValueError",
    ))
    guards, export_ids, export_else = guarded_ids("EXPORT_FIGURES")
    assert len(guards) == 1
    mkdirs, saves = calls_to("mkdir"), calls_to("savefig")
    assert len(mkdirs) == 1
    assert ast.unparse(mkdirs[0][1].func.value) == "FIGURE_DIRECTORY"
    assert id(mkdirs[0][1]) in export_ids
    assert saves and all(id(call) in export_ids for _, call in saves)
    export_source = ast.unparse(guards[0][1])
    assert all(token in export_source for token in (
        "SELECTED_FIGURES.items()", "FIGURE_FORMATS", "bbox_inches", "'tight'",
    ))
    assert re.search(r"Unsupported|unsupported", export_source)
    assert (
        "isinstance" in export_source and "Figure" in export_source
        or "type(" in export_source and "matplotlib.figure" in export_source
    )
    assert any(
        isinstance(node, ast.If) and "'png'" in ast.unparse(node.test)
        and "FIGURE_PNG_DPI" in ast.unparse(node.body)
        for node in ast.walk(guards[0][1])
    )
    assert "plt.show" not in export_source and "plt.close" not in export_source
    manifest_paths = [
        node for _, node in NODES if assigned_name(node) == "FIGURE_MANIFEST_PATH"
    ]
    assert len(manifest_paths) == 1 and id(manifest_paths[0]) in export_ids
    assert ast.unparse(manifest_paths[0].value) == (
        "FIGURE_DIRECTORY / 'figure_manifest.json'"
    )
    writes = [
        call for _, call in calls_to("write_text")
        if ast.unparse(call.func.value) == "FIGURE_MANIFEST_PATH"
    ]
    assert len(writes) == 1 and id(writes[0]) in export_ids
    assert id(writes[0]) not in export_else and "json.dumps" in ast.unparse(writes[0])
    candidates = []
    for node in (node for _, node in NODES if isinstance(node, ast.Dict)):
        keys = {
            key.value for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if id(node) in export_ids and {"run_directory", "run_name", "bias", "formats"} <= keys:
            candidates.append((node, keys))
    assert len(candidates) == 1
    manifest, keys = candidates[0]
    alternatives = (
        {"notebook_identifier"}, {"analysis_coordinates", "xy_plane_z_nm"},
        {"analyse_quantum_outputs", "quantum_analysis_enabled"},
        {"quantum", "quantum_configuration"}, {"selected_figure_stems"},
        {"figure_files", "relative_filenames"}, {"png_dpi", "figure_png_dpi"},
    )
    assert all(group & keys for group in alternatives)
    manifest_source = ast.unparse(manifest).casefold()
    assert not any(
        token in manifest_source
        for token in ("timestamp", "secret", "nextnanopy", "raw_array", "table")
    )
    assert "str(figure_path)" not in export_source


def test_notebook_is_clean():
    assert CELLS
    for index, cell in enumerate(CELLS):
        assert SOURCES[index].strip() and not cell.get("attachments")
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_only_authorized_files_changed():
    if not (ROOT / ".git").exists():
        return

    def git_lines(*args):
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return set(filter(None, result.stdout.splitlines()))

    changed = git_lines("diff", "--name-only", "HEAD", "--")
    changed |= git_lines("ls-files", "--others", "--exclude-standard")
    assert changed <= ALLOWED_CHANGES, sorted(changed - ALLOWED_CHANGES)
    assert not any(
        path.startswith(("src/", "notebooks/experiments/")) for path in changed
    )
