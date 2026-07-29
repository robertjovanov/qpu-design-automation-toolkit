import ast
import json
import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "notebooks"
    / "experiments"
    / "03_generate_nextnano_input_from_phidl_layout.ipynb"
)

EXPECTED_BASE_REQUIRED_OUTPUTS = (
    "Structure/materials.vtr",
    "potential.vtr",
    "bandedges.vtr",
    "density_hole.vtr",
    "iteration_quantum_poisson.dat",
    "integrated_density_hole.dat",
    "total_charges.txt",
)

EXPECTED_QUANTUM_REQUIRED_OUTPUTS = (
    "Quantum/c-Ge_QW/HH/density.vtr",
    "Quantum/c-Ge_QW/HH/probability_shift_k00000_0001.vtr",
    "Quantum/c-Ge_QW/HH/occupation.dat",
    "Quantum/c-Ge_QW/HH/energy_spectrum_k00000.dat",
)

DEFAULT_CLASSICAL_FIGURES = {
    "HOLE_DENSITY_PLANE_FIGURE": {
        "plot_api": "plot_bias_volume_slice",
        "loader_api": "load_vtr_plane",
        "data_name": "HOLE_DENSITY_PLANE_DATA",
        "path": "DENSITY_HOLE_VTR",
        "quantity": "density_hole",
        "variable": "Hole_density",
        "axis_name": "SLICE_AXIS",
        "axis": "z",
        "coordinate_name": "SLICE_COORDINATE_NM",
        "coordinate": "XY_PLANE_Z_NM",
    },
    "HOLE_DENSITY_LINE_FIGURE": {
        "plot_api": "plot_bias_volume_linecut",
        "loader_api": "load_vtr_linecut",
        "data_name": "HOLE_DENSITY_LINE_DATA",
        "path": "DENSITY_HOLE_VTR",
        "quantity": "density_hole",
        "variable": "Hole_density",
        "axis_name": "LINE_AXIS",
        "axis": "x",
        "coordinate_name": "FIXED_COORDINATES_NM",
        "coordinates": {
            "y": "X_LINE_Y_NM",
            "z": "X_LINE_Z_NM",
        },
    },
    "POTENTIAL_PLANE_FIGURE": {
        "plot_api": "plot_bias_volume_slice",
        "loader_api": "load_vtr_plane",
        "data_name": "POTENTIAL_PLANE_DATA",
        "path": "POTENTIAL_VTR",
        "quantity": "potential",
        "variable": "Potential",
        "axis_name": "SLICE_AXIS",
        "axis": "z",
        "coordinate_name": "SLICE_COORDINATE_NM",
        "coordinate": "XY_PLANE_Z_NM",
    },
    "POTENTIAL_XZ_PLANE_FIGURE": {
        "plot_api": "plot_bias_volume_slice",
        "loader_api": "load_vtr_plane",
        "data_name": "POTENTIAL_XZ_PLANE_DATA",
        "path": "POTENTIAL_VTR",
        "quantity": "potential",
        "variable": "Potential",
        "axis_name": "SLICE_AXIS",
        "axis": "y",
        "coordinate_name": "SLICE_COORDINATE_NM",
        "coordinate": "XZ_PLANE_Y_NM",
    },
    "POTENTIAL_LINE_FIGURE": {
        "plot_api": "plot_bias_volume_linecut",
        "loader_api": "load_vtr_linecut",
        "data_name": "POTENTIAL_LINE_DATA",
        "path": "POTENTIAL_VTR",
        "quantity": "potential",
        "variable": "Potential",
        "axis_name": "LINE_AXIS",
        "axis": "x",
        "coordinate_name": "FIXED_COORDINATES_NM",
        "coordinates": {
            "y": "X_LINE_Y_NM",
            "z": "X_LINE_Z_NM",
        },
    },
    "HH_BAND_PLANE_FIGURE": {
        "plot_api": "plot_bias_volume_slice",
        "loader_api": "load_vtr_plane",
        "data_name": "HH_BAND_PLANE_DATA",
        "path": "BANDEDGES_VTR",
        "quantity": "bandedges",
        "variable": "HH",
        "axis_name": "SLICE_AXIS",
        "axis": "z",
        "coordinate_name": "SLICE_COORDINATE_NM",
        "coordinate": "XY_PLANE_Z_NM",
    },
    "HH_BAND_XZ_PLANE_FIGURE": {
        "plot_api": "plot_bias_volume_slice",
        "loader_api": "load_vtr_plane",
        "data_name": "HH_BAND_XZ_PLANE_DATA",
        "path": "BANDEDGES_VTR",
        "quantity": "bandedges",
        "variable": "HH",
        "axis_name": "SLICE_AXIS",
        "axis": "y",
        "coordinate_name": "SLICE_COORDINATE_NM",
        "coordinate": "XZ_PLANE_Y_NM",
    },
    "HH_BAND_LINE_FIGURE": {
        "plot_api": "plot_bias_volume_linecut",
        "loader_api": "load_vtr_linecut",
        "data_name": "HH_BAND_LINE_DATA",
        "path": "BANDEDGES_VTR",
        "quantity": "bandedges",
        "variable": "HH",
        "axis_name": "LINE_AXIS",
        "axis": "x",
        "coordinate_name": "FIXED_COORDINATES_NM",
        "coordinates": {
            "y": "X_LINE_Y_NM",
            "z": "X_LINE_Z_NM",
        },
    },
    "HH_BAND_Z_LINE_FIGURE": {
        "plot_api": "plot_bias_volume_linecut",
        "loader_api": "load_vtr_linecut",
        "data_name": "HH_BAND_Z_LINE_DATA",
        "path": "BANDEDGES_VTR",
        "quantity": "bandedges",
        "variable": "HH",
        "axis_name": "LINE_AXIS",
        "axis": "z",
        "coordinate_name": "FIXED_COORDINATES_NM",
        "coordinates": {
            "x": "Z_LINE_X_NM",
            "y": "Z_LINE_Y_NM",
        },
    },
}

REMOVED_CLASSICAL_WRAPPERS = {
    "_format_fixed_coords_for_title",
    "plot_default_1d_diagnostic",
}


def cell_source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def call_name(call):
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def keyword_map(call):
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}


def assignment_name(node):
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def assignment_value(node):
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    raise TypeError(f"Unsupported assignment node: {type(node).__name__}")


def call_argument(call, position, keyword):
    if len(call.args) > position:
        return call.args[position]
    return keyword_map(call).get(keyword)


def literal_or_none(node):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return None


def contains_name(node, name):
    return any(
        isinstance(descendant, ast.Name) and descendant.id == name
        for descendant in ast.walk(node)
    )


class GenerateNextnanoInputNotebookClassicalOutputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_notebook = NOTEBOOK_PATH.read_text(encoding="utf-8")
        cls.notebook = json.loads(cls.raw_notebook)
        cls.cells = cls.notebook["cells"]
        cls.code_cells = [
            (index, cell)
            for index, cell in enumerate(cls.cells)
            if cell["cell_type"] == "code"
        ]
        cls.code_sources = {
            index: cell_source(cell) for index, cell in cls.code_cells
        }
        cls.trees = {}
        for index, source in cls.code_sources.items():
            try:
                cls.trees[index] = ast.parse(source)
            except SyntaxError as exc:
                raise AssertionError(
                    f"Notebook code cell {index} is not valid Python: {exc}"
                ) from exc

        cls.top_assignments = {}
        for index, tree in cls.trees.items():
            for node in tree.body:
                name = assignment_name(node)
                if name is not None:
                    cls.top_assignments.setdefault(name, []).append((index, node))

        cls.calls = [
            (index, node)
            for index, tree in cls.trees.items()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        cls.combined_code = "\n".join(cls.code_sources.values())
        cls.combined_source = "\n".join(
            cell_source(cell) for cell in cls.cells
        )

        markdown = {
            index: cell_source(cell)
            for index, cell in enumerate(cls.cells)
            if cell["cell_type"] == "markdown"
        }
        classical_headings = [
            index
            for index, source in markdown.items()
            if re.search(
                r"^##(?!#)\s+Classical outputs\s*$",
                source,
                flags=re.MULTILINE,
            )
        ]
        if len(classical_headings) != 1:
            raise AssertionError(
                "Expected one H2 heading named 'Classical outputs'"
            )
        cls.classical_start = classical_headings[0]

        quantum_headings = [
            index
            for index, source in markdown.items()
            if index > cls.classical_start
            and "quantum outputs" in source.lower()
        ]
        if len(quantum_headings) != 1:
            raise AssertionError(
                "Expected one later markdown heading containing 'quantum outputs'"
            )
        cls.quantum_start = quantum_headings[0]

        cls.classical_code_indices = [
            index
            for index, _ in cls.code_cells
            if cls.classical_start < index < cls.quantum_start
        ]
        cls.classical_trees = {
            index: cls.trees[index] for index in cls.classical_code_indices
        }
        cls.classical_calls = [
            (index, node)
            for index in cls.classical_code_indices
            for node in ast.walk(cls.trees[index])
            if isinstance(node, ast.Call)
        ]
        cls.classical_code = "\n".join(
            cls.code_sources[index] for index in cls.classical_code_indices
        )
        cls.classical_source = "\n".join(
            cell_source(cell)
            for index, cell in enumerate(cls.cells)
            if cls.classical_start <= index < cls.quantum_start
        )

    def one_top_assignment(self, name):
        matches = self.top_assignments.get(name, [])
        self.assertEqual(
            len(matches),
            1,
            f"Expected one top-level assignment to {name}",
        )
        return matches[0]

    def calls_named(self, name, *, classical_only=False):
        calls = self.classical_calls if classical_only else self.calls
        return [
            (index, call)
            for index, call in calls
            if call_name(call) == name
        ]

    def guarded_node_ids(self, flag):
        guarded = set()
        for tree in self.classical_trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and contains_name(node.test, flag):
                    guarded.update(id(descendant) for descendant in ast.walk(node))
        return guarded

    def resolved_literal(self, node):
        if isinstance(node, ast.Name):
            matches = self.top_assignments.get(node.id, [])
            if len(matches) == 1:
                return literal_or_none(assignment_value(matches[0][1]))
        return literal_or_none(node)

    def classical_output_pair(self, call):
        quantity = literal_or_none(call_argument(call, 1, "quantity"))
        variable = self.resolved_literal(keyword_map(call).get("variable"))
        return quantity, variable

    def metadata_output_pair(self, call):
        path = ast.unparse(call.args[0]) if call.args else None
        variable = self.resolved_literal(keyword_map(call).get("variable"))
        return path, variable

    def assigned_call_targets(self, function_name):
        matches = {}
        for index, tree in self.classical_trees.items():
            for node in ast.walk(tree):
                name = assignment_name(node)
                if name is None:
                    continue
                value = assignment_value(node)
                if isinstance(value, ast.Call) and call_name(value) == function_name:
                    matches[id(value)] = (index, name)
        return matches

    def one_cell_assignment(self, index, name):
        matches = [
            node
            for node in ast.walk(self.trees[index])
            if assignment_name(node) == name
        ]
        self.assertEqual(
            len(matches),
            1,
            f"Expected one assignment to {name} in cell {index}",
        )
        return matches[0]

    def assigned_call(self, target, api):
        matches = []
        for index, tree in self.classical_trees.items():
            for node in ast.walk(tree):
                if assignment_name(node) != target:
                    continue
                value = assignment_value(node)
                if isinstance(value, ast.Call) and call_name(value) == api:
                    matches.append((index, node, value))
        self.assertEqual(
            len(matches),
            1,
            f"Expected one {api} call assigned to {target}",
        )
        return matches[0]

    def assert_local_plot_configuration(self, index, call, specification):
        quantity = self.one_cell_assignment(index, "QUANTITY")
        variable = self.one_cell_assignment(index, "VARIABLE")
        axis = self.one_cell_assignment(index, specification["axis_name"])
        coordinate = self.one_cell_assignment(
            index,
            specification["coordinate_name"],
        )

        for name, node in (
            ("QUANTITY", quantity),
            ("VARIABLE", variable),
            (specification["axis_name"], axis),
            (specification["coordinate_name"], coordinate),
        ):
            with self.subTest(cell=index, local=name):
                self.assertLess(
                    node.lineno,
                    call.lineno,
                    f"{name} must be defined before the plotting call",
                )

        self.assertEqual(
            ast.literal_eval(assignment_value(quantity)),
            specification["quantity"],
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(variable)),
            specification["variable"],
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(axis)),
            specification["axis"],
        )

        coordinate_value = assignment_value(coordinate)
        if "coordinate" in specification:
            self.assertEqual(
                ast.unparse(coordinate_value),
                specification["coordinate"],
            )
        else:
            self.assertIsInstance(coordinate_value, ast.Dict)
            coordinate_items = {
                ast.literal_eval(key): ast.unparse(value)
                for key, value in zip(
                    coordinate_value.keys,
                    coordinate_value.values,
                )
            }
            self.assertEqual(
                coordinate_items,
                specification["coordinates"],
            )

        keywords = keyword_map(call)
        self.assertEqual(ast.unparse(call.args[0]), "RUN_DIRECTORY")
        self.assertEqual(ast.unparse(call.args[1]), "QUANTITY")
        self.assertEqual(ast.unparse(keywords["bias"]), "BIAS")
        self.assertEqual(ast.unparse(keywords["variable"]), "VARIABLE")
        if specification["plot_api"] == "plot_bias_volume_slice":
            self.assertEqual(
                ast.unparse(keywords["slice_axis"]),
                "SLICE_AXIS",
            )
            self.assertEqual(
                ast.unparse(keywords["slice_value"]),
                "SLICE_COORDINATE_NM",
            )
        else:
            self.assertEqual(ast.unparse(keywords["axis"]), "LINE_AXIS")
            self.assertEqual(
                ast.unparse(keywords["fixed_coords"]),
                "FIXED_COORDINATES_NM",
            )

    def test_public_classical_apis_are_directly_imported(self):
        imported_by_module = {}
        aliases_by_module = {}
        wildcard_imports = []
        for tree in self.trees.values():
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                imported_by_module.setdefault(node.module, set()).update(
                    alias.name for alias in node.names
                )
                aliases_by_module.setdefault(node.module, []).extend(node.names)
                wildcard_imports.extend(
                    alias for alias in node.names if alias.name == "*"
                )

        required_apis = {
            "find_vtr_plane_extrema",
            "load_vtr_linecut",
            "load_vtr_plane",
            "plot_bias_volume_3d",
            "plot_bias_volume_linecut",
            "plot_bias_volume_slice",
            "resolve_bias_output_file",
        }
        self.assertTrue(
            required_apis.issubset(
                imported_by_module.get("nextnanopp_tools", set())
            )
        )
        self.assertTrue(
            all(
                alias.asname is None
                for alias in aliases_by_module.get("nextnanopp_tools", [])
            )
        )
        self.assertEqual(wildcard_imports, [])
        self.assertNotIn("nextnano_analysis", self.combined_code)
        self.assertNotIn("importlib.reload", self.combined_code)
        self.assertFalse(self.calls_named("plot_vtr_linecut"))

        for name in required_apis:
            with self.subTest(name=name):
                self.assertTrue(self.calls_named(name, classical_only=True))

    def test_classical_configuration_is_explicit_and_centralized(self):
        expected_literals = {
            "XY_PLANE_Z_NM": -4.0,
            "XZ_PLANE_Y_NM": 0.0,
            "X_LINE_Y_NM": 0.0,
            "X_LINE_Z_NM": -4.0,
            "Z_LINE_X_NM": 1150.0,
            "Z_LINE_Y_NM": 0.0,
            "SHOW_OPTIONAL_CLASSICAL_3D": False,
            "SHOW_SECONDARY_BANDS": False,
            "SHOW_ELECTRON_DENSITY": False,
            "SECONDARY_BAND_VARIABLES": ("LH", "SO"),
        }
        config_cells = set()
        for name, expected in expected_literals.items():
            with self.subTest(name=name):
                index, node = self.one_top_assignment(name)
                config_cells.add(index)
                self.assertEqual(
                    ast.literal_eval(assignment_value(node)),
                    expected,
                )

        self.assertEqual(len(config_cells), 1)
        config_cell = config_cells.pop()
        self.assertTrue(
            self.classical_start < config_cell < self.quantum_start
        )

        for obsolete_name in (
            "QW_PLANE_Z_NM",
            "LINE_FIXED_COORDINATES",
            "DEFAULT_1D_LINE_AXIS",
            "DEFAULT_1D_LINE_FIXED_COORDS",
            "HOLE_DENSITY_VARIABLE",
            "POTENTIAL_VARIABLE",
            "HH_BAND_VARIABLE",
            "ELECTRON_DENSITY_VARIABLE",
        ):
            with self.subTest(obsolete_name=obsolete_name):
                self.assertNotIn(obsolete_name, self.top_assignments)
                self.assertIsNone(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(obsolete_name)}"
                        rf"(?![A-Za-z0-9_])",
                        self.classical_source,
                    )
                )

    def test_old_wrappers_and_classical_local_implementations_are_absent(self):
        definitions = {
            node.name
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        }
        called = {
            call_name(call)
            for _, call in self.calls
            if call_name(call) is not None
        }
        self.assertTrue(REMOVED_CLASSICAL_WRAPPERS.isdisjoint(definitions))
        self.assertTrue(REMOVED_CLASSICAL_WRAPPERS.isdisjoint(called))
        for name in REMOVED_CLASSICAL_WRAPPERS:
            with self.subTest(name=name):
                self.assertIsNone(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(name)}"
                        rf"(?![A-Za-z0-9_])",
                        self.combined_source,
                    )
                )

        for index, tree in self.classical_trees.items():
            self.assertTrue(
                tree.body,
                f"Classical code cell {index} is empty or comment-only",
            )
            for node in ast.walk(tree):
                self.assertNotIsInstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                    f"Notebook-local implementation remains in cell {index}",
                )

    def test_primary_planes_and_lines_use_cell_local_configuration(self):
        figure_cells = {}
        quantities = []
        for figure_name, specification in DEFAULT_CLASSICAL_FIGURES.items():
            index, _, call = self.assigned_call(
                figure_name,
                specification["plot_api"],
            )
            figure_cells[figure_name] = index
            quantities.append(specification["quantity"])
            with self.subTest(figure=figure_name, cell=index):
                self.assert_local_plot_configuration(
                    index,
                    call,
                    specification,
                )
                self.assertIs(
                    ast.literal_eval(keyword_map(call)["interactive"]),
                    False,
                )
                self.assertTrue(
                    any(
                        display_index == index
                        and display.args
                        and any(
                            isinstance(node, ast.Name)
                            and node.id == figure_name
                            for node in ast.walk(display.args[0])
                        )
                        for display_index, display in self.calls_named(
                            "display",
                            classical_only=True,
                        )
                    ),
                    f"{figure_name} is not displayed in its plotting cell",
                )

        self.assertEqual(len(set(figure_cells.values())), 9)
        self.assertEqual(quantities.count("density_hole"), 2)
        self.assertEqual(quantities.count("potential"), 3)
        self.assertEqual(quantities.count("bandedges"), 4)

    def test_primary_metadata_loaders_match_each_plotting_cell(self):
        for figure_name, specification in DEFAULT_CLASSICAL_FIGURES.items():
            plot_index, _, plot_call = self.assigned_call(
                figure_name,
                specification["plot_api"],
            )
            loader_index, loader_assignment, loader_call = self.assigned_call(
                specification["data_name"],
                specification["loader_api"],
            )
            with self.subTest(figure=figure_name, cell=plot_index):
                self.assertEqual(loader_index, plot_index)
                self.assertLess(loader_assignment.lineno, plot_call.lineno)
                self.assertEqual(
                    ast.unparse(loader_call.args[0]),
                    specification["path"],
                )
                loader_keywords = keyword_map(loader_call)
                self.assertEqual(
                    ast.unparse(loader_keywords["variable"]),
                    "VARIABLE",
                )
                if specification["loader_api"] == "load_vtr_plane":
                    self.assertEqual(
                        ast.unparse(loader_keywords["slice_axis"]),
                        "SLICE_AXIS",
                    )
                    self.assertEqual(
                        ast.unparse(loader_keywords["slice_value"]),
                        "SLICE_COORDINATE_NM",
                    )
                else:
                    self.assertEqual(
                        ast.unparse(loader_keywords["axis"]),
                        "LINE_AXIS",
                    )
                    self.assertEqual(
                        ast.unparse(loader_keywords["fixed_coords"]),
                        "FIXED_COORDINATES_NM",
                    )

    def test_primary_titles_reference_actual_selected_grid_coordinates(self):
        for figure_name, specification in DEFAULT_CLASSICAL_FIGURES.items():
            index, _, call = self.assigned_call(
                figure_name,
                specification["plot_api"],
            )
            if specification["plot_api"] == "plot_bias_volume_slice":
                with self.subTest(figure=figure_name, kind="plane"):
                    title_source = ast.unparse(keyword_map(call)["title"])
                    self.assertIn(specification["data_name"], title_source)
                    self.assertIn("'slice_coordinate'", title_source)
                    self.assertIn(
                        f"{specification['axis']} = ",
                        title_source,
                    )
                    self.assertIn("nm", title_source)
                continue

            title_calls = [
                candidate
                for candidate_index, candidate in self.classical_calls
                if candidate_index == index
                and call_name(candidate) == "set_title"
                and isinstance(candidate.func, ast.Attribute)
                and ast.unparse(candidate.func.value).startswith(figure_name)
            ]
            with self.subTest(figure=figure_name, kind="line"):
                self.assertEqual(len(title_calls), 1)
                title_source = ast.unparse(title_calls[0].args[0])
                self.assertIn(specification["data_name"], title_source)
                self.assertIn("'chosen_coords'", title_source)
                self.assertIn("LINE_AXIS", title_source)
                for coordinate_axis in specification["coordinates"]:
                    self.assertIn(
                        f"['{coordinate_axis}']",
                        title_source,
                    )
                    self.assertIn(f"{coordinate_axis} = ", title_source)
                self.assertIn("nm", title_source)

    def test_all_classical_cuts_define_coordinates_in_their_own_cells(self):
        plane_apis = (
            "find_vtr_plane_extrema",
            "load_vtr_plane",
            "plot_bias_volume_slice",
        )
        for api in plane_apis:
            for index, call in self.calls_named(api, classical_only=True):
                axis_assignment = self.one_cell_assignment(
                    index,
                    "SLICE_AXIS",
                )
                coordinate_assignment = self.one_cell_assignment(
                    index,
                    "SLICE_COORDINATE_NM",
                )
                keywords = keyword_map(call)
                with self.subTest(api=api, kind="plane", cell=index):
                    self.assertLess(axis_assignment.lineno, call.lineno)
                    self.assertLess(coordinate_assignment.lineno, call.lineno)
                    self.assertEqual(
                        ast.unparse(keywords["slice_axis"]),
                        "SLICE_AXIS",
                    )
                    self.assertEqual(
                        ast.unparse(keywords["slice_value"]),
                        "SLICE_COORDINATE_NM",
                    )
                    axis = ast.literal_eval(
                        assignment_value(axis_assignment)
                    )
                    coordinate = ast.unparse(
                        assignment_value(coordinate_assignment)
                    )
                    self.assertEqual(
                        (axis, coordinate),
                        {
                            "z": ("z", "XY_PLANE_Z_NM"),
                            "y": ("y", "XZ_PLANE_Y_NM"),
                        }[axis],
                    )

        for api in ("load_vtr_linecut", "plot_bias_volume_linecut"):
            for index, call in self.calls_named(api, classical_only=True):
                axis_assignment = self.one_cell_assignment(index, "LINE_AXIS")
                coordinates_assignment = self.one_cell_assignment(
                    index,
                    "FIXED_COORDINATES_NM",
                )
                keywords = keyword_map(call)
                with self.subTest(api=api, kind="line", cell=index):
                    self.assertLess(axis_assignment.lineno, call.lineno)
                    self.assertLess(coordinates_assignment.lineno, call.lineno)
                    self.assertEqual(
                        ast.unparse(keywords["axis"]),
                        "LINE_AXIS",
                    )
                    self.assertEqual(
                        ast.unparse(keywords["fixed_coords"]),
                        "FIXED_COORDINATES_NM",
                    )
                    axis = ast.literal_eval(
                        assignment_value(axis_assignment)
                    )
                    coordinate_value = assignment_value(
                        coordinates_assignment
                    )
                    self.assertIsInstance(coordinate_value, ast.Dict)
                    coordinates = {
                        ast.literal_eval(key): ast.unparse(value)
                        for key, value in zip(
                            coordinate_value.keys,
                            coordinate_value.values,
                        )
                    }
                    self.assertEqual(
                        coordinates,
                        {
                            "x": {
                                "y": "X_LINE_Y_NM",
                                "z": "X_LINE_Z_NM",
                            },
                            "z": {
                                "x": "Z_LINE_X_NM",
                                "y": "Z_LINE_Y_NM",
                            },
                        }[axis],
                    )

        self.assertNotIn("-7.5", self.classical_source)
        self.assertIsNone(
            re.search(
                r"y\s*=\s*100(?:\.0)?\s*nm",
                self.classical_source,
                flags=re.IGNORECASE,
            )
        )

    def test_secondary_bands_electron_density_and_3d_are_guarded(self):
        secondary_guarded = self.guarded_node_ids("SHOW_SECONDARY_BANDS")
        electron_guarded = self.guarded_node_ids("SHOW_ELECTRON_DENSITY")
        volume_guarded = self.guarded_node_ids("SHOW_OPTIONAL_CLASSICAL_3D")
        self.assertTrue(secondary_guarded)
        self.assertTrue(electron_guarded)
        self.assertTrue(volume_guarded)

        _, secondary_variables = self.one_top_assignment(
            "SECONDARY_BAND_VARIABLES"
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(secondary_variables)),
            ("LH", "SO"),
        )
        secondary_loops = [
            node
            for tree in self.classical_trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and ast.unparse(node.iter) == "SECONDARY_BAND_VARIABLES"
        ]
        self.assertEqual(len(secondary_loops), 1)
        self.assertIn(id(secondary_loops[0]), secondary_guarded)

        secondary_calls = [
            (index, call)
            for index, call in self.classical_calls
            if id(call) in secondary_guarded
            and call_name(call)
            in {
                "load_vtr_plane",
                "load_vtr_linecut",
                "plot_bias_volume_slice",
                "plot_bias_volume_linecut",
            }
        ]
        self.assertEqual(
            {call_name(call) for _, call in secondary_calls},
            {
                "load_vtr_plane",
                "load_vtr_linecut",
                "plot_bias_volume_slice",
                "plot_bias_volume_linecut",
            },
        )
        self.assertEqual(
            len({index for index, _ in secondary_calls}),
            1,
        )
        secondary_cell = secondary_calls[0][0]
        secondary_quantity = self.one_cell_assignment(
            secondary_cell,
            "QUANTITY",
        )
        secondary_variable = self.one_cell_assignment(
            secondary_cell,
            "VARIABLE",
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(secondary_quantity)),
            "bandedges",
        )
        self.assertEqual(
            ast.unparse(assignment_value(secondary_variable)),
            "band_variable",
        )
        for _, call in secondary_calls:
            keywords = keyword_map(call)
            self.assertEqual(ast.unparse(keywords["variable"]), "VARIABLE")
            if call_name(call).startswith("plot_"):
                self.assertEqual(ast.unparse(call.args[1]), "QUANTITY")

        electron_calls = [
            (index, call)
            for index, call in self.classical_calls
            if id(call) in electron_guarded
            and call_name(call)
            in {
                "resolve_bias_output_file",
                "load_vtr_plane",
                "load_vtr_linecut",
                "plot_bias_volume_slice",
                "plot_bias_volume_linecut",
                "plot_bias_volume_3d",
            }
        ]
        self.assertTrue(electron_calls)
        self.assertTrue(
            {
                "resolve_bias_output_file",
                "load_vtr_plane",
                "load_vtr_linecut",
                "plot_bias_volume_slice",
                "plot_bias_volume_linecut",
            }.issubset({call_name(call) for _, call in electron_calls})
        )
        self.assertEqual(len({index for index, _ in electron_calls}), 1)
        electron_cell = electron_calls[0][0]
        electron_quantity = self.one_cell_assignment(
            electron_cell,
            "QUANTITY",
        )
        electron_variable = self.one_cell_assignment(
            electron_cell,
            "VARIABLE",
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(electron_quantity)),
            "density_electron",
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(electron_variable)),
            "Electron_density",
        )
        for _, call in electron_calls:
            keywords = keyword_map(call)
            if "variable" in keywords:
                self.assertEqual(ast.unparse(keywords["variable"]), "VARIABLE")
            if call_name(call) in {
                "resolve_bias_output_file",
                "plot_bias_volume_slice",
                "plot_bias_volume_linecut",
                "plot_bias_volume_3d",
            }:
                self.assertEqual(ast.unparse(call.args[1]), "QUANTITY")

        volume_calls = self.calls_named(
            "plot_bias_volume_3d",
            classical_only=True,
        )
        self.assertTrue(volume_calls)
        self.assertTrue(
            all(id(call) in volume_guarded for _, call in volume_calls)
        )

    def test_classical_variable_selection_is_explicit(self):
        variable_consumers = {
            "find_vtr_plane_extrema",
            "load_vtr_linecut",
            "load_vtr_plane",
            "plot_bias_volume_3d",
            "plot_bias_volume_linecut",
            "plot_bias_volume_slice",
        }
        for index, call in self.classical_calls:
            if call_name(call) not in variable_consumers:
                continue
            with self.subTest(api=call_name(call), cell=index):
                variable = keyword_map(call).get("variable")
                self.assertIsNotNone(variable)
                self.assertIsNotNone(
                    literal_or_none(variable)
                    if isinstance(variable, ast.Constant)
                    else ast.unparse(variable)
                )

        for index, tree in self.classical_trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Subscript):
                    continue
                self.assertFalse(
                    isinstance(node.value, ast.Call)
                    and call_name(node.value) == "list_variables"
                    and literal_or_none(node.slice) == 0,
                    f"First available variable selected in cell {index}",
                )

    def test_hh_plane_minimum_summary_uses_public_extrema_api(self):
        extrema_calls = self.calls_named(
            "find_vtr_plane_extrema",
            classical_only=True,
        )
        self.assertEqual(len(extrema_calls), 1)
        extrema_index, extrema_call = extrema_calls[0]
        keywords = keyword_map(extrema_call)
        self.assertEqual(ast.unparse(extrema_call.args[0]), "BANDEDGES_VTR")
        self.assertEqual(ast.unparse(keywords["variable"]), "VARIABLE")
        self.assertEqual(ast.unparse(keywords["slice_axis"]), "SLICE_AXIS")
        self.assertEqual(
            ast.unparse(keywords["slice_value"]),
            "SLICE_COORDINATE_NM",
        )
        self.assertEqual(ast.literal_eval(keywords["kind"]), "min")
        self.assertEqual(ast.literal_eval(keywords["n_extrema"]), 1)
        self.assertEqual(
            ast.literal_eval(
                assignment_value(
                    self.one_cell_assignment(extrema_index, "QUANTITY")
                )
            ),
            "bandedges",
        )
        self.assertEqual(
            ast.literal_eval(
                assignment_value(
                    self.one_cell_assignment(extrema_index, "VARIABLE")
                )
            ),
            "HH",
        )
        self.assertEqual(
            ast.literal_eval(
                assignment_value(
                    self.one_cell_assignment(extrema_index, "SLICE_AXIS")
                )
            ),
            "z",
        )
        self.assertEqual(
            ast.unparse(
                assignment_value(
                    self.one_cell_assignment(
                        extrema_index,
                        "SLICE_COORDINATE_NM",
                    )
                )
            ),
            "XY_PLANE_Z_NM",
        )

        extrema_targets = self.assigned_call_targets("find_vtr_plane_extrema")
        self.assertIn(id(extrema_call), extrema_targets)
        _, result_name = extrema_targets[id(extrema_call)]
        self.assertTrue(
            any(
                call.args
                and any(
                    isinstance(node, ast.Name) and node.id == result_name
                    for node in ast.walk(call.args[0])
                )
                for _, call in self.calls_named("display", classical_only=True)
            ),
            f"{result_name} is not displayed as a concise summary",
        )

    def test_obsolete_conduction_examples_and_inactive_cells_are_absent(self):
        for fragment in (
            "DEFAULT_1D_LINE_DIAGNOSTIC",
            "Gamma",
            "Delta_1",
            "L_1",
            "uncomment",
        ):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.classical_source)

        for index in self.classical_code_indices:
            self.assertTrue(
                self.code_sources[index].strip(),
                f"Empty classical code cell {index}",
            )
            self.assertTrue(
                self.classical_trees[index].body,
                f"Inactive classical code cell {index}",
            )

    def test_required_outputs_and_surrounding_workflows_are_preserved(self):
        _, analyse_quantum = self.one_top_assignment(
            "ANALYSE_QUANTUM_OUTPUTS"
        )
        self.assertIs(
            ast.literal_eval(assignment_value(analyse_quantum)),
            True,
        )
        _, base_required_outputs = self.one_top_assignment(
            "BASE_REQUIRED_OUTPUTS"
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(base_required_outputs)),
            EXPECTED_BASE_REQUIRED_OUTPUTS,
        )
        _, quantum_required_outputs = self.one_top_assignment(
            "QUANTUM_REQUIRED_OUTPUTS"
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(quantum_required_outputs)),
            EXPECTED_QUANTUM_REQUIRED_OUTPUTS,
        )
        _, required_outputs = self.one_top_assignment("REQUIRED_OUTPUTS")
        self.assertEqual(
            ast.unparse(assignment_value(required_outputs)),
            (
                "BASE_REQUIRED_OUTPUTS + "
                "(QUANTUM_REQUIRED_OUTPUTS "
                "if ANALYSE_QUANTUM_OUTPUTS else ())"
            ),
        )
        self.assertNotIn(
            "density_electron.vtr",
            EXPECTED_BASE_REQUIRED_OUTPUTS
            + EXPECTED_QUANTUM_REQUIRED_OUTPUTS,
        )

        _, run_simulation = self.one_top_assignment("RUN_SIMULATION")
        self.assertIs(ast.literal_eval(assignment_value(run_simulation)), False)
        for name in ("RUN_DIRECTORY", "BIAS", "BIAS_DIRECTORY"):
            with self.subTest(name=name):
                self.assertIn(name, self.combined_code)
        self.assertEqual(len(self.calls_named("run_input_file")), 1)
        self.assertEqual(len(self.calls_named("validate_run_directory")), 1)

        preserved_structure_diagnostics = {
            "convergence_summary",
            "integrated_density_region_columns",
            "plot_convergence",
            "plot_integrated_density_hole",
            "plot_structure_plane",
            "plot_total_charges",
            "read_index_table",
            "read_integrated_density_hole",
            "read_total_charges",
            "resolve_structure_file",
        }
        preserved_quantum = {
            "plot_quantum_density_volume_3d",
            "plot_quantum_density_volume_linecut",
            "plot_quantum_density_volume_slice",
            "plot_quantum_probability_volume_3d",
            "plot_quantum_probability_volume_linecut",
            "plot_quantum_probability_volume_slice",
            "resolve_quantum_output_file",
            "resolve_quantum_probability_state_file",
        }
        for name in preserved_structure_diagnostics | preserved_quantum:
            with self.subTest(name=name):
                self.assertTrue(self.calls_named(name))

    def test_classical_section_markdown_and_notebook_state_are_clean(self):
        self.assertLess(self.classical_start, self.quantum_start)
        for phrase in (
            "classical hole density",
            "electrostatic potential",
            "valence-band edge",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.classical_source.lower())

        for index, cell in self.code_cells:
            with self.subTest(cell=index):
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                self.assertTrue(cell_source(cell).strip())
        self.assertFalse(
            self.cells[-1]["cell_type"] == "code"
            and not cell_source(self.cells[-1]).strip()
        )
        self.assertNotIn("/Users/robertjovanov/", self.combined_source)
        self.assertIsNone(re.search(r"[A-Za-z]:\\Users\\", self.combined_source))


if __name__ == "__main__":
    unittest.main()
