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

PRIMARY_CLASSICAL_OUTPUTS = {
    ("density_hole", "Hole_density"),
    ("potential", "Potential"),
    ("bandedges", "HH"),
}

PRIMARY_METADATA_OUTPUTS = {
    ("DENSITY_HOLE_VTR", "Hole_density"),
    ("POTENTIAL_VTR", "Potential"),
    ("BANDEDGES_VTR", "HH"),
}

PRIMARY_METADATA_PATHS = {
    ("density_hole", "Hole_density"): "DENSITY_HOLE_VTR",
    ("potential", "Potential"): "POTENTIAL_VTR",
    ("bandedges", "HH"): "BANDEDGES_VTR",
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
            if "classical physical outputs" in source.lower()
        ]
        if len(classical_headings) != 1:
            raise AssertionError(
                "Expected one markdown heading containing "
                "'classical physical outputs'"
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
            "QW_PLANE_Z_NM": -7.5,
            "LINE_AXIS": "x",
            "SHOW_OPTIONAL_CLASSICAL_3D": False,
            "SHOW_SECONDARY_BANDS": False,
            "SHOW_ELECTRON_DENSITY": False,
            "HH_BAND_VARIABLE": "HH",
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

        fixed_index, fixed_node = self.one_top_assignment(
            "LINE_FIXED_COORDINATES"
        )
        config_cells.add(fixed_index)
        fixed_value = assignment_value(fixed_node)
        self.assertIsInstance(fixed_value, ast.Dict)
        fixed_items = {
            ast.literal_eval(key): value
            for key, value in zip(fixed_value.keys, fixed_value.values)
        }
        self.assertEqual(set(fixed_items), {"y", "z"})
        self.assertEqual(ast.literal_eval(fixed_items["y"]), 100.0)
        self.assertEqual(ast.unparse(fixed_items["z"]), "QW_PLANE_Z_NM")

        self.assertEqual(len(config_cells), 1)
        config_cell = config_cells.pop()
        self.assertTrue(
            self.classical_start < config_cell < self.quantum_start
        )

        _, default_axis = self.one_top_assignment("DEFAULT_1D_LINE_AXIS")
        self.assertEqual(
            ast.unparse(assignment_value(default_axis)),
            "LINE_AXIS",
        )
        _, default_coords = self.one_top_assignment(
            "DEFAULT_1D_LINE_FIXED_COORDS"
        )
        self.assertEqual(
            ast.unparse(assignment_value(default_coords)),
            "dict(LINE_FIXED_COORDINATES)",
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

    def test_primary_planes_and_lines_use_exact_variables_and_shared_coordinates(self):
        plane_calls = self.calls_named(
            "plot_bias_volume_slice",
            classical_only=True,
        )
        line_calls = self.calls_named(
            "plot_bias_volume_linecut",
            classical_only=True,
        )

        primary_plane_calls = {
            self.classical_output_pair(call): call
            for _, call in plane_calls
            if self.classical_output_pair(call) in PRIMARY_CLASSICAL_OUTPUTS
        }
        primary_line_calls = {
            self.classical_output_pair(call): call
            for _, call in line_calls
            if self.classical_output_pair(call) in PRIMARY_CLASSICAL_OUTPUTS
        }
        self.assertEqual(set(primary_plane_calls), PRIMARY_CLASSICAL_OUTPUTS)
        self.assertEqual(set(primary_line_calls), PRIMARY_CLASSICAL_OUTPUTS)

        for pair, call in primary_plane_calls.items():
            keywords = keyword_map(call)
            with self.subTest(kind="plane", output=pair):
                self.assertEqual(ast.unparse(call.args[0]), "RUN_DIRECTORY")
                self.assertEqual(ast.unparse(keywords["bias"]), "BIAS")
                self.assertEqual(ast.literal_eval(keywords["slice_axis"]), "z")
                self.assertEqual(
                    ast.unparse(keywords["slice_value"]),
                    "QW_PLANE_Z_NM",
                )

        for pair, call in primary_line_calls.items():
            keywords = keyword_map(call)
            with self.subTest(kind="line", output=pair):
                self.assertEqual(ast.unparse(call.args[0]), "RUN_DIRECTORY")
                self.assertEqual(ast.unparse(keywords["bias"]), "BIAS")
                self.assertEqual(ast.unparse(keywords["axis"]), "LINE_AXIS")
                self.assertEqual(
                    ast.unparse(keywords["fixed_coords"]),
                    "LINE_FIXED_COORDINATES",
                )

    def test_primary_metadata_loaders_use_exact_variables_and_shared_coordinates(self):
        plane_calls = self.calls_named("load_vtr_plane", classical_only=True)
        line_calls = self.calls_named("load_vtr_linecut", classical_only=True)
        primary_plane_calls = {
            self.metadata_output_pair(call): call
            for _, call in plane_calls
            if self.metadata_output_pair(call) in PRIMARY_METADATA_OUTPUTS
        }
        primary_line_calls = {
            self.metadata_output_pair(call): call
            for _, call in line_calls
            if self.metadata_output_pair(call) in PRIMARY_METADATA_OUTPUTS
        }
        self.assertEqual(set(primary_plane_calls), PRIMARY_METADATA_OUTPUTS)
        self.assertEqual(set(primary_line_calls), PRIMARY_METADATA_OUTPUTS)

        for pair, call in primary_plane_calls.items():
            keywords = keyword_map(call)
            with self.subTest(kind="plane metadata", output=pair):
                self.assertEqual(ast.literal_eval(keywords["slice_axis"]), "z")
                self.assertEqual(
                    ast.unparse(keywords["slice_value"]),
                    "QW_PLANE_Z_NM",
                )

        for pair, call in primary_line_calls.items():
            keywords = keyword_map(call)
            with self.subTest(kind="line metadata", output=pair):
                self.assertEqual(ast.unparse(keywords["axis"]), "LINE_AXIS")
                self.assertEqual(
                    ast.unparse(keywords["fixed_coords"]),
                    "LINE_FIXED_COORDINATES",
                )

    def test_primary_titles_reference_actual_selected_grid_coordinates(self):
        plane_metadata_targets = self.assigned_call_targets("load_vtr_plane")
        line_metadata_targets = self.assigned_call_targets("load_vtr_linecut")
        primary_plane_metadata = {
            self.metadata_output_pair(call): plane_metadata_targets[id(call)][1]
            for _, call in self.calls_named("load_vtr_plane", classical_only=True)
            if self.metadata_output_pair(call) in PRIMARY_METADATA_OUTPUTS
        }
        primary_line_metadata = {
            self.metadata_output_pair(call): line_metadata_targets[id(call)][1]
            for _, call in self.calls_named("load_vtr_linecut", classical_only=True)
            if self.metadata_output_pair(call) in PRIMARY_METADATA_OUTPUTS
        }

        primary_plane_calls = [
            call
            for _, call in self.calls_named(
                "plot_bias_volume_slice",
                classical_only=True,
            )
            if self.classical_output_pair(call) in PRIMARY_CLASSICAL_OUTPUTS
        ]
        line_targets = self.assigned_call_targets(
            "plot_bias_volume_linecut"
        )
        primary_line_calls = [
            call
            for _, call in self.calls_named(
                "plot_bias_volume_linecut",
                classical_only=True,
            )
            if self.classical_output_pair(call) in PRIMARY_CLASSICAL_OUTPUTS
        ]
        display_calls = self.calls_named("display", classical_only=True)

        for call in primary_plane_calls:
            output_pair = self.classical_output_pair(call)
            metadata_pair = (
                PRIMARY_METADATA_PATHS[output_pair],
                output_pair[1],
            )
            with self.subTest(kind="plane", output=output_pair):
                metadata_name = primary_plane_metadata[metadata_pair]
                title_source = ast.unparse(keyword_map(call)["title"])
                self.assertIn(metadata_name, title_source)
                self.assertIn("'slice_coordinate'", title_source)

        for call in primary_line_calls:
            output_pair = self.classical_output_pair(call)
            metadata_pair = (
                PRIMARY_METADATA_PATHS[output_pair],
                output_pair[1],
            )
            with self.subTest(kind="line", output=output_pair):
                self.assertIn(
                    id(call),
                    line_targets,
                    "Primary line-cut figure must be assigned before titling",
                )
                _, figure_name = line_targets[id(call)]
                title_calls = [
                    candidate
                    for _, candidate in self.classical_calls
                    if call_name(candidate) in {"set_title", "update_layout"}
                    and isinstance(candidate.func, ast.Attribute)
                    and ast.unparse(candidate.func.value).startswith(figure_name)
                ]
                self.assertEqual(len(title_calls), 1)
                title_call = title_calls[0]
                title_value = (
                    title_call.args[0]
                    if title_call.args
                    else keyword_map(title_call).get("title")
                )
                self.assertIsNotNone(title_value)
                title_source = ast.unparse(title_value)
                metadata_name = primary_line_metadata[metadata_pair]
                self.assertIn(metadata_name, title_source)
                self.assertIn("'chosen_coords'", title_source)
                self.assertIn("['y']", title_source)
                self.assertIn("['z']", title_source)
                self.assertTrue(
                    any(
                        display.args
                        and any(
                            isinstance(node, ast.Name)
                            and node.id == figure_name
                            for node in ast.walk(display.args[0])
                        )
                        for _, display in display_calls
                    ),
                    f"{figure_name} is not displayed after title formatting",
                )

    def test_all_classical_cuts_use_the_shared_coordinate_configuration(self):
        for api in ("load_vtr_plane", "plot_bias_volume_slice"):
            for index, call in self.calls_named(api, classical_only=True):
                keywords = keyword_map(call)
                with self.subTest(api=api, kind="plane", cell=index):
                    self.assertEqual(
                        ast.literal_eval(keywords["slice_axis"]),
                        "z",
                    )
                    self.assertEqual(
                        ast.unparse(keywords["slice_value"]),
                        "QW_PLANE_Z_NM",
                    )
                    self.assertNotIsInstance(
                        keywords["slice_value"],
                        ast.Constant,
                    )

        for api in ("load_vtr_linecut", "plot_bias_volume_linecut"):
            for index, call in self.calls_named(api, classical_only=True):
                keywords = keyword_map(call)
                with self.subTest(api=api, kind="line", cell=index):
                    self.assertEqual(
                        ast.unparse(keywords["axis"]),
                        "LINE_AXIS",
                    )
                    self.assertEqual(
                        ast.unparse(keywords["fixed_coords"]),
                        "LINE_FIXED_COORDINATES",
                    )

        for old_coordinate in (
            r"z\s*=\s*-4(?:\.0)?\s*nm",
            r"z\s*=\s*-2(?:\.0)?\s*nm",
        ):
            with self.subTest(old_coordinate=old_coordinate):
                self.assertIsNone(
                    re.search(
                        old_coordinate,
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
            call
            for _, call in self.classical_calls
            if call_name(call)
            in {"plot_bias_volume_slice", "plot_bias_volume_linecut"}
            and (
                literal_or_none(keyword_map(call).get("variable"))
                in {"LH", "SO"}
                or (
                    literal_or_none(call_argument(call, 1, "quantity"))
                    == "bandedges"
                    and id(call) in secondary_guarded
                )
            )
        ]
        self.assertTrue(
            {"plot_bias_volume_slice", "plot_bias_volume_linecut"}.issubset(
                {call_name(call) for call in secondary_calls}
            )
        )
        self.assertTrue(
            all(id(call) in secondary_guarded for call in secondary_calls)
        )
        secondary_metadata_calls = [
            call
            for _, call in self.classical_calls
            if call_name(call) in {"load_vtr_plane", "load_vtr_linecut"}
            and ast.unparse(call.args[0]) == "BANDEDGES_VTR"
            and id(call) in secondary_guarded
        ]
        self.assertEqual(
            {call_name(call) for call in secondary_metadata_calls},
            {"load_vtr_plane", "load_vtr_linecut"},
        )
        self.assertTrue(
            all(id(call) in secondary_guarded for call in secondary_metadata_calls)
        )

        electron_calls = [
            call
            for _, call in self.classical_calls
            if literal_or_none(call_argument(call, 1, "quantity"))
            == "density_electron"
        ]
        self.assertTrue(electron_calls)
        self.assertTrue(all(id(call) in electron_guarded for call in electron_calls))
        self.assertTrue(
            {
                "resolve_bias_output_file",
                "plot_bias_volume_slice",
                "plot_bias_volume_linecut",
            }.issubset({call_name(call) for call in electron_calls})
        )
        electron_plot_variables = {
            self.resolved_literal(keyword_map(call).get("variable"))
            for call in electron_calls
            if call_name(call).startswith("plot_")
        }
        self.assertEqual(electron_plot_variables, {"Electron_density"})
        electron_metadata_calls = [
            call
            for _, call in self.classical_calls
            if call_name(call) in {"load_vtr_plane", "load_vtr_linecut"}
            and ast.unparse(call.args[0]) == "DENSITY_ELECTRON_VTR"
        ]
        self.assertEqual(
            {call_name(call) for call in electron_metadata_calls},
            {"load_vtr_plane", "load_vtr_linecut"},
        )
        self.assertTrue(
            all(id(call) in electron_guarded for call in electron_metadata_calls)
        )

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
        _, extrema_call = extrema_calls[0]
        keywords = keyword_map(extrema_call)
        self.assertEqual(ast.unparse(extrema_call.args[0]), "BANDEDGES_VTR")
        self.assertEqual(ast.unparse(keywords["variable"]), "HH_BAND_VARIABLE")
        self.assertEqual(ast.literal_eval(keywords["slice_axis"]), "z")
        self.assertEqual(
            ast.unparse(keywords["slice_value"]),
            "QW_PLANE_Z_NM",
        )
        self.assertEqual(ast.literal_eval(keywords["kind"]), "min")
        self.assertEqual(ast.literal_eval(keywords["n_extrema"]), 1)

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
            "valence-band edges",
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
