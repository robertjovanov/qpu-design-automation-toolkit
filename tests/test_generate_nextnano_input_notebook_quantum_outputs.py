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

REQUIRED_QUANTUM_APIS = {
    "find_probability_peaks",
    "list_variables",
    "plot_quantum_density_volume_3d",
    "plot_quantum_density_volume_linecut",
    "plot_quantum_density_volume_slice",
    "plot_quantum_energy_spectrum",
    "plot_quantum_occupation",
    "plot_quantum_probability_volume_3d",
    "plot_quantum_probability_volume_linecut",
    "plot_quantum_probability_volume_slice",
    "read_quantum_energy_spectrum",
    "read_quantum_occupation",
    "resolve_quantum_output_file",
    "resolve_quantum_probability_state_file",
}

PRESERVED_STRUCTURE_DIAGNOSTIC_APIS = {
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

PRESERVED_CLASSICAL_APIS = {
    "find_vtr_plane_extrema",
    "load_vtr_linecut",
    "load_vtr_plane",
    "plot_bias_volume_3d",
    "plot_bias_volume_linecut",
    "plot_bias_volume_slice",
    "resolve_bias_output_file",
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


def h2_title(source):
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.fullmatch(r"##(?!#)\s+(.+?)\s*", stripped)
        return None if match is None else match.group(1)
    return None


def contains_name(node, name):
    return any(
        isinstance(descendant, ast.Name) and descendant.id == name
        for descendant in ast.walk(node)
    )


class GenerateNextnanoInputNotebookQuantumOutputsTests(unittest.TestCase):
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

        cls.calls = [
            (index, node)
            for index, tree in cls.trees.items()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        cls.top_assignments = {}
        for index, tree in cls.trees.items():
            for node in tree.body:
                name = assignment_name(node)
                if name is not None:
                    cls.top_assignments.setdefault(name, []).append((index, node))

        cls.combined_code = "\n".join(cls.code_sources.values())
        cls.combined_source = "\n".join(cell_source(cell) for cell in cls.cells)
        cls.markdown_sources = {
            index: cell_source(cell)
            for index, cell in enumerate(cls.cells)
            if cell["cell_type"] == "markdown"
        }

        quantum_headings = [
            index
            for index, source in cls.markdown_sources.items()
            if (title := h2_title(source)) is not None
            and title.casefold() == "quantum outputs"
        ]
        if len(quantum_headings) != 1:
            raise AssertionError(
                "Expected exactly one 'Quantum outputs' H2 heading"
            )
        cls.quantum_start = quantum_headings[0]
        cls.quantum_end = next(
            (
                index
                for index, source in cls.markdown_sources.items()
                if index > cls.quantum_start and h2_title(source) is not None
            ),
            len(cls.cells),
        )

        cls.quantum_code_indices = [
            index
            for index, _ in cls.code_cells
            if cls.quantum_start < index < cls.quantum_end
        ]
        cls.quantum_trees = {
            index: cls.trees[index] for index in cls.quantum_code_indices
        }
        cls.quantum_calls = [
            (index, node)
            for index, tree in cls.quantum_trees.items()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        cls.quantum_code = "\n".join(
            cls.code_sources[index] for index in cls.quantum_code_indices
        )
        cls.quantum_source = "\n".join(
            cell_source(cell)
            for index, cell in enumerate(cls.cells)
            if cls.quantum_start <= index < cls.quantum_end
        )

    def one_top_assignment(self, name):
        matches = self.top_assignments.get(name, [])
        self.assertEqual(len(matches), 1, f"Expected one assignment to {name}")
        return matches[0]

    def calls_named(self, name, *, quantum_only=False):
        calls = self.quantum_calls if quantum_only else self.calls
        return [
            (index, call)
            for index, call in calls
            if call_name(call) == name
        ]

    def assigned_call(self, variable, api):
        index, assignment = self.one_quantum_assignment(variable)
        self.assertIn(index, self.quantum_code_indices)
        value = assignment_value(assignment)
        self.assertIsInstance(value, ast.Call)
        self.assertEqual(call_name(value), api)
        return index, value

    def one_quantum_assignment(self, name):
        matches = [
            (index, node)
            for index, tree in self.quantum_trees.items()
            for node in ast.walk(tree)
            if assignment_name(node) == name
        ]
        self.assertEqual(
            len(matches),
            1,
            f"Expected one quantum-section assignment to {name}",
        )
        return matches[0]

    def one_cell_assignment(self, index, name):
        matches = [
            node
            for node in ast.walk(self.quantum_trees[index])
            if assignment_name(node) == name
        ]
        self.assertEqual(
            len(matches),
            1,
            f"Expected one {name} assignment in quantum cell {index}",
        )
        return assignment_value(matches[0])

    def one_assigned_quantum_call(self, api):
        matches = []
        for index, tree in self.quantum_trees.items():
            for node in ast.walk(tree):
                name = assignment_name(node)
                if name is None:
                    continue
                value = assignment_value(node)
                if isinstance(value, ast.Call) and call_name(value) == api:
                    matches.append((index, name, value))
        self.assertEqual(
            len(matches),
            1,
            f"Expected one assigned quantum-section call to {api}",
        )
        return matches[0]

    def displayed_names(self):
        return {
            node.id
            for _, call in self.calls_named("display", quantum_only=True)
            if call.args
            for node in ast.walk(call.args[0])
            if isinstance(node, ast.Name)
        }

    def guarded_call_ids(self, condition):
        guards = []
        call_ids = set()
        else_call_ids = set()
        for index, tree in self.quantum_trees.items():
            for node in ast.walk(tree):
                if (
                    not isinstance(node, ast.If)
                    or ast.unparse(node.test) != condition
                ):
                    continue
                guards.append((index, node))
                for statement in node.body:
                    call_ids.update(
                        id(call)
                        for call in ast.walk(statement)
                        if isinstance(call, ast.Call)
                    )
                for statement in node.orelse:
                    else_call_ids.update(
                        id(call)
                        for call in ast.walk(statement)
                        if isinstance(call, ast.Call)
                    )
        return guards, call_ids, else_call_ids

    def assert_run_configuration(self, call, *, probability=False):
        self.assertTrue(call.args)
        self.assertEqual(ast.unparse(call.args[0]), "RUN_DIRECTORY")
        keywords = keyword_map(call)
        expected_names = {
            "region": "QUANTUM_REGION",
            "band": "QUANTUM_BAND",
            "bias": "BIAS",
        }
        if probability:
            expected_names.update(
                {
                    "state": "QUANTUM_STATE",
                    "kpoint": "QUANTUM_KPOINT",
                }
            )
        for keyword, expected_name in expected_names.items():
            with self.subTest(api=call_name(call), keyword=keyword):
                self.assertIn(keyword, keywords)
                self.assertEqual(ast.unparse(keywords[keyword]), expected_name)
        if probability:
            self.assertIn("shifted", keywords)
            self.assertIs(ast.literal_eval(keywords["shifted"]), True)

    def test_public_quantum_apis_are_directly_imported_and_called(self):
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

        self.assertTrue(
            REQUIRED_QUANTUM_APIS.issubset(
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
        self.assertNotIn("import nextnanopp_tools as", self.combined_code)

        for name in REQUIRED_QUANTUM_APIS:
            with self.subTest(name=name):
                self.assertTrue(
                    self.calls_named(name, quantum_only=True),
                    f"Expected a quantum-section call to {name}",
                )

    def test_quantum_configuration_is_explicit_centralized_and_exact(self):
        _, analyse_quantum = self.one_top_assignment(
            "ANALYSE_QUANTUM_OUTPUTS"
        )
        self.assertIs(
            ast.literal_eval(assignment_value(analyse_quantum)),
            True,
        )

        expected_literals = {
            "QUANTUM_REGION": "c-Ge_QW",
            "QUANTUM_BAND": "HH",
            "QUANTUM_KPOINT": "k00000",
            "QUANTUM_STATE": 1,
            "SHOW_OPTIONAL_QUANTUM_3D": False,
            "EXPECTED_PROBABILITY_PEAKS": 2,
            "PROBABILITY_PEAK_MIN_SEPARATION_NM": 50.0,
        }
        config_cells = set()
        for name, expected in expected_literals.items():
            with self.subTest(name=name):
                index, assignment = self.one_top_assignment(name)
                config_cells.add(index)
                self.assertIn(index, self.quantum_code_indices)
                self.assertEqual(
                    ast.literal_eval(assignment_value(assignment)),
                    expected,
                )
        self.assertEqual(len(config_cells), 1)

    def test_complete_quantum_analysis_is_guarded(self):
        guards, guarded_call_ids, else_call_ids = self.guarded_call_ids(
            "ANALYSE_QUANTUM_OUTPUTS"
        )
        self.assertEqual(
            {index for index, _ in guards},
            set(self.quantum_code_indices),
        )

        unguarded_configuration = {
            "QUANTUM_REGION",
            "QUANTUM_BAND",
            "QUANTUM_KPOINT",
            "QUANTUM_STATE",
            "SHOW_OPTIONAL_QUANTUM_3D",
            "EXPECTED_PROBABILITY_PEAKS",
            "PROBABILITY_PEAK_MIN_SEPARATION_NM",
        }
        for index, tree in self.quantum_trees.items():
            for statement in tree.body:
                name = assignment_name(statement)
                if name in unguarded_configuration:
                    continue
                with self.subTest(cell=index, statement=type(statement).__name__):
                    self.assertIsInstance(statement, ast.If)
                    self.assertEqual(
                        ast.unparse(statement.test),
                        "ANALYSE_QUANTUM_OUTPUTS",
                    )

        for api in REQUIRED_QUANTUM_APIS:
            for index, call in self.calls_named(api, quantum_only=True):
                with self.subTest(api=api, cell=index):
                    self.assertIn(id(call), guarded_call_ids)
                    self.assertNotIn(id(call), else_call_ids)

        display_calls = self.calls_named("display", quantum_only=True)
        disabled_display_calls = [
            call for _, call in display_calls if id(call) in else_call_ids
        ]
        self.assertEqual(len(disabled_display_calls), 1)
        self.assertEqual(
            ast.literal_eval(disabled_display_calls[0].args[0]),
            "Quantum analysis is disabled (ANALYSE_QUANTUM_OUTPUTS=False).",
        )
        for _, call in display_calls:
            self.assertIn(id(call), guarded_call_ids | else_call_ids)

    def test_required_quantum_files_are_resolved_exactly_and_summarized(self):
        _, density_resolver = self.assigned_call(
            "QUANTUM_DENSITY_VTR",
            "resolve_quantum_output_file",
        )
        self.assertEqual(len(density_resolver.args), 2)
        self.assertEqual(ast.unparse(density_resolver.args[0]), "RUN_DIRECTORY")
        self.assertEqual(ast.literal_eval(density_resolver.args[1]), "density")
        density_keywords = keyword_map(density_resolver)
        for keyword, expected in {
            "region": "QUANTUM_REGION",
            "band": "QUANTUM_BAND",
            "kpoint": "QUANTUM_KPOINT",
            "bias": "BIAS",
        }.items():
            self.assertEqual(ast.unparse(density_keywords[keyword]), expected)
        self.assertEqual(
            ast.literal_eval(density_keywords["preferred_extensions"]),
            ("vtr",),
        )

        _, probability_resolver = self.assigned_call(
            "QUANTUM_PROBABILITY_SHIFT_VTR",
            "resolve_quantum_probability_state_file",
        )
        self.assert_run_configuration(probability_resolver, probability=True)
        self.assertEqual(
            ast.literal_eval(
                keyword_map(probability_resolver)["preferred_extensions"]
            ),
            ("vtr",),
        )

        self.assertEqual(
            len(self.calls_named("resolve_quantum_output_file", quantum_only=True)),
            1,
        )
        self.assertEqual(
            len(
                self.calls_named(
                    "resolve_quantum_probability_state_file",
                    quantum_only=True,
                )
            ),
            1,
        )
        variable_calls = self.calls_named("list_variables", quantum_only=True)
        self.assertEqual(len(variable_calls), 2)
        self.assertEqual(
            {ast.unparse(call.args[0]) for _, call in variable_calls},
            {"QUANTUM_DENSITY_VTR", "QUANTUM_PROBABILITY_SHIFT_VTR"},
        )
        variable_call_cells = {index for index, _ in variable_calls}
        display_cells = {
            index
            for index, _ in self.calls_named("display", quantum_only=True)
        }
        self.assertTrue(variable_call_cells & display_cells)

        self.assertNotIn("Quantum/", self.quantum_code)
        for banned_call in (
            "glob",
            "iterdir",
            "rglob",
            "walk",
        ):
            with self.subTest(banned_call=banned_call):
                self.assertFalse(
                    self.calls_named(banned_call, quantum_only=True)
                )

    def test_default_density_and_probability_plots_use_local_coordinates(self):
        default_plots = {
            "plot_quantum_density_volume_slice": {
                "probability": False,
                "quantity": "density",
                "variable": "Density",
            },
            "plot_quantum_density_volume_linecut": {
                "probability": False,
                "quantity": "density",
                "variable": "Density",
            },
            "plot_quantum_probability_volume_slice": {
                "probability": True,
                "quantity": "probability_shift",
                "variable": "probability",
            },
            "plot_quantum_probability_volume_linecut": {
                "probability": True,
                "quantity": "probability_shift",
                "variable": "probability",
            },
        }
        _, analysis_call_ids, _ = self.guarded_call_ids(
            "ANALYSE_QUANTUM_OUTPUTS"
        )
        _, optional_3d_call_ids, _ = self.guarded_call_ids(
            "SHOW_OPTIONAL_QUANTUM_3D"
        )

        plot_cells = {}
        for api, expected in default_plots.items():
            calls = self.calls_named(api, quantum_only=True)
            self.assertEqual(len(calls), 1)
            index, call = calls[0]
            plot_cells[api] = index
            self.assertIn(id(call), analysis_call_ids)
            self.assertNotIn(id(call), optional_3d_call_ids)
            self.assert_run_configuration(
                call,
                probability=expected["probability"],
            )

            quantity = self.one_cell_assignment(index, "QUANTITY")
            self.assertEqual(ast.literal_eval(quantity), expected["quantity"])
            variable = self.one_cell_assignment(index, "VARIABLE")
            if expected["variable"] == "Density":
                self.assertEqual(ast.literal_eval(variable), "Density")
            else:
                self.assertIsInstance(variable, ast.JoinedStr)
                self.assertEqual(
                    ast.unparse(variable),
                    "f'Psi^2_{QUANTUM_STATE}'",
                )

            keywords = keyword_map(call)
            self.assertEqual(ast.unparse(keywords["variable"]), "VARIABLE")
            assignments = {
                assignment_name(node): node
                for node in ast.walk(self.quantum_trees[index])
                if assignment_name(node) in {"QUANTITY", "VARIABLE"}
            }
            self.assertLess(assignments["QUANTITY"].lineno, call.lineno)
            self.assertLess(assignments["VARIABLE"].lineno, call.lineno)

        self.assertEqual(len(set(plot_cells.values())), len(default_plots))

        for api in (
            "plot_quantum_density_volume_slice",
            "plot_quantum_probability_volume_slice",
        ):
            index, call = self.calls_named(api, quantum_only=True)[0]
            self.assertEqual(
                ast.literal_eval(
                    self.one_cell_assignment(index, "SLICE_AXIS")
                ),
                "z",
            )
            self.assertEqual(
                ast.unparse(
                    self.one_cell_assignment(
                        index,
                        "SLICE_COORDINATE_NM",
                    )
                ),
                "XY_PLANE_Z_NM",
            )

            keywords = keyword_map(call)
            self.assertEqual(ast.unparse(keywords["slice_axis"]), "SLICE_AXIS")
            self.assertEqual(
                ast.unparse(keywords["slice_value"]),
                "SLICE_COORDINATE_NM",
            )

        for api in (
            "plot_quantum_density_volume_linecut",
            "plot_quantum_probability_volume_linecut",
        ):
            index, call = self.calls_named(api, quantum_only=True)[0]
            self.assertEqual(
                ast.literal_eval(
                    self.one_cell_assignment(index, "LINE_AXIS")
                ),
                "x",
            )
            fixed_coordinates = self.one_cell_assignment(
                index,
                "FIXED_COORDINATES_NM",
            )
            self.assertIsInstance(fixed_coordinates, ast.Dict)
            fixed_items = {
                ast.literal_eval(key): ast.unparse(value)
                for key, value in zip(
                    fixed_coordinates.keys,
                    fixed_coordinates.values,
                )
            }
            self.assertEqual(
                fixed_items,
                {
                    "y": "X_LINE_Y_NM",
                    "z": "X_LINE_Z_NM",
                },
            )

            keywords = keyword_map(call)
            self.assertEqual(ast.unparse(keywords["axis"]), "LINE_AXIS")
            self.assertEqual(
                ast.unparse(keywords["fixed_coords"]),
                "FIXED_COORDINATES_NM",
            )

        displayed = self.displayed_names()
        for api in (
            "plot_quantum_density_volume_slice",
            "plot_quantum_density_volume_linecut",
            "plot_quantum_probability_volume_slice",
            "plot_quantum_probability_volume_linecut",
        ):
            _, figure_name, _ = self.one_assigned_quantum_call(api)
            with self.subTest(api=api, figure_name=figure_name):
                self.assertIn(figure_name, displayed)

    def test_quantum_3d_plots_are_only_inside_the_disabled_guard(self):
        matching_guards, guarded_call_ids, _ = self.guarded_call_ids(
            "SHOW_OPTIONAL_QUANTUM_3D"
        )
        _, analysis_call_ids, _ = self.guarded_call_ids(
            "ANALYSE_QUANTUM_OUTPUTS"
        )
        self.assertEqual(len(matching_guards), 1)

        for api in (
            "plot_quantum_density_volume_3d",
            "plot_quantum_probability_volume_3d",
        ):
            calls = self.calls_named(api, quantum_only=True)
            self.assertEqual(len(calls), 1)
            _, call = calls[0]
            self.assertIn(id(call), guarded_call_ids)
            self.assertIn(id(call), analysis_call_ids)
            self.assert_run_configuration(
                call,
                probability=api == "plot_quantum_probability_volume_3d",
            )

    def test_probability_peaks_are_explicit_and_conservatively_labelled(self):
        peaks_index, peaks_call = self.assigned_call(
            "QUANTUM_PROBABILITY_PEAKS",
            "find_probability_peaks",
        )
        self.assertEqual(
            ast.unparse(peaks_call.args[0]),
            "QUANTUM_PROBABILITY_SHIFT_VTR",
        )
        peak_keywords = keyword_map(peaks_call)
        self.assertEqual(
            ast.unparse(peak_keywords["variable"]),
            "VARIABLE",
        )
        self.assertEqual(
            ast.literal_eval(
                self.one_cell_assignment(peaks_index, "QUANTITY")
            ),
            "probability_shift",
        )
        probability_variable = self.one_cell_assignment(
            peaks_index,
            "VARIABLE",
        )
        self.assertIsInstance(probability_variable, ast.JoinedStr)
        self.assertIn("Psi^2_", ast.unparse(probability_variable))
        self.assertIn("QUANTUM_STATE", ast.unparse(probability_variable))
        self.assertEqual(
            ast.unparse(peak_keywords["n_peaks"]),
            "EXPECTED_PROBABILITY_PEAKS",
        )
        self.assertEqual(
            ast.unparse(peak_keywords["min_lateral_separation_nm"]),
            "PROBABILITY_PEAK_MIN_SEPARATION_NM",
        )
        self.assertIn(
            "QUANTUM_PROBABILITY_PEAKS_SUMMARY",
            self.displayed_names(),
        )

        normalized_markdown = (
            self.quantum_source.casefold()
            .replace("–", "-")
            .replace("—", "-")
        )
        self.assertRegex(
            normalized_markdown,
            r"probability(?:-|\s+)density peaks?",
        )

    def test_occupation_summary_and_plot_use_public_apis(self):
        _, occupation_name, read_call = self.one_assigned_quantum_call(
            "read_quantum_occupation"
        )
        self.assert_run_configuration(read_call)
        _, figure_name, plot_call = self.one_assigned_quantum_call(
            "plot_quantum_occupation"
        )
        self.assert_run_configuration(plot_call)

        summary_index, summary_assignment = self.one_quantum_assignment(
            "QUANTUM_OCCUPATION_SUMMARY"
        )
        self.assertIn(summary_index, self.quantum_code_indices)
        self.assertTrue(
            contains_name(
                assignment_value(summary_assignment),
                occupation_name,
            )
        )
        displayed = self.displayed_names()
        self.assertIn("QUANTUM_OCCUPATION_SUMMARY", displayed)
        self.assertIn(figure_name, displayed)
        self.assertNotIn(occupation_name, displayed)

    def test_energy_summary_and_plot_use_configured_kpoint(self):
        _, energy_name, read_call = self.one_assigned_quantum_call(
            "read_quantum_energy_spectrum"
        )
        self.assert_run_configuration(read_call)
        self.assertEqual(
            ast.unparse(keyword_map(read_call)["kpoint"]),
            "QUANTUM_KPOINT",
        )
        _, figure_name, plot_call = self.one_assigned_quantum_call(
            "plot_quantum_energy_spectrum"
        )
        self.assert_run_configuration(plot_call)
        self.assertEqual(
            ast.unparse(keyword_map(plot_call)["kpoint"]),
            "QUANTUM_KPOINT",
        )

        summary_index, summary_assignment = self.one_quantum_assignment(
            "QUANTUM_ENERGY_SUMMARY"
        )
        self.assertIn(summary_index, self.quantum_code_indices)
        self.assertTrue(
            contains_name(
                assignment_value(summary_assignment),
                energy_name,
            )
        )
        displayed = self.displayed_names()
        self.assertIn("QUANTUM_ENERGY_SUMMARY", displayed)
        self.assertIn(figure_name, displayed)
        self.assertNotIn(energy_name, displayed)

    def test_quantum_section_has_no_local_implementation_or_silent_fallback(self):
        for index, tree in self.quantum_trees.items():
            for node in ast.walk(tree):
                with self.subTest(cell=index, node=type(node).__name__):
                    self.assertNotIsInstance(
                        node,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                            ast.Try,
                        ),
                    )
            for node in tree.body:
                self.assertNotIsInstance(
                    node,
                    (ast.Import, ast.ImportFrom),
                    f"Quantum-section import remains in cell {index}",
                )

        for banned_call in (
            "open",
            "read_csv",
            "read_table",
            "loadtxt",
            "genfromtxt",
        ):
            with self.subTest(banned_call=banned_call):
                self.assertFalse(
                    self.calls_named(banned_call, quantum_only=True)
                )

        for obsolete_name in (
            "QUANTUM_QW_SLICE_Z_NM",
            "QUANTUM_LINE_AXIS",
            "QUANTUM_LINE_FIXED_COORDS",
            "QUANTUM_OUTPUTS_AVAILABLE",
            "QUANTUM_MISSING_MESSAGE",
        ):
            with self.subTest(obsolete_name=obsolete_name):
                self.assertIsNone(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(obsolete_name)}"
                        rf"(?![A-Za-z0-9_])",
                        self.quantum_source,
                    )
                )

        for index in self.quantum_code_indices:
            for line in self.code_sources[index].splitlines():
                if not line.lstrip().startswith("#"):
                    continue
                with self.subTest(cell=index, line=line):
                    self.assertIsNone(
                        re.search(
                            r"(?:plot|read|resolve)_quantum_",
                            line,
                        )
                    )

    def test_required_outputs_conditionally_include_quantum_files(self):
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

    def test_dual_mode_and_previous_analysis_sections_are_preserved(self):
        _, run_simulation = self.one_top_assignment("RUN_SIMULATION")
        self.assertIs(ast.literal_eval(assignment_value(run_simulation)), False)
        for name in (
            "RUN_DIRECTORY",
            "SIMULATION_OUTPUT_ROOT",
            "SIMULATION_STAGING_ROOT",
            "BIAS",
            "BIAS_DIRECTORY",
            "STRUCTURE_DIRECTORY",
        ):
            with self.subTest(name=name):
                self.assertIn(name, self.combined_code)
        self.assertEqual(len(self.calls_named("run_input_file")), 1)
        self.assertEqual(len(self.calls_named("validate_run_directory")), 1)

        for name in (
            PRESERVED_STRUCTURE_DIAGNOSTIC_APIS | PRESERVED_CLASSICAL_APIS
        ):
            with self.subTest(api=name):
                self.assertTrue(self.calls_named(name))

        for banned_selection in (
            "find_latest_run",
            "st_mtime",
            "getmtime",
            "matching_runs",
            "tagged_runs",
        ):
            with self.subTest(banned_selection=banned_selection):
                self.assertNotIn(banned_selection, self.combined_code)

    def test_quantum_plots_share_the_six_canonical_analysis_coordinates(self):
        expected_coordinates = {
            "XY_PLANE_Z_NM": -4.0,
            "XZ_PLANE_Y_NM": 0.0,
            "X_LINE_Y_NM": 0.0,
            "X_LINE_Z_NM": -4.0,
            "Z_LINE_X_NM": 1150.0,
            "Z_LINE_Y_NM": 0.0,
        }
        coordinate_cells = set()
        for name, expected in expected_coordinates.items():
            with self.subTest(name=name):
                index, assignment = self.one_top_assignment(name)
                coordinate_cells.add(index)
                self.assertEqual(
                    ast.literal_eval(assignment_value(assignment)),
                    expected,
                )
        self.assertEqual(len(coordinate_cells), 1)

        for obsolete_name in (
            "QW_PLANE_Z_NM",
            "LINE_FIXED_COORDINATES",
            "DEFAULT_1D_LINE_AXIS",
            "DEFAULT_1D_LINE_FIXED_COORDS",
        ):
            with self.subTest(obsolete_name=obsolete_name):
                self.assertNotIn(obsolete_name, self.top_assignments)
                self.assertIsNone(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(obsolete_name)}"
                        rf"(?![A-Za-z0-9_])",
                        self.combined_code,
                    )
                )

    def test_quantum_markdown_is_concise_and_ordered(self):
        quantum_markdown = [
            (index, source)
            for index, source in self.markdown_sources.items()
            if self.quantum_start <= index < self.quantum_end
        ]

        def matching_index(fragment):
            matches = [
                index
                for index, source in quantum_markdown
                if fragment in source.casefold()
            ]
            self.assertEqual(
                len(matches),
                1,
                f"Expected one quantum markdown cell containing {fragment!r}",
            )
            return matches[0]

        ordered = [
            self.quantum_start,
            matching_index(
                "### quantum-calculated hh density — xy plane"
            ),
            matching_index(
                "### shifted state probability density — xy plane"
            ),
            matching_index("### state occupation"),
            matching_index("### energy spectrum"),
        ]
        self.assertEqual(ordered, sorted(ordered))

        later_markdown = "\n".join(
            source
            for index, source in self.markdown_sources.items()
            if index >= self.quantum_end
        ).casefold()
        self.assertTrue(
            "conclusion" in later_markdown or "figure export" in later_markdown
        )

    def test_notebook_state_paths_and_trailing_cells_are_clean(self):
        self.assertLess(self.quantum_start, self.quantum_end)
        self.assertTrue(self.quantum_code_indices)
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
