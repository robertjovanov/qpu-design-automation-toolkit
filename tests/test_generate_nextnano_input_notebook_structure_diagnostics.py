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

REMOVED_STRUCTURE_HELPERS = {
    "read_index_lookup",
    "read_ascii_vtr_index",
    "read_avs_fld",
    "header_int",
    "summarize_index_grid",
    "centers_to_edges",
    "contiguous_runs",
    "index_z_intervals",
    "lookup_index_name",
    "add_box_mesh",
    "sample_contact_points",
    "plot_structure_3d",
    "infer_slice_axes",
    "plot_fld_index_slice",
}

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


class GenerateNextnanoInputNotebookStructureDiagnosticsTests(unittest.TestCase):
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

        def one_h2_index(title):
            expected_line = f"## {title}"
            matches = [
                index
                for index, cell in enumerate(cls.cells)
                if cell["cell_type"] == "markdown"
                and any(
                    line.strip() == expected_line
                    for line in cell_source(cell).splitlines()
                )
            ]
            if len(matches) != 1:
                raise AssertionError(
                    f"Expected exactly one {expected_line!r} heading; "
                    f"found {len(matches)}"
                )
            return matches[0]

        cls.structure_start = one_h2_index("Structure and diagnostics")
        cls.physical_outputs_start = one_h2_index("Classical outputs")
        if cls.physical_outputs_start <= cls.structure_start:
            raise AssertionError(
                "Classical outputs must follow Structure and diagnostics"
            )
        cls.structure_code_indices = [
            index
            for index, _ in cls.code_cells
            if cls.structure_start < index < cls.physical_outputs_start
        ]
        cls.structure_trees = {
            index: cls.trees[index] for index in cls.structure_code_indices
        }
        cls.structure_code = "\n".join(
            cls.code_sources[index] for index in cls.structure_code_indices
        )

    def one_top_assignment(self, name):
        matches = self.top_assignments.get(name, [])
        self.assertEqual(len(matches), 1, f"Expected one assignment to {name}")
        return matches[0]

    def calls_named(self, name, *, section_only=False):
        allowed_indices = (
            set(self.structure_code_indices) if section_only else None
        )
        return [
            (index, call)
            for index, call in self.calls
            if call_name(call) == name
            and (allowed_indices is None or index in allowed_indices)
        ]

    def assert_displayed(self, name):
        displayed_names = {
            node.id
            for _, call in self.calls_named("display", section_only=True)
            if call.args
            for node in ast.walk(call.args[0])
            if isinstance(node, ast.Name)
        }
        self.assertIn(
            name,
            displayed_names,
            f"{name} is not displayed in the structure/diagnostics section",
        )

    def test_replaced_structure_helpers_and_manual_parsers_are_absent(self):
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
        self.assertTrue(REMOVED_STRUCTURE_HELPERS.isdisjoint(definitions))
        self.assertTrue(REMOVED_STRUCTURE_HELPERS.isdisjoint(called))
        for helper in REMOVED_STRUCTURE_HELPERS:
            with self.subTest(helper=helper):
                self.assertIsNone(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(helper)}"
                        rf"(?![A-Za-z0-9_])",
                        self.combined_source,
                    )
                )

        banned_parser_fragments = (
            "VTR_DATA_ARRAY_RE",
            "COORDINATE_ARRAY_NAMES",
            "<DataArray",
            "variable_skip",
            "coord1_skip",
            "coord2_skip",
            "MATERIAL_COLORS",
            "CONTACT_COLOR_SEQUENCE",
            "Mesh3d",
            "Scatter3d",
        )
        for fragment in banned_parser_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.structure_code)

        banned_calls = {
            "read_text",
            "read_csv",
            "fromstring",
            "reshape",
            "imshow",
            "rglob",
        }
        for index, tree in self.structure_trees.items():
            for node in ast.walk(tree):
                self.assertNotIsInstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Lambda,
                    ),
                    f"Reusable implementation remains in code cell {index}",
                )
                if isinstance(node, ast.Call):
                    self.assertNotIn(
                        call_name(node),
                        banned_calls,
                        f"Manual parser/plotting call remains in code cell {index}",
                    )
                if isinstance(node, ast.Dict):
                    numeric_keys = [
                        key.value
                        for key in node.keys
                        if isinstance(key, ast.Constant)
                        and isinstance(key.value, (int, float))
                    ]
                    self.assertEqual(
                        numeric_keys,
                        [],
                        f"Hard-coded numeric index mapping in code cell {index}",
                    )

    def test_public_structure_and_diagnostic_apis_are_directly_imported(self):
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
            "resolve_structure_file",
            "read_index_table",
            "plot_structure_plane",
            "convergence_summary",
            "plot_convergence",
            "read_integrated_density_hole",
            "integrated_density_region_columns",
            "plot_integrated_density_hole",
            "read_total_charges",
            "plot_total_charges",
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

        for name in required_apis:
            with self.subTest(name=name):
                self.assertTrue(self.calls_named(name, section_only=True))

    def test_structure_files_and_index_name_mappings(self):
        _, quantities = self.one_top_assignment("STRUCTURE_FILE_QUANTITIES")
        self.assertEqual(
            ast.literal_eval(assignment_value(quantities)),
            ("materials", "contacts", "regions_all"),
        )

        resolver_calls = self.calls_named(
            "resolve_structure_file",
            section_only=True,
        )
        self.assertEqual(len(resolver_calls), 3)
        for index, call in resolver_calls:
            with self.subTest(cell=index, call=ast.unparse(call)):
                self.assertEqual(ast.unparse(call.args[0]), "RUN_DIRECTORY")
                self.assertEqual(
                    ast.literal_eval(keyword_map(call)["preferred_extensions"]),
                    ("vtr",)
                    if ast.unparse(call.args[1]) == "quantity"
                    else ("txt",),
                )

        index_calls = self.calls_named("read_index_table", section_only=True)
        self.assertEqual(len(index_calls), 2)
        self.assertEqual(
            {ast.unparse(call.args[0]) for _, call in index_calls},
            {"MATERIAL_INDEX_PATH", "CONTACT_INDEX_PATH"},
        )

        for displayed_name in (
            "STRUCTURE_FILE_SUMMARY",
            "MATERIAL_INDEX_TABLE",
            "CONTACT_INDEX_TABLE",
        ):
            with self.subTest(displayed_name=displayed_name):
                self.assert_displayed(displayed_name)

        self.assertNotIn("material_indices.txt", self.structure_code)
        self.assertNotIn("contact_indices.txt", self.structure_code)

    def test_representative_structure_planes_use_generated_sections(self):
        plane_calls = self.calls_named(
            "plot_structure_plane",
            section_only=True,
        )
        self.assertEqual(len(plane_calls), 2)
        quantities = []
        for index, call in plane_calls:
            keywords = keyword_map(call)
            with self.subTest(cell=index):
                self.assertEqual(ast.unparse(call.args[0]), "RUN_DIRECTORY")
                self.assertIs(
                    ast.literal_eval(keywords["interactive"]),
                    False,
                )
                self.assertNotIn("slice_value", keywords)
                self.assertEqual(
                    ast.unparse(keywords["quantity"]),
                    "QUANTITY",
                )
                quantity_assignments = [
                    node
                    for node in self.structure_trees[index].body
                    if assignment_name(node) == "QUANTITY"
                ]
                self.assertEqual(len(quantity_assignments), 1)
                quantities.append(
                    ast.literal_eval(
                        assignment_value(quantity_assignments[0])
                    )
                )
        self.assertEqual(
            set(quantities),
            {"regions_all_2d_xy_QD", "materials_2d_xz_QD"},
        )

        self.assertNotIn("STRUCTURE_LINE_CUT", self.combined_source)
        self.assertNotIn("1d_z_BG2", self.structure_code)
        self.assertFalse(self.calls_named("plot_structure_linecut"))
        self.assertIn(
            "does not define a compatible 1D structure section",
            self.combined_source,
        )

    def test_convergence_diagnostics_are_summary_only_and_fail_clearly(self):
        summary_calls = self.calls_named(
            "convergence_summary",
            section_only=True,
        )
        plot_calls = self.calls_named(
            "plot_convergence",
            section_only=True,
        )
        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(len(plot_calls), 1)
        self.assertEqual(
            ast.unparse(summary_calls[0][1].args[0]),
            "BIAS_DIRECTORY",
        )
        self.assertEqual(
            ast.unparse(plot_calls[0][1].args[0]),
            "BIAS_DIRECTORY",
        )
        self.assertIs(
            ast.literal_eval(keyword_map(plot_calls[0][1])["interactive"]),
            False,
        )
        self.assert_displayed("CONVERGENCE_SUMMARY")
        self.assert_displayed("CONVERGENCE_FIGURE")
        self.assertFalse(self.calls_named("read_convergence_table"))

        for index, tree in self.structure_trees.items():
            for node in ast.walk(tree):
                self.assertNotIsInstance(
                    node,
                    (ast.Try, ast.ExceptHandler),
                    f"Diagnostics fallback found in code cell {index}",
                )

    def test_integrated_density_diagnostics_use_shared_reader_and_plotter(self):
        expected_calls = {
            "read_integrated_density_hole": 1,
            "integrated_density_region_columns": 1,
            "plot_integrated_density_hole": 1,
        }
        for name, expected_count in expected_calls.items():
            with self.subTest(name=name):
                self.assertEqual(
                    len(self.calls_named(name, section_only=True)),
                    expected_count,
                )

        read_call = self.calls_named(
            "read_integrated_density_hole",
            section_only=True,
        )[0][1]
        plot_call = self.calls_named(
            "plot_integrated_density_hole",
            section_only=True,
        )[0][1]
        for call in (read_call, plot_call):
            self.assertEqual(ast.unparse(call.args[0]), "RUN_DIRECTORY")

        plot_keywords = keyword_map(plot_call)
        self.assertEqual(
            ast.unparse(plot_keywords["region_columns"]),
            "INTEGRATED_REGION_COLUMNS",
        )
        self.assertIs(ast.literal_eval(plot_keywords["interactive"]), False)
        self.assertIs(
            ast.literal_eval(plot_keywords["label_with_materials"]),
            False,
        )
        for displayed_name in (
            "INTEGRATED_HOLE_DENSITY",
            "INTEGRATED_HOLE_DENSITY_FIGURE",
        ):
            with self.subTest(displayed_name=displayed_name):
                self.assert_displayed(displayed_name)

        self.assertFalse(self.calls_named("map_integrated_density_regions"))
        self.assertFalse(self.calls_named("build_region_material_map"))
        self.assertNotIn("INTEGRATED_REGION_MAP", self.combined_source)
        self.assertIn(
            "does not guess that mapping",
            self.combined_source,
        )

    def test_total_charge_diagnostics_use_selected_bias(self):
        read_calls = self.calls_named("read_total_charges", section_only=True)
        plot_calls = self.calls_named("plot_total_charges", section_only=True)
        self.assertEqual(len(read_calls), 1)
        self.assertEqual(len(plot_calls), 1)
        self.assertEqual(
            ast.unparse(read_calls[0][1].args[0]),
            "BIAS_DIRECTORY",
        )
        self.assertEqual(
            ast.unparse(plot_calls[0][1].args[0]),
            "BIAS_DIRECTORY",
        )
        self.assertIs(
            ast.literal_eval(keyword_map(plot_calls[0][1])["interactive"]),
            False,
        )
        self.assert_displayed("TOTAL_CHARGES")
        self.assert_displayed("TOTAL_CHARGES_FIGURE")

    def test_required_outputs_and_local_transferred_workflow_are_preserved(self):
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

        _, run_simulation = self.one_top_assignment("RUN_SIMULATION")
        self.assertIs(
            ast.literal_eval(assignment_value(run_simulation)),
            False,
        )
        for variable in (
            "RUN_DIRECTORY",
            "BIAS_DIRECTORY",
            "STRUCTURE_DIRECTORY",
        ):
            with self.subTest(variable=variable):
                self.assertIn(variable, self.combined_code)
        self.assertEqual(len(self.calls_named("run_input_file")), 1)
        self.assertEqual(len(self.calls_named("validate_run_directory")), 1)

        for banned_selection in (
            "find_latest_run",
            "st_mtime",
            "globals()",
            "matching_runs",
            "tagged_runs",
        ):
            with self.subTest(banned_selection=banned_selection):
                self.assertNotIn(banned_selection, self.combined_code)

        preserved_calls = {
            "LinearDotArrayDevice",
            "make_reference_sige_ge_process_stack",
            "build_simulation_layout",
            "write_nextnano_input_from_template",
            "resolve_bias_output_file",
            "plot_bias_volume_linecut",
            "plot_bias_volume_3d",
            "plot_bias_volume_slice",
            "resolve_quantum_output_file",
            "resolve_quantum_probability_state_file",
            "plot_quantum_density_volume_3d",
            "plot_quantum_density_volume_slice",
            "plot_quantum_density_volume_linecut",
            "plot_quantum_probability_volume_3d",
            "plot_quantum_probability_volume_slice",
            "plot_quantum_probability_volume_linecut",
        }
        for name in preserved_calls:
            with self.subTest(name=name):
                self.assertTrue(self.calls_named(name))

    def test_markdown_order_distinguishes_intended_and_solver_structure(self):
        markdown = {
            index: cell_source(cell)
            for index, cell in enumerate(self.cells)
            if cell["cell_type"] == "markdown"
        }

        def heading_index(title):
            expected = re.compile(
                rf"^#{{2,6}}\s+{re.escape(title)}\s*$",
                re.MULTILINE,
            )
            matches = [
                index
                for index, source in markdown.items()
                if expected.search(source)
            ]
            self.assertEqual(
                len(matches),
                1,
                f"Expected one heading titled {title!r}",
            )
            return matches[0]

        ordered_indices = [
            heading_index("Structure and diagnostics"),
            heading_index("Quantum–Poisson convergence"),
            heading_index("Integrated hole density"),
            heading_index("Total charge"),
            heading_index("Classical outputs"),
        ]
        self.assertEqual(ordered_indices, sorted(ordered_indices))

        structure_markdown = "\n".join(
            source
            for index, source in markdown.items()
            if self.structure_start <= index < self.physical_outputs_start
        ).lower()
        for phrase in (
            "intended geometry",
            "phidl",
            "process-stack",
            "explicitly validated solver run",
            "solver-structure views",
            "does not define a compatible 1d structure section",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, structure_markdown)

    def test_notebook_outputs_paths_and_trailing_cells_are_clean(self):
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
