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

EXPECTED_BASE_THESIS_FIGURES = {
    "structure_xy_qw": "HORIZONTAL_STRUCTURE_FIGURE",
    "structure_xz_qw": "VERTICAL_STRUCTURE_FIGURE",
    "convergence": "CONVERGENCE_FIGURE",
    "integrated_hole_density": "INTEGRATED_HOLE_DENSITY_FIGURE",
    "total_charge": "TOTAL_CHARGES_FIGURE",
    "classical_hole_density_xy_qw": "HOLE_DENSITY_PLANE_FIGURE",
    "classical_hole_density_x_cut": "HOLE_DENSITY_LINE_FIGURE",
    "electrostatic_potential_xy_qw": "POTENTIAL_PLANE_FIGURE",
    "electrostatic_potential_xz_center": "POTENTIAL_XZ_PLANE_FIGURE",
    "electrostatic_potential_x_cut": "POTENTIAL_LINE_FIGURE",
    "hh_bandedge_xy_qw": "HH_BAND_PLANE_FIGURE",
    "hh_bandedge_xz_center": "HH_BAND_XZ_PLANE_FIGURE",
    "hh_bandedge_x_cut": "HH_BAND_LINE_FIGURE",
    "hh_bandedge_z_cut_x_1150": "HH_BAND_Z_LINE_FIGURE",
}

EXPECTED_QUANTUM_THESIS_FIGURES = {
    "quantum_hh_density_xy_qw": "QUANTUM_DENSITY_PLANE_FIGURE",
    "quantum_hh_density_x_cut": "QUANTUM_DENSITY_LINE_FIGURE",
    "hh_probability_state_0001_xy_qw": "QUANTUM_PROBABILITY_PLANE_FIGURE",
    "hh_probability_state_0001_x_cut": "QUANTUM_PROBABILITY_LINE_FIGURE",
    "hh_occupation": "QUANTUM_OCCUPATION_FIGURE",
    "hh_energy_spectrum_k00000": "QUANTUM_ENERGY_FIGURE",
}

OPTIONAL_FIGURE_NAMES = {
    "SECONDARY_BAND_FIGURES",
    "CLASSICAL_3D_FIGURES",
    "ELECTRON_DENSITY_PLANE_FIGURE",
    "ELECTRON_DENSITY_LINE_FIGURE",
    "ELECTRON_DENSITY_3D_FIGURE",
    "QUANTUM_DENSITY_VOLUME_FIGURE",
    "QUANTUM_PROBABILITY_VOLUME_FIGURE",
}

EXPECTED_STATIC_CALL_TARGETS = {
    "HORIZONTAL_STRUCTURE_FIGURE": "plot_structure_plane",
    "VERTICAL_STRUCTURE_FIGURE": "plot_structure_plane",
    "CONVERGENCE_FIGURE": "plot_convergence",
    "INTEGRATED_HOLE_DENSITY_FIGURE": "plot_integrated_density_hole",
    "TOTAL_CHARGES_FIGURE": "plot_total_charges",
    "HOLE_DENSITY_PLANE_FIGURE": "plot_bias_volume_slice",
    "HOLE_DENSITY_LINE_FIGURE": "plot_bias_volume_linecut",
    "POTENTIAL_PLANE_FIGURE": "plot_bias_volume_slice",
    "POTENTIAL_XZ_PLANE_FIGURE": "plot_bias_volume_slice",
    "POTENTIAL_LINE_FIGURE": "plot_bias_volume_linecut",
    "HH_BAND_PLANE_FIGURE": "plot_bias_volume_slice",
    "HH_BAND_XZ_PLANE_FIGURE": "plot_bias_volume_slice",
    "HH_BAND_LINE_FIGURE": "plot_bias_volume_linecut",
    "HH_BAND_Z_LINE_FIGURE": "plot_bias_volume_linecut",
    "QUANTUM_DENSITY_PLANE_FIGURE": "plot_quantum_density_volume_slice",
    "QUANTUM_DENSITY_LINE_FIGURE": "plot_quantum_density_volume_linecut",
    "QUANTUM_PROBABILITY_PLANE_FIGURE":
        "plot_quantum_probability_volume_slice",
    "QUANTUM_PROBABILITY_LINE_FIGURE":
        "plot_quantum_probability_volume_linecut",
    "QUANTUM_OCCUPATION_FIGURE": "plot_quantum_occupation",
    "QUANTUM_ENERGY_FIGURE": "plot_quantum_energy_spectrum",
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


def keyword_map(call):
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def contains_identity(root, target):
    return any(node is target for node in ast.walk(root))


def dict_items(node):
    if not isinstance(node, ast.Dict):
        raise TypeError(f"Expected an AST dict, got {type(node).__name__}")
    return list(zip(node.keys, node.values))


class GenerateNextnanoInputNotebookFigureExportTests(unittest.TestCase):
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
        cls.markdown_sources = {
            index: cell_source(cell)
            for index, cell in enumerate(cls.cells)
            if cell["cell_type"] == "markdown"
        }

        final_headings = [
            index
            for index, source in cls.markdown_sources.items()
            if source.lstrip().splitlines()[0].strip().casefold()
            == "## figure export"
        ]
        if len(final_headings) != 1:
            raise AssertionError(
                "Expected exactly one figure-export heading"
            )
        cls.final_section_start = final_headings[0]
        cls.final_code_indices = [
            index
            for index, _ in cls.code_cells
            if index > cls.final_section_start
        ]
        cls.final_trees = {
            index: cls.trees[index] for index in cls.final_code_indices
        }
        cls.final_code = "\n".join(
            cls.code_sources[index] for index in cls.final_code_indices
        )

    def one_top_assignment(self, name):
        matches = self.top_assignments.get(name, [])
        self.assertEqual(
            len(matches),
            1,
            f"Expected one top-level assignment to {name}",
        )
        return matches[0]

    def calls_named(self, name):
        return [
            (index, call)
            for index, call in self.calls
            if call_name(call) == name
        ]

    def literal_top_assignment(self, name):
        return ast.literal_eval(
            assignment_value(self.one_top_assignment(name)[1])
        )

    def guarded_node_ids(self, condition):
        guards = []
        body_ids = set()
        else_ids = set()
        for index, tree in self.trees.items():
            for node in ast.walk(tree):
                if (
                    not isinstance(node, ast.If)
                    or ast.unparse(node.test) != condition
                ):
                    continue
                guards.append((index, node))
                for statement in node.body:
                    body_ids.update(id(item) for item in ast.walk(statement))
                for statement in node.orelse:
                    else_ids.update(id(item) for item in ast.walk(statement))
        return guards, body_ids, else_ids

    def assigned_target(self, call):
        matches = []
        for tree in self.trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    if contains_identity(node.value, call):
                        matches.extend(ast.unparse(target) for target in node.targets)
                elif isinstance(node, ast.AnnAssign):
                    if node.value is not None and contains_identity(node.value, call):
                        matches.append(ast.unparse(node.target))
        self.assertEqual(
            len(matches),
            1,
            f"Expected one assignment target for {ast.unparse(call)}",
        )
        return matches[0]

    def resolve_string(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if not isinstance(node, ast.JoinedStr):
            self.fail(f"Expected a string expression, got {ast.unparse(node)}")

        result = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                result.append(str(part.value))
                continue
            self.assertIsInstance(part, ast.FormattedValue)
            self.assertIsInstance(part.value, ast.Name)
            value = self.literal_top_assignment(part.value.id)
            format_spec = ""
            if part.format_spec is not None:
                self.assertIsInstance(part.format_spec, ast.JoinedStr)
                format_spec = "".join(
                    str(value_part.value)
                    for value_part in part.format_spec.values
                    if isinstance(value_part, ast.Constant)
                )
            result.append(format(value, format_spec))
        return "".join(result)

    def resolved_figure_mapping(self, node):
        mapping = {}
        for key, value in dict_items(node):
            stem = self.resolve_string(key)
            self.assertIsInstance(
                value,
                ast.Name,
                f"Figure {stem!r} must map directly to a named figure",
            )
            mapping[stem] = value.id
        return mapping

    def assigned_call(self, target, api):
        matches = []
        for index, tree in self.trees.items():
            for node in ast.walk(tree):
                if assignment_name(node) != target:
                    continue
                value = assignment_value(node)
                if isinstance(value, ast.Call) and call_name(value) == api:
                    matches.append((index, value))

        self.assertEqual(
            len(matches),
            1,
            f"Expected one {target} assignment from {api}",
        )
        return matches[0]

    def test_export_configuration_is_disabled_external_and_central(self):
        self.assertIs(self.literal_top_assignment("EXPORT_FIGURES"), False)

        _, output_root = self.one_top_assignment("FIGURE_OUTPUT_ROOT")
        output_expression = ast.unparse(assignment_value(output_root))
        for fragment in (
            "os.environ.get('THESIS_FIGURE_ROOT'",
            "REPO_ROOT.parent / 'qpu-local-outputs' / 'refactoring' / 'figures'",
            ".expanduser()",
            ".resolve()",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, output_expression)

        _, directory = self.one_top_assignment("FIGURE_DIRECTORY")
        self.assertEqual(
            ast.unparse(assignment_value(directory)),
            (
                "FIGURE_OUTPUT_ROOT / "
                "'notebook_03_lateral_double_quantum_dot'"
            ),
        )
        self.assertEqual(
            self.literal_top_assignment("FIGURE_FORMATS"),
            ("pdf", "png"),
        )
        self.assertGreaterEqual(
            self.literal_top_assignment("FIGURE_PNG_DPI"),
            300,
        )

        guards = [
            node
            for tree in self.trees.values()
            for node in tree.body
            if isinstance(node, ast.If)
            and "FIGURE_OUTPUT_ROOT == REPO_ROOT" in ast.unparse(node.test)
            and "REPO_ROOT in FIGURE_OUTPUT_ROOT.parents"
            in ast.unparse(node.test)
        ]
        self.assertEqual(len(guards), 1)
        guard_raises = [
            node
            for node in ast.walk(guards[0])
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and call_name(node.exc) == "ValueError"
        ]
        self.assertEqual(len(guard_raises), 1)
        self.assertIn("outside the repository", ast.unparse(guard_raises[0]))

        self.assertIn("THESIS_FIGURE_ROOT", self.combined_source)
        self.assertRegex(
            self.combined_source,
            r"THESIS_FIGURE_ROOT.*thesis|thesis.*THESIS_FIGURE_ROOT",
        )
        self.assertNotIn("/Users/robertjovanov/", self.combined_source)
        self.assertIsNone(re.search(r"[A-Za-z]:\\Users\\", self.combined_source))

    def test_figure_directory_is_created_only_inside_export_guard(self):
        guards, export_ids, export_else_ids = self.guarded_node_ids(
            "EXPORT_FIGURES"
        )
        self.assertEqual(len(guards), 1)

        figure_mkdir_calls = [
            call
            for _, call in self.calls_named("mkdir")
            if isinstance(call.func, ast.Attribute)
            and "FIGURE_" in ast.unparse(call.func.value)
        ]
        self.assertEqual(len(figure_mkdir_calls), 1)
        mkdir_call = figure_mkdir_calls[0]
        self.assertEqual(
            ast.unparse(mkdir_call.func.value),
            "FIGURE_DIRECTORY",
        )
        self.assertIn(id(mkdir_call), export_ids)
        self.assertNotIn(id(mkdir_call), export_else_ids)
        mkdir_keywords = keyword_map(mkdir_call)
        self.assertIs(ast.literal_eval(mkdir_keywords["parents"]), True)
        self.assertIs(ast.literal_eval(mkdir_keywords["exist_ok"]), True)

        export_control_index = self.one_top_assignment("EXPORT_FIGURES")[0]
        export_setting_indices = {
            self.one_top_assignment(name)[0]
            for name in (
                "FIGURE_OUTPUT_ROOT",
                "FIGURE_DIRECTORY",
                "FIGURE_FORMATS",
                "FIGURE_PNG_DPI",
            )
        }
        self.assertLess(export_control_index, self.final_section_start)
        self.assertEqual(len(export_setting_indices), 1)
        self.assertTrue(
            all(
                index > self.final_section_start
                for index in export_setting_indices
            )
        )
        self.assertFalse(
            any(
                index in export_setting_indices
                for index, _ in self.calls_named("mkdir")
            )
        )

    def test_default_static_figures_are_assigned_and_still_displayed(self):
        displayed_names = {
            node.id
            for _, call in self.calls_named("display")
            if call.args
            for node in ast.walk(call.args[0])
            if isinstance(node, ast.Name)
        }

        for expected_target, api in EXPECTED_STATIC_CALL_TARGETS.items():
            index, call = self.assigned_call(expected_target, api)
            with self.subTest(api=api, target=expected_target):
                self.assertIn(expected_target, displayed_names)
                keywords = keyword_map(call)
                self.assertIn("interactive", keywords)
                self.assertIs(
                    ast.literal_eval(keywords["interactive"]),
                    False,
                )

                if expected_target == "HORIZONTAL_STRUCTURE_FIGURE":
                    self.assertIn(
                        'QUANTITY = "regions_all_2d_xy_QD"',
                        self.code_sources[index],
                    )
                    self.assertEqual(
                        ast.unparse(keywords["quantity"]),
                        "QUANTITY",
                    )
                elif expected_target == "VERTICAL_STRUCTURE_FIGURE":
                    self.assertIn(
                        'QUANTITY = "materials_2d_xz_QD"',
                        self.code_sources[index],
                    )
                    self.assertEqual(
                        ast.unparse(keywords["quantity"]),
                        "QUANTITY",
                    )

        captured_plot_apis = {
            "plot_structure_plane",
            "plot_convergence",
            "plot_integrated_density_hole",
            "plot_total_charges",
            "plot_bias_volume_slice",
            "plot_bias_volume_linecut",
            "plot_bias_volume_3d",
            "plot_quantum_density_volume_slice",
            "plot_quantum_density_volume_linecut",
            "plot_quantum_density_volume_3d",
            "plot_quantum_probability_volume_slice",
            "plot_quantum_probability_volume_linecut",
            "plot_quantum_probability_volume_3d",
            "plot_quantum_occupation",
            "plot_quantum_energy_spectrum",
        }
        for api in captured_plot_apis:
            for _, call in self.calls_named(api):
                with self.subTest(api=api, call=ast.unparse(call)):
                    self.assigned_target(call)

    def test_thesis_mapping_has_base_and_conditional_quantum_figures(self):
        mapping_index, mapping_assignment = self.one_top_assignment(
            "THESIS_FIGURES"
        )
        self.assertGreater(mapping_index, self.final_section_start)
        base_mapping = self.resolved_figure_mapping(
            assignment_value(mapping_assignment)
        )
        self.assertEqual(base_mapping, EXPECTED_BASE_THESIS_FIGURES)

        update_calls = [
            (index, call)
            for index, call in self.calls_named("update")
            if isinstance(call.func, ast.Attribute)
            and ast.unparse(call.func.value) == "THESIS_FIGURES"
        ]
        self.assertEqual(len(update_calls), 1)
        update_index, update_call = update_calls[0]
        self.assertGreater(update_index, self.final_section_start)
        self.assertEqual(len(update_call.args), 1)
        quantum_mapping = self.resolved_figure_mapping(update_call.args[0])
        self.assertEqual(quantum_mapping, EXPECTED_QUANTUM_THESIS_FIGURES)

        quantum_guards, guarded_ids, guarded_else_ids = self.guarded_node_ids(
            "ANALYSE_QUANTUM_OUTPUTS"
        )
        self.assertTrue(quantum_guards)
        self.assertIn(id(update_call), guarded_ids)
        self.assertNotIn(id(update_call), guarded_else_ids)

        all_mapping_source = (
            ast.unparse(assignment_value(mapping_assignment))
            + "\n"
            + ast.unparse(update_call.args[0])
        )
        for optional_name in OPTIONAL_FIGURE_NAMES:
            with self.subTest(optional_name=optional_name):
                self.assertNotIn(optional_name, all_mapping_source)
        self.assertNotIn("None", all_mapping_source)

        for plotly_api in (
            "plot_bias_volume_3d",
            "plot_quantum_density_volume_3d",
            "plot_quantum_probability_volume_3d",
        ):
            for _, call in self.calls_named(plotly_api):
                self.assertNotIn(
                    self.assigned_target(call),
                    set(base_mapping.values()) | set(quantum_mapping.values()),
                )

    def test_export_guard_validates_and_saves_every_configured_format(self):
        guards, export_ids, export_else_ids = self.guarded_node_ids(
            "EXPORT_FIGURES"
        )
        self.assertEqual(len(guards), 1)
        export_guard = guards[0][1]

        save_calls = [
            call
            for _, call in self.calls_named("savefig")
            if id(call) in export_ids
        ]
        self.assertTrue(save_calls)
        self.assertFalse(
            [
                call
                for _, call in self.calls_named("savefig")
                if id(call) not in export_ids
            ]
        )
        self.assertTrue(all(id(call) not in export_else_ids for call in save_calls))

        figure_loops = [
            node
            for node in ast.walk(export_guard)
            if isinstance(node, ast.For)
            and ast.unparse(node.iter) == "THESIS_FIGURES.items()"
        ]
        self.assertEqual(len(figure_loops), 1)
        figure_loop = figure_loops[0]
        format_loops = [
            node
            for node in ast.walk(figure_loop)
            if isinstance(node, ast.For)
            and ast.unparse(node.iter) == "FIGURE_FORMATS"
        ]
        self.assertEqual(len(format_loops), 1)
        format_loop = format_loops[0]
        self.assertTrue(
            any(contains_identity(format_loop, call) for call in save_calls)
        )

        format_loop_source = ast.unparse(format_loop)
        for fragment in (
            "FIGURE_DIRECTORY",
            "figure_stem",
            "figure_format",
            "bbox_inches",
            "'tight'",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, format_loop_source)

        type_checks = [
            node
            for node in ast.walk(export_guard)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and assignment_name(node) == "is_matplotlib_figure"
        ]
        self.assertEqual(len(type_checks), 1)
        type_check_source = ast.unparse(assignment_value(type_checks[0]))
        for fragment in (
            "type(figure).__module__ == 'matplotlib.figure'",
            "type(figure).__name__ == 'Figure'",
        ):
            with self.subTest(type_fragment=fragment):
                self.assertIn(fragment, type_check_source)
        type_errors = [
            node
            for node in ast.walk(export_guard)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and call_name(node.exc) == "TypeError"
        ]
        self.assertEqual(len(type_errors), 1)

        value_errors = [
            node
            for node in ast.walk(export_guard)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and call_name(node.exc) == "ValueError"
        ]
        self.assertTrue(value_errors)
        value_error_source = "\n".join(ast.unparse(node) for node in value_errors)
        self.assertIn("Unsupported", value_error_source)
        self.assertIn("format", value_error_source.casefold())
        self.assertIn("FIGURE_FORMATS", ast.unparse(export_guard))

        png_guards = [
            node
            for node in ast.walk(format_loop)
            if isinstance(node, ast.If)
            and "figure_format" in ast.unparse(node.test)
            and "'png'" in ast.unparse(node.test)
        ]
        self.assertEqual(len(png_guards), 1)
        png_guard = png_guards[0]
        self.assertIn("FIGURE_PNG_DPI", ast.unparse(png_guard.body))
        self.assertNotIn("FIGURE_PNG_DPI", ast.unparse(png_guard.orelse))

        direct_dpi_keywords = [
            keyword
            for call in save_calls
            for keyword in call.keywords
            if keyword.arg == "dpi"
        ]
        self.assertTrue(
            all(
                any(contains_identity(png_guard, keyword) for _ in (None,))
                for keyword in direct_dpi_keywords
            )
        )
        for call in save_calls:
            self.assertFalse(
                any(keyword.arg == "dpi" for keyword in call.keywords)
                and not contains_identity(png_guard, call)
            )

        call_names = {
            call_name(node)
            for node in ast.walk(export_guard)
            if isinstance(node, ast.Call)
        }
        self.assertTrue({"show", "close"}.isdisjoint(call_names))

        mapped_figure_variables = set(EXPECTED_BASE_THESIS_FIGURES.values())
        mapped_figure_variables.update(EXPECTED_QUANTUM_THESIS_FIGURES.values())
        overwritten = {
            node.id
            for node in ast.walk(export_guard)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in mapped_figure_variables
        }
        self.assertEqual(overwritten, set())

    def test_manifest_is_guarded_relative_and_reproducible(self):
        _, export_ids, export_else_ids = self.guarded_node_ids(
            "EXPORT_FIGURES"
        )
        manifest_writes = [
            call
            for _, call in self.calls_named("write_text")
            if id(call) in export_ids
            and isinstance(call.func, ast.Attribute)
            and ast.unparse(call.func.value) == "FIGURE_MANIFEST_PATH"
        ]
        self.assertEqual(len(manifest_writes), 1)
        manifest_write = manifest_writes[0]
        self.assertNotIn(id(manifest_write), export_else_ids)
        self.assertIn("json.dumps", ast.unparse(manifest_write))
        self.assertEqual(
            ast.literal_eval(keyword_map(manifest_write)["encoding"]),
            "utf-8",
        )
        self.assertFalse(
            [
                call
                for _, call in self.calls_named("write_text")
                if isinstance(call.func, ast.Attribute)
                and ast.unparse(call.func.value) == "FIGURE_MANIFEST_PATH"
                and id(call) not in export_ids
            ]
        )
        manifest_path_assignments = [
            node
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if id(node) in export_ids
            and assignment_name(node) == "FIGURE_MANIFEST_PATH"
        ]
        self.assertEqual(len(manifest_path_assignments), 1)
        self.assertEqual(
            ast.unparse(assignment_value(manifest_path_assignments[0])),
            "FIGURE_DIRECTORY / 'figure_manifest.json'",
        )

        required_manifest_fields = {
            "notebook_identifier",
            "run_directory",
            "bias",
            "generated_input_path",
            "xy_plane_z_nm",
            "xz_plane_y_nm",
            "x_line_fixed_coordinates_nm",
            "z_line_fixed_coordinates_nm",
            "analyse_quantum_outputs",
            "quantum",
            "figure_files",
            "formats",
            "png_dpi",
        }
        manifest_candidates = []
        for tree in self.trees.values():
            for node in ast.walk(tree):
                if id(node) not in export_ids or not isinstance(node, ast.Dict):
                    continue
                keys = {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                }
                if required_manifest_fields.issubset(keys):
                    manifest_candidates.append(node)
        self.assertEqual(len(manifest_candidates), 1)
        manifest = manifest_candidates[0]
        manifest_mapping = {
            ast.literal_eval(key): value
            for key, value in dict_items(manifest)
            if isinstance(key, ast.Constant)
        }
        self.assertEqual(
            ast.literal_eval(manifest_mapping["notebook_identifier"]),
            "03_generate_nextnano_input_from_phidl_layout",
        )
        self.assertEqual(
            ast.unparse(manifest_mapping["run_directory"]),
            "str(RUN_DIRECTORY)",
        )
        self.assertEqual(ast.unparse(manifest_mapping["bias"]), "BIAS")
        generated_input_source = ast.unparse(
            manifest_mapping["generated_input_path"]
        )
        self.assertIn("str(GENERATED_INPUT_PATH)", generated_input_source)
        self.assertIn("RUN_SIMULATION", generated_input_source)
        self.assertIn("else None", generated_input_source)
        self.assertEqual(
            ast.unparse(manifest_mapping["xy_plane_z_nm"]),
            "XY_PLANE_Z_NM",
        )
        self.assertEqual(
            ast.unparse(manifest_mapping["xz_plane_y_nm"]),
            "XZ_PLANE_Y_NM",
        )
        x_line_mapping = {
            ast.literal_eval(key): ast.unparse(value)
            for key, value in dict_items(
                manifest_mapping["x_line_fixed_coordinates_nm"]
            )
        }
        self.assertEqual(
            x_line_mapping,
            {"y": "X_LINE_Y_NM", "z": "X_LINE_Z_NM"},
        )
        z_line_mapping = {
            ast.literal_eval(key): ast.unparse(value)
            for key, value in dict_items(
                manifest_mapping["z_line_fixed_coordinates_nm"]
            )
        }
        self.assertEqual(
            z_line_mapping,
            {"x": "Z_LINE_X_NM", "y": "Z_LINE_Y_NM"},
        )
        self.assertTrue(
            {
                "qw_plane_z_nm",
                "line_axis",
                "line_fixed_coordinates_nm",
            }.isdisjoint(manifest_mapping)
        )
        self.assertEqual(
            ast.unparse(manifest_mapping["analyse_quantum_outputs"]),
            "ANALYSE_QUANTUM_OUTPUTS",
        )
        self.assertIn(
            "FIGURE_FORMATS",
            ast.unparse(manifest_mapping["formats"]),
        )
        self.assertEqual(
            ast.unparse(manifest_mapping["png_dpi"]),
            "FIGURE_PNG_DPI",
        )

        quantum_source = ast.unparse(manifest_mapping["quantum"])
        for required_name in (
            "ANALYSE_QUANTUM_OUTPUTS",
            "QUANTUM_REGION",
            "QUANTUM_BAND",
            "QUANTUM_KPOINT",
            "QUANTUM_STATE",
        ):
            with self.subTest(required_name=required_name):
                self.assertIn(required_name, quantum_source)

        figures_name = ast.unparse(manifest_mapping["figure_files"])
        self.assertEqual(figures_name, "exported_figure_files")
        self.assertIn(
            "figure_filenames.append(figure_filename)",
            self.final_code,
        )
        self.assertIn(
            "exported_figure_files[figure_stem] = figure_filenames",
            self.final_code,
        )
        self.assertNotIn("str(figure_path)", self.final_code)
        self.assertNotIn("timestamp", self.final_code.casefold())

    def test_final_summary_is_concise_and_complete(self):
        summary_candidates = []
        required_names = {
            "RUN_DIRECTORY",
            "BIAS",
            "XY_PLANE_Z_NM",
            "XZ_PLANE_Y_NM",
            "X_LINE_Y_NM",
            "X_LINE_Z_NM",
            "Z_LINE_X_NM",
            "Z_LINE_Y_NM",
            "ANALYSE_QUANTUM_OUTPUTS",
            "THESIS_FIGURES",
            "FIGURE_DIRECTORY",
            "EXPORT_FIGURES",
        }
        for index, tree in self.final_trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = assignment_value(node)
                if not isinstance(value, ast.Call):
                    continue
                if call_name(value) not in {"Series", "DataFrame"}:
                    continue
                source = ast.unparse(value)
                if all(name in source for name in required_names):
                    summary_candidates.append((index, node))
        self.assertEqual(len(summary_candidates), 1)
        summary_index, summary_assignment = summary_candidates[0]
        summary_name = assignment_name(summary_assignment)
        self.assertIsNotNone(summary_name)
        self.assertIn(
            "len(THESIS_FIGURES)",
            ast.unparse(assignment_value(summary_assignment)),
        )

        displayed = [
            call
            for index, call in self.calls_named("display")
            if index == summary_index
            and call.args
            and any(
                isinstance(node, ast.Name) and node.id == summary_name
                for node in ast.walk(call.args[0])
            )
        ]
        self.assertEqual(len(displayed), 1)

    def test_previous_workflow_and_static_safety_are_preserved(self):
        self.assertIs(self.literal_top_assignment("RUN_SIMULATION"), False)
        self.assertIs(
            self.literal_top_assignment("ANALYSE_QUANTUM_OUTPUTS"),
            True,
        )
        self.assertIn("RUN_DIRECTORY", self.top_assignments)
        self.assertEqual(len(self.calls_named("validate_run_directory")), 1)

        h2_headings = [
            line.removeprefix("## ").strip()
            for source in self.markdown_sources.values()
            for line in source.splitlines()
            if line.startswith("## ")
        ]
        self.assertEqual(
            h2_headings,
            [
                "Setup",
                "User controls",
                "Device layout",
                "Process stack and simulation layout",
                "Input generation",
                "Run or select completed output",
                "Structure and diagnostics",
                "Classical outputs",
                "Quantum outputs",
                "Figure export",
                "Summary",
            ],
        )

        definitions = {
            node.name
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        }
        self.assertEqual(definitions, set())

        for fragment in (
            "sys.path.append",
            "sys.path.insert",
            "importlib.reload",
            "find_latest_run",
            "st_mtime",
            "getmtime",
            "matching_runs",
            "tagged_runs",
            "kaleido",
            "imageio",
            "plotly.io",
            "nextnano_analysis",
            ".write_image",
            ".to_image",
        ):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.combined_source)

        imported_modules = {
            alias.name
            for tree in self.trees.values()
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("sys", imported_modules)
        self.assertNotIn("importlib", imported_modules)

    def test_notebook_json_is_clean_and_has_no_empty_cells(self):
        self.assertTrue(self.cells)
        for index, cell in enumerate(self.cells):
            with self.subTest(cell=index):
                self.assertTrue(
                    cell_source(cell).strip(),
                    f"Notebook cell {index} is empty",
                )
                self.assertFalse(cell.get("attachments"))
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])

        self.assertTrue(cell_source(self.cells[-1]).strip())


if __name__ == "__main__":
    unittest.main()
