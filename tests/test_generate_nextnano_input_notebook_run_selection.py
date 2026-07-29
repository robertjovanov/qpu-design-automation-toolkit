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


class GenerateNextnanoInputNotebookRunSelectionTests(unittest.TestCase):
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

    def one_top_assignment(self, name):
        matches = self.top_assignments.get(name, [])
        self.assertEqual(len(matches), 1, f"Expected one top-level assignment to {name}")
        return matches[0]

    def calls_named(self, name):
        return [
            (index, call)
            for index, call in self.calls
            if call_name(call) == name
        ]

    def test_installed_direct_imports_replace_path_and_reload_hacks(self):
        imported_modules = set()
        imported_by_module = {}
        aliases_by_module = {}
        for tree in self.trees.values():
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported_by_module.setdefault(node.module, set()).update(
                        alias.name for alias in node.names
                    )
                    aliases_by_module.setdefault(node.module, []).extend(node.names)

        self.assertIn("os", imported_modules)
        self.assertIn("re", imported_modules)
        self.assertIn("Path", imported_by_module.get("pathlib", set()))
        self.assertTrue(
            {
                "LinearDotArrayDevice",
                "build_simulation_layout",
                "make_reference_sige_ge_process_stack",
                "write_nextnano_input_from_template",
            }.issubset(imported_by_module.get("qd_design", set()))
        )

        required_nextnanopp_imports = {
            "get_bias_dir",
            "get_output_directory",
            "list_variables",
            "plot_bias_volume_linecut",
            "plot_bias_volume_3d",
            "plot_bias_volume_slice",
            "plot_quantum_density_volume_3d",
            "plot_quantum_density_volume_linecut",
            "plot_quantum_density_volume_slice",
            "plot_quantum_probability_volume_3d",
            "plot_quantum_probability_volume_linecut",
            "plot_quantum_probability_volume_slice",
            "resolve_bias_output_file",
            "resolve_quantum_output_file",
            "resolve_quantum_probability_state_file",
            "run_input_file",
            "validate_run_directory",
        }
        self.assertTrue(
            required_nextnanopp_imports.issubset(
                imported_by_module.get("nextnanopp_tools", set())
            )
        )
        self.assertTrue(
            all(
                alias.asname is None
                for alias in aliases_by_module.get("nextnanopp_tools", [])
            )
        )

        for banned_fragment in (
            "sys.path.append",
            "sys.path.insert",
            "importlib.reload",
            "import nextnanopp_tools as",
            "nnt.",
        ):
            with self.subTest(banned_fragment=banned_fragment):
                self.assertNotIn(banned_fragment, self.combined_code)
        self.assertNotIn("sys", imported_modules)
        self.assertNotIn("importlib", imported_modules)

    def test_repository_root_searches_cwd_and_parents(self):
        _, start_assignment = self.one_top_assignment("START_PATH")
        self.assertEqual(
            ast.unparse(assignment_value(start_assignment)),
            "Path.cwd().resolve()",
        )

        _, root_assignment = self.one_top_assignment("REPO_ROOT")
        root_expression = assignment_value(root_assignment)
        self.assertIsInstance(root_expression, ast.Call)
        self.assertEqual(ast.unparse(root_expression.func), "next")
        self.assertEqual(len(root_expression.args), 2)
        self.assertIsInstance(root_expression.args[0], ast.GeneratorExp)
        self.assertIsNone(ast.literal_eval(root_expression.args[1]))
        root_source = ast.unparse(root_expression)
        for fragment in (
            "(START_PATH, *START_PATH.parents)",
            "'pyproject.toml'",
            "'src'",
            ".is_file()",
            ".is_dir()",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, root_source)

        guards = [
            node
            for tree in self.trees.values()
            for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "REPO_ROOT is None"
        ]
        self.assertEqual(len(guards), 1)
        raises = [
            node for node in ast.walk(guards[0]) if isinstance(node, ast.Raise)
        ]
        self.assertEqual(len(raises), 1)
        self.assertEqual(call_name(raises[0].exc), "RuntimeError")
        self.assertIn(
            "Could not locate the repository root",
            ast.unparse(raises[0]),
        )

    def test_external_output_root_and_inside_repository_guard(self):
        _, output_root_assignment = self.one_top_assignment(
            "SIMULATION_OUTPUT_ROOT"
        )
        output_root_source = ast.unparse(assignment_value(output_root_assignment))
        for fragment in (
            "os.environ.get('NEXTNANO_OUTPUT_ROOT'",
            "REPO_ROOT.parent / 'qpu-local-outputs' / 'refactoring' / 'runs'",
            ".expanduser()",
            ".resolve()",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, output_root_source)

        _, staging_root_assignment = self.one_top_assignment(
            "SIMULATION_STAGING_ROOT"
        )
        staging_root_source = ast.unparse(
            assignment_value(staging_root_assignment)
        )
        self.assertEqual(
            staging_root_source,
            "SIMULATION_OUTPUT_ROOT / '_staging'",
        )
        for tracked_root in ("GENERATED_INPUT_DIR", "REPO_ROOT"):
            with self.subTest(tracked_root=tracked_root):
                self.assertNotIn(tracked_root, staging_root_source)

        guards = [
            node
            for tree in self.trees.values()
            for node in tree.body
            if isinstance(node, ast.If)
            and "SIMULATION_OUTPUT_ROOT == REPO_ROOT" in ast.unparse(node.test)
            and "REPO_ROOT in SIMULATION_OUTPUT_ROOT.parents"
            in ast.unparse(node.test)
        ]
        self.assertEqual(len(guards), 1)
        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(guards[0]))
        )

        output_root_mkdir_calls = [
            call
            for _, call in self.calls_named("mkdir")
            if isinstance(call.func, ast.Attribute)
            and ast.unparse(call.func.value) == "SIMULATION_OUTPUT_ROOT"
        ]
        self.assertEqual(output_root_mkdir_calls, [])
        staging_root_mkdir_calls = [
            call
            for _, call in self.calls_named("mkdir")
            if isinstance(call.func, ast.Attribute)
            and ast.unparse(call.func.value) == "SIMULATION_STAGING_ROOT"
        ]
        self.assertEqual(staging_root_mkdir_calls, [])
        self.assertNotIn(
            "GENERATED_INPUT_DIR / 'staged_runs'",
            self.combined_code,
        )

    def test_central_run_configuration_and_required_outputs(self):
        config_names = (
            "RUN_SIMULATION",
            "RUN_TAG",
            "BIAS",
            "ANALYSE_QUANTUM_OUTPUTS",
            "SIMULATION_OUTPUT_ROOT",
            "SIMULATION_STAGING_ROOT",
            "BASE_REQUIRED_OUTPUTS",
            "QUANTUM_REQUIRED_OUTPUTS",
            "REQUIRED_OUTPUTS",
        )
        config_cells = {
            self.one_top_assignment(name)[0] for name in config_names
        }

        run_directory_initial = [
            (index, node)
            for index, node in self.top_assignments.get("RUN_DIRECTORY", [])
            if isinstance(node, ast.AnnAssign)
        ]
        self.assertEqual(len(run_directory_initial), 1)
        config_cells.add(run_directory_initial[0][0])
        self.assertEqual(len(config_cells), 1)
        config_cell = config_cells.pop()

        self.assertIs(
            ast.literal_eval(
                assignment_value(self.one_top_assignment("RUN_SIMULATION")[1])
            ),
            False,
        )
        self.assertEqual(
            ast.literal_eval(
                assignment_value(self.one_top_assignment("RUN_TAG")[1])
            ),
            "geometry_check_no_padding",
        )
        self.assertEqual(
            ast.literal_eval(assignment_value(self.one_top_assignment("BIAS")[1])),
            "bias_00000",
        )
        self.assertIs(
            ast.literal_eval(
                assignment_value(
                    self.one_top_assignment("ANALYSE_QUANTUM_OUTPUTS")[1]
                )
            ),
            True,
        )

        _, run_directory_node = run_directory_initial[0]
        self.assertEqual(ast.unparse(run_directory_node.annotation), "Path | None")
        self.assertIsNone(ast.literal_eval(run_directory_node.value))
        config_source = self.code_sources[config_cell]
        self.assertIn(
            "RUN_SIMULATION=True and leave RUN_DIRECTORY=None",
            config_source,
        )
        self.assertIn(
            "RUN_SIMULATION=False and set RUN_DIRECTORY",
            config_source,
        )
        self.assertIn(
            "True: require and analyse HH quantum outputs.",
            config_source,
        )
        self.assertIn(
            "False: allow a classical-only run and skip the quantum section.",
            config_source,
        )

        base_required_outputs = ast.literal_eval(
            assignment_value(
                self.one_top_assignment("BASE_REQUIRED_OUTPUTS")[1]
            )
        )
        self.assertEqual(
            base_required_outputs,
            EXPECTED_BASE_REQUIRED_OUTPUTS,
        )
        quantum_required_outputs = ast.literal_eval(
            assignment_value(
                self.one_top_assignment("QUANTUM_REQUIRED_OUTPUTS")[1]
            )
        )
        self.assertEqual(
            quantum_required_outputs,
            EXPECTED_QUANTUM_REQUIRED_OUTPUTS,
        )

        required_outputs_expression = assignment_value(
            self.one_top_assignment("REQUIRED_OUTPUTS")[1]
        )
        self.assertIsInstance(required_outputs_expression, ast.BinOp)
        self.assertIsInstance(required_outputs_expression.op, ast.Add)
        self.assertEqual(
            ast.unparse(required_outputs_expression.left),
            "BASE_REQUIRED_OUTPUTS",
        )
        self.assertIsInstance(required_outputs_expression.right, ast.IfExp)
        self.assertEqual(
            ast.unparse(required_outputs_expression.right.test),
            "ANALYSE_QUANTUM_OUTPUTS",
        )
        self.assertEqual(
            ast.unparse(required_outputs_expression.right.body),
            "QUANTUM_REQUIRED_OUTPUTS",
        )
        self.assertEqual(
            ast.literal_eval(required_outputs_expression.right.orelse),
            (),
        )

        all_required_outputs = (
            base_required_outputs + quantum_required_outputs
        )
        self.assertNotIn("density_electron.vtr", all_required_outputs)
        self.assertFalse(
            any("/LH/" in path or "/SO/" in path for path in all_required_outputs)
        )

    def test_quantum_solver_compatibility_is_checked_before_local_execution(self):
        selection_matches = [
            node
            for tree in self.trees.values()
            for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "RUN_SIMULATION"
        ]
        self.assertEqual(len(selection_matches), 1)
        selection = selection_matches[0]

        quantum_guards = [
            node
            for node in selection.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "ANALYSE_QUANTUM_OUTPUTS"
        ]
        self.assertEqual(len(quantum_guards), 1)
        quantum_guard = quantum_guards[0]

        run_calls = [
            node
            for node in ast.walk(selection)
            if isinstance(node, ast.Call)
            and call_name(node) == "run_input_file"
        ]
        self.assertEqual(len(run_calls), 1)
        run_call = run_calls[0]
        run_statements = [
            statement
            for statement in selection.body
            if any(node is run_call for node in ast.walk(statement))
        ]
        self.assertEqual(len(run_statements), 1)
        self.assertLess(
            selection.body.index(quantum_guard),
            selection.body.index(run_statements[0]),
        )
        self.assertFalse(
            any(node is run_call for node in ast.walk(quantum_guard))
        )

        read_calls = [
            node
            for node in ast.walk(quantum_guard)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
            and ast.unparse(node.func.value) == "GENERATED_INPUT_PATH"
        ]
        self.assertEqual(len(read_calls), 1)
        self.assertEqual(
            ast.literal_eval(keyword_map(read_calls[0])["encoding"]),
            "utf-8",
        )

        fullmatch_calls = [
            node
            for node in ast.walk(quantum_guard)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value) == "re"
            and node.func.attr == "fullmatch"
        ]
        self.assertEqual(len(fullmatch_calls), 1)
        pattern = ast.literal_eval(fullmatch_calls[0].args[0])
        self.assertEqual(
            pattern,
            r"\$(quantum_poisson|quantum)\s*=\s*([01])",
        )

        solver_pattern = re.compile(pattern)

        def parse_solver_line(line):
            uncommented = line.split("#", 1)[0].strip()
            return solver_pattern.fullmatch(uncommented)

        accepted_lines = {
            "$quantum=1": ("quantum", "1"),
            "  $quantum = 0  ": ("quantum", "0"),
            "\t$quantum_poisson\t=\t1": ("quantum_poisson", "1"),
            "$quantum_poisson = 0 # ignored comment": (
                "quantum_poisson",
                "0",
            ),
        }
        for line, expected_groups in accepted_lines.items():
            with self.subTest(accepted=line):
                match = parse_solver_line(line)
                self.assertIsNotNone(match)
                self.assertEqual(match.groups(), expected_groups)

        for line in (
            "# $quantum = 1",
            "$quantum_solver = 1",
            "$quantum_poisson_extra = 1",
            "$not_quantum = 1",
            "$quantum = 10",
            "prefix $quantum = 1",
        ):
            with self.subTest(rejected=line):
                self.assertIsNone(parse_solver_line(line))

        any_calls = [
            node
            for node in ast.walk(quantum_guard)
            if isinstance(node, ast.Call)
            and call_name(node) == "any"
        ]
        self.assertEqual(len(any_calls), 1)
        any_source = ast.unparse(any_calls[0])
        self.assertIn("quantum_solver_switches.get(name, 0)", any_source)
        self.assertIn("('quantum', 'quantum_poisson')", any_source)

        runtime_errors = [
            node
            for node in ast.walk(quantum_guard)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and call_name(node.exc) == "RuntimeError"
        ]
        self.assertEqual(len(runtime_errors), 1)
        error_source = ast.unparse(runtime_errors[0])
        for fragment in (
            "template/generated input",
            "one-shot quantum solver",
            "quantum–Poisson solver",
            "required quantum outputs",
            "not changed automatically",
        ):
            with self.subTest(error_fragment=fragment):
                self.assertIn(fragment, error_source)

        mutation_calls = {
            "write_text",
            "write_bytes",
            "replace",
            "sub",
            "subn",
        }
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and call_name(node) in mutation_calls
                for node in ast.walk(quantum_guard)
            )
        )

    def test_dual_mode_selection_and_validation_are_explicit(self):
        selection_matches = [
            (index, node)
            for index, tree in self.trees.items()
            for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "RUN_SIMULATION"
        ]
        self.assertEqual(len(selection_matches), 1)
        selection_index, selection = selection_matches[0]
        selection_source = ast.unparse(selection)

        for fragment in (
            "if RUN_DIRECTORY is not None:",
            "Set RUN_DIRECTORY=None when RUN_SIMULATION=True.",
            "if not GENERATED_INPUT_PATH.is_file():",
            "FileNotFoundError",
            "elif RUN_DIRECTORY is None:",
            "Set RUN_DIRECTORY to an explicit completed run when RUN_SIMULATION=False.",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, selection_source)

        run_calls = self.calls_named("run_input_file")
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0][0], selection_index)
        run_call = run_calls[0][1]
        self.assertEqual(ast.unparse(run_call.args[0]), "GENERATED_INPUT_PATH")
        run_keywords = keyword_map(run_call)
        self.assertEqual(
            {
                name: ast.unparse(run_keywords[name])
                for name in (
                    "output_root",
                    "tag",
                    "add_timestamp",
                    "show_log",
                    "convergenceCheck",
                    "staging_root",
                    "keep_staged_input",
                )
            },
            {
                "output_root": "SIMULATION_OUTPUT_ROOT",
                "tag": "RUN_TAG",
                "add_timestamp": "True",
                "show_log": "True",
                "convergenceCheck": "False",
                "staging_root": "SIMULATION_STAGING_ROOT",
                "keep_staged_input": "True",
            },
        )
        self.assertNotIn("outputdirectory", run_keywords)

        executed_assignments = [
            node
            for node in ast.walk(selection)
            if assignment_name(node) == "executed_input"
        ]
        self.assertEqual(len(executed_assignments), 1)
        self.assertIs(
            assignment_value(executed_assignments[0]),
            run_call,
        )

        get_output_assignments = [
            node
            for node in ast.walk(selection)
            if assignment_name(node) == "RUN_DIRECTORY"
            and isinstance(assignment_value(node), ast.Call)
            and call_name(assignment_value(node)) == "get_output_directory"
        ]
        self.assertEqual(len(get_output_assignments), 1)
        self.assertEqual(
            ast.unparse(assignment_value(get_output_assignments[0])),
            "get_output_directory(executed_input)",
        )

        validator_assignments = [
            (index, node)
            for index, tree in self.trees.items()
            for node in tree.body
            if assignment_name(node) == "RUN_DIRECTORY"
            and isinstance(assignment_value(node), ast.Call)
            and call_name(assignment_value(node)) == "validate_run_directory"
        ]
        self.assertEqual(len(validator_assignments), 1)
        self.assertEqual(validator_assignments[0][0], selection_index)
        validator_call = assignment_value(validator_assignments[0][1])
        self.assertEqual(ast.unparse(validator_call.args[0]), "RUN_DIRECTORY")
        validator_keywords = keyword_map(validator_call)
        self.assertEqual(ast.unparse(validator_keywords["bias"]), "BIAS")
        self.assertIs(
            ast.literal_eval(validator_keywords["require_complete"]),
            True,
        )
        self.assertEqual(
            ast.unparse(validator_keywords["required_outputs"]),
            "REQUIRED_OUTPUTS",
        )

        _, bias_directory = self.one_top_assignment("BIAS_DIRECTORY")
        self.assertEqual(
            ast.unparse(assignment_value(bias_directory)),
            "get_bias_dir(RUN_DIRECTORY, BIAS)",
        )
        _, structure_directory = self.one_top_assignment("STRUCTURE_DIRECTORY")
        self.assertEqual(
            ast.unparse(assignment_value(structure_directory)),
            "RUN_DIRECTORY / 'Structure'",
        )

    def test_no_implicit_or_competing_run_selection_remains(self):
        banned_fragments = (
            "find_latest_run",
            "find_runs_for_input",
            "st_mtime",
            "globals()",
            "matching_runs",
            "tagged_runs",
            "default_run_name",
        )
        for fragment in banned_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.combined_code)

        banned_identifiers = {
            "RUNS_DIR",
            "OUTPUT_RUN_DIR",
            "RUN_DIR",
            "STRUCTURE_DIR",
            "BIAS_INDEX",
            "selected_run",
            "run_root",
        }
        identifiers = {
            node.id
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        self.assertTrue(banned_identifiers.isdisjoint(identifiers))
        for identifier in banned_identifiers:
            with self.subTest(identifier=identifier):
                self.assertIsNone(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(identifier)}"
                        rf"(?![A-Za-z0-9_])",
                        self.combined_source,
                    )
                )

        stores = [
            node
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == "RUN_DIRECTORY"
        ]
        self.assertEqual(len(stores), 3)

    def test_downstream_analysis_uses_the_canonical_run_and_named_bias(self):
        run_root_consumers = {
            "resolve_bias_output_file",
            "plot_bias_volume_3d",
            "plot_bias_volume_slice",
            "resolve_quantum_output_file",
            "resolve_quantum_probability_state_file",
            "plot_quantum_density_volume_3d",
            "plot_quantum_density_volume_linecut",
            "plot_quantum_density_volume_slice",
            "plot_quantum_probability_volume_3d",
            "plot_quantum_probability_volume_linecut",
            "plot_quantum_probability_volume_slice",
        }
        for name in run_root_consumers:
            calls = self.calls_named(name)
            self.assertTrue(calls, f"Expected at least one call to {name}")
            for index, call in calls:
                with self.subTest(name=name, cell=index):
                    self.assertTrue(call.args)
                    self.assertEqual(
                        ast.unparse(call.args[0]),
                        "RUN_DIRECTORY",
                    )

        self.assertIn("STRUCTURE_DIRECTORY", self.combined_code)
        self.assertNotIn("BIAS_INDEX", self.combined_code)
        bias_keywords = [
            keyword.value
            for _, call in self.calls
            for keyword in call.keywords
            if keyword.arg == "bias"
            and call_name(call) in run_root_consumers
        ]
        self.assertTrue(bias_keywords)
        self.assertTrue(
            all(ast.unparse(value) in {"BIAS", "bias"} for value in bias_keywords)
        )

    def test_generation_paths_geometry_and_execution_arguments_are_preserved(self):
        expected_path_fragments = {
            "OUTPUT_DIR": "REPO_ROOT / 'data' / 'gds'",
            "GENERATED_INPUT_DIR": (
                "REPO_ROOT / 'configs' / 'robert_inputs' / 'generated'"
            ),
            "TEMPLATE_INPUT_PATH": (
                "REPO_ROOT / 'configs' / 'robert_inputs' / 'double_qd' / "
                "'3d' / 'Double_Quantum_Dot_3D.in'"
            ),
            "GENERATED_INPUT_PATH": (
                "GENERATED_INPUT_DIR / 'Double_Quantum_Dot_3D_from_PHIDL.in'"
            ),
        }
        for name, expected in expected_path_fragments.items():
            with self.subTest(name=name):
                _, node = self.one_top_assignment(name)
                self.assertEqual(ast.unparse(assignment_value(node)), expected)

        for filename in (
            "double_dot_from_phidl.gds",
            "double_dot_from_phidl.svg",
            "double_dot_from_phidl_layout_spec.json",
            "double_dot_from_phidl_simulation_layout.json",
        ):
            with self.subTest(filename=filename):
                self.assertIn(filename, self.combined_code)

        write_calls = self.calls_named("write_nextnano_input_from_template")
        self.assertEqual(len(write_calls), 1)
        write_keywords = keyword_map(write_calls[0][1])
        self.assertEqual(
            ast.unparse(write_keywords["simulation_layout"]),
            "simulation_layout",
        )
        self.assertEqual(
            ast.unparse(write_keywords["template_path"]),
            "TEMPLATE_INPUT_PATH",
        )
        self.assertEqual(
            ast.unparse(write_keywords["output_path"]),
            "GENERATED_INPUT_PATH",
        )
        self.assertEqual(
            ast.literal_eval(write_keywords["voltage_overrides"]),
            {
                "V_P1": -3.0,
                "V_P2": -3.0,
                "V_B1": 0.0,
                "V_B2": 0.0,
                "V_B3": 0.0,
                "V_OC_L": 0.0,
                "V_OC_R": 0.0,
            },
        )

        builder_calls = self.calls_named("LinearDotArrayDevice")
        self.assertEqual(len(builder_calls), 1)
        builder_keywords = {
            name: ast.literal_eval(value)
            for name, value in keyword_map(builder_calls[0][1]).items()
        }
        self.assertEqual(
            builder_keywords,
            {
                "name": "double_dot_from_phidl",
                "n_dots": 2,
                "device_y_size_nm": 200.0,
                "ohmic_width_nm": 40.0,
                "ohmic_length_nm": 200.0,
                "barrier_width_nm": 40.0,
                "barrier_length_nm": 140.0,
                "plunger_body_width_nm": 40.0,
                "plunger_body_length_nm": 50.0,
                "plunger_head_top_width_nm": 60.0,
                "plunger_head_max_width_nm": 100.0,
                "plunger_head_height_nm": 100.0,
                "plunger_upper_taper_height_nm": 25.0,
                "plunger_lower_taper_height_nm": 25.0,
                "ohmic_to_barrier_gap_nm": 20.0,
                "barrier_to_plunger_gap_nm": 20.0,
            },
        )

    def test_geometry_and_quantum_analysis_are_preserved(self):
        expected_call_counts = {
            "LinearDotArrayDevice": 1,
            "make_reference_sige_ge_process_stack": 1,
            "build_simulation_layout": 1,
            "write_nextnano_input_from_template": 1,
            "resolve_quantum_output_file": 1,
            "resolve_quantum_probability_state_file": 1,
            "plot_quantum_density_volume_3d": 1,
            "plot_quantum_density_volume_slice": 1,
            "plot_quantum_density_volume_linecut": 1,
            "plot_quantum_probability_volume_3d": 1,
            "plot_quantum_probability_volume_slice": 1,
            "plot_quantum_probability_volume_linecut": 1,
        }
        for name, expected_count in expected_call_counts.items():
            with self.subTest(name=name):
                self.assertEqual(len(self.calls_named(name)), expected_count)

        expected_helpers = {
            "print_json",
            "print_header",
        }
        defined_helpers = {
            node.name
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(defined_helpers, expected_helpers)

        preserved_assignments = {
            "QUANTUM_REGION": "c-Ge_QW",
            "QUANTUM_BAND": "HH",
            "QUANTUM_KPOINT": "k00000",
            "QUANTUM_STATE": 1,
        }
        for name, expected in preserved_assignments.items():
            with self.subTest(name=name):
                _, node = self.one_top_assignment(name)
                self.assertEqual(
                    ast.literal_eval(assignment_value(node)),
                    expected,
                )

        _, default_axis = self.one_top_assignment("DEFAULT_1D_LINE_AXIS")
        self.assertEqual(
            ast.unparse(assignment_value(default_axis)),
            "LINE_AXIS",
        )
        _, default_fixed_coords = self.one_top_assignment(
            "DEFAULT_1D_LINE_FIXED_COORDS"
        )
        self.assertEqual(
            ast.unparse(assignment_value(default_fixed_coords)),
            "dict(LINE_FIXED_COORDINATES)",
        )

    def test_notebook_is_path_safe_and_has_no_stored_outputs(self):
        self.assertNotIn("/Users/robertjovanov/", self.combined_source)
        self.assertIsNone(re.search(r"[A-Za-z]:\\Users\\", self.combined_source))
        self.assertNotIn("../../runs", self.combined_source)

        for index, cell in self.code_cells:
            with self.subTest(cell=index):
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                self.assertTrue(cell_source(cell).strip())
        for index, cell in enumerate(self.cells):
            with self.subTest(attachments=index):
                self.assertFalse(cell.get("attachments"))
        self.assertFalse(
            self.cells[-1]["cell_type"] == "code"
            and not cell_source(self.cells[-1]).strip()
        )


if __name__ == "__main__":
    unittest.main()
