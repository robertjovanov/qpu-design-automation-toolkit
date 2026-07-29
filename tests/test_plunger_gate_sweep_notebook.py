import ast
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT / "notebooks" / "analysis" / "plunger_gate_sweep.ipynb"
)
EXPECTED_PLUNGER_VOLTAGES = [
    0.0,
    -0.2,
    -0.4,
    -0.6,
    -0.8,
    -1.0,
    -1.2,
    -1.4,
    -1.6,
    -1.8,
    -2.0,
    -2.2,
    -2.4,
    -2.6,
    -2.8,
    -3.0,
    -6.0,
    -10.0,
    -25.0,
    -50.0,
]


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


def contains_call(node, name):
    return any(
        isinstance(child, ast.Call) and call_name(child) == name
        for child in ast.walk(node)
    )


class PlungerGateSweepNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
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

        cls.assignments = {}
        for index, tree in cls.trees.items():
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    cls.assignments.setdefault(target.id, []).append((index, node.value))

        cls.calls = []
        for index, tree in cls.trees.items():
            cls.calls.extend(
                (index, node)
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
            )

    def assignment_value(self, name):
        matches = self.assignments.get(name, [])
        self.assertEqual(len(matches), 1, f"Expected one assignment to {name}")
        return ast.literal_eval(matches[0][1])

    def calls_named(self, name):
        return [
            (index, call)
            for index, call in self.calls
            if call_name(call) == name
        ]

    def test_no_legacy_code_or_local_implementations(self):
        combined_code = "\n".join(self.code_sources.values())
        banned_fragments = (
            "sys.path",
            "importlib.reload",
            "nextnano_analysis",
            "make_xqd_potential_gif",
            "parse_vpg",
            "imageio",
            ".gif",
            "mimsave",
            "imwrite",
            "savefig",
            "write_gif",
            "get_writer",
            "NextnanoRun",
            "integrated_density",
        )
        for fragment in banned_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, combined_code)

        forbidden_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        for index, tree in self.trees.items():
            for node in ast.walk(tree):
                self.assertNotIsInstance(
                    node,
                    forbidden_nodes,
                    f"Reusable implementation found in code cell {index}",
                )

    def test_imports_and_public_api_calls(self):
        imported_by_module = {}
        for tree in self.trees.values():
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    imported_by_module.setdefault(node.module, set()).update(
                        alias.name for alias in node.names
                    )

        self.assertIn("Path", imported_by_module.get("pathlib", set()))
        required_api = {
            "run_sweep",
            "discover_sweep_runs",
            "load_sweep_outputs",
            "plot_sweep_lines",
        }
        self.assertTrue(
            required_api.issubset(imported_by_module.get("nextnanopp_tools", set()))
        )

        expected_call_counts = {
            "run_sweep": 1,
            "discover_sweep_runs": 1,
            "load_sweep_outputs": 1,
            "plot_sweep_lines": 2,
        }
        for name, expected_count in expected_call_counts.items():
            with self.subTest(name=name):
                self.assertEqual(len(self.calls_named(name)), expected_count)
        self.assertFalse(self.calls_named("print"))

    def test_configuration_is_explicit_and_disabled_by_default(self):
        self.assertIs(self.assignment_value("RUN_SWEEP"), False)
        self.assertIs(self.assignment_value("SHOW_INTERACTIVE"), False)
        self.assertEqual(self.assignment_value("SWEEP_VARIABLE"), "V_PG")
        self.assertEqual(self.assignment_value("SELECTED_BIAS"), "bias_00000")
        self.assertEqual(
            ast.unparse(self.assignments["SELECTED_OUTPUT"][0][1]),
            "Path('potential_1d_x_QD.dat')",
        )

        configuration_names = {
            "START_PATH",
            "REPO_ROOT",
            "RUN_SWEEP",
            "SHOW_INTERACTIVE",
            "SWEEP_VARIABLE",
            "PLUNGER_VOLTAGES",
            "SIMULATION_OUTPUT_ROOT",
            "EMPTY_QD_INPUT",
            "ACCUMULATED_QD_INPUT",
            "EMPTY_QD_SWEEP_ROOT",
            "ACCUMULATED_QD_SWEEP_ROOT",
            "SELECTED_BIAS",
            "SELECTED_OUTPUT",
            "SELECTED_VARIABLE",
            "REFERENCE_COORD",
            "X_LIMITS",
            "Y_LIMITS",
            "DELETE_OLD_FILES",
            "DELETE_INPUT_FILES",
            "OVERWRITE",
        }
        configuration_cells = {
            self.assignments[name][0][0] for name in configuration_names
        }
        self.assertEqual(len(configuration_cells), 1)
        self.assertIs(self.assignment_value("SELECTED_VARIABLE"), None)
        self.assertIs(self.assignment_value("REFERENCE_COORD"), None)
        self.assertEqual(self.assignment_value("X_LIMITS"), (-200.0, 200.0))
        self.assertIs(self.assignment_value("Y_LIMITS"), None)
        self.assertIs(self.assignment_value("DELETE_OLD_FILES"), True)
        self.assertIs(self.assignment_value("DELETE_INPUT_FILES"), False)
        self.assertIs(self.assignment_value("OVERWRITE"), True)

    def test_repository_root_searches_current_directory_and_parents(self):
        self.assertEqual(
            ast.unparse(self.assignments["START_PATH"][0][1]),
            "Path.cwd().resolve()",
        )

        repository_root_expression = self.assignments["REPO_ROOT"][0][1]
        self.assertIsInstance(repository_root_expression, ast.Call)
        self.assertIsInstance(repository_root_expression.func, ast.Name)
        self.assertEqual(repository_root_expression.func.id, "next")
        self.assertEqual(len(repository_root_expression.args), 2)
        self.assertIsInstance(repository_root_expression.args[0], ast.GeneratorExp)
        self.assertIsInstance(repository_root_expression.args[1], ast.Constant)
        self.assertIsNone(repository_root_expression.args[1].value)

        search_source = ast.unparse(repository_root_expression)
        for required_fragment in (
            "(START_PATH, *START_PATH.parents)",
            "'pyproject.toml'",
            "'src'",
            ".is_file()",
            ".is_dir()",
        ):
            with self.subTest(required_fragment=required_fragment):
                self.assertIn(required_fragment, search_source)

        failure_guards = [
            node
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "REPO_ROOT is None"
        ]
        self.assertEqual(len(failure_guards), 1)
        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(failure_guards[0]))
        )
        self.assertIn(
            "Could not locate the repository root",
            ast.unparse(failure_guards[0]),
        )

    def test_simulation_output_root_is_single_external_configuration(self):
        output_root_assignments = self.assignments.get("SIMULATION_OUTPUT_ROOT", [])
        self.assertEqual(len(output_root_assignments), 1)
        configured_root_names = [
            name
            for name in self.assignments
            if name.isupper() and name.endswith("OUTPUT_ROOT")
        ]
        self.assertEqual(configured_root_names, ["SIMULATION_OUTPUT_ROOT"])

        output_root_source = ast.unparse(output_root_assignments[0][1])
        for required_fragment in (
            "os.environ.get('NEXTNANO_OUTPUT_ROOT'",
            "REPO_ROOT.parent / 'qpu-local-outputs' / 'refactoring' / 'runs'",
            ".expanduser()",
            ".resolve()",
        ):
            with self.subTest(required_fragment=required_fragment):
                self.assertIn(required_fragment, output_root_source)

        external_guards = [
            node
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "SIMULATION_OUTPUT_ROOT == REPO_ROOT" in ast.unparse(node.test)
            and "REPO_ROOT in SIMULATION_OUTPUT_ROOT.parents" in ast.unparse(node.test)
        ]
        self.assertEqual(len(external_guards), 1)
        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(external_guards[0]))
        )

        expected_sweep_expressions = {
            "EMPTY_QD_SWEEP_ROOT": (
                "SIMULATION_OUTPUT_ROOT / "
                "f'{EMPTY_QD_INPUT.stem}_sweep__{SWEEP_VARIABLE}'"
            ),
            "ACCUMULATED_QD_SWEEP_ROOT": (
                "SIMULATION_OUTPUT_ROOT / "
                "f'{ACCUMULATED_QD_INPUT.stem}_sweep__{SWEEP_VARIABLE}'"
            ),
        }
        for name, expected_source in expected_sweep_expressions.items():
            with self.subTest(name=name):
                actual = self.assignments[name][0][1]
                expected = ast.parse(expected_source, mode="eval").body
                self.assertEqual(
                    ast.dump(actual, include_attributes=False),
                    ast.dump(expected, include_attributes=False),
                )

        sweep_variable = self.assignment_value("SWEEP_VARIABLE")
        derived_names = {
            "empty": (
                f"{Path(self.assignments['EMPTY_QD_INPUT'][0][1].right.value).stem}"
                f"_sweep__{sweep_variable}"
            ),
            "accumulated": (
                f"{Path(self.assignments['ACCUMULATED_QD_INPUT'][0][1].right.value).stem}"
                f"_sweep__{sweep_variable}"
            ),
        }
        self.assertEqual(
            derived_names,
            {
                "empty": "Single_Empty_Quantum_Dot_2D_sweep__V_PG",
                "accumulated": "Single_Quantum_Dot_2D_sweep__V_PG",
            },
        )

    def test_exact_voltage_sequence_is_preserved_and_used(self):
        self.assertEqual(
            self.assignment_value("PLUNGER_VOLTAGES"),
            EXPECTED_PLUNGER_VOLTAGES,
        )
        self.assertEqual(len(set(EXPECTED_PLUNGER_VOLTAGES)), 20)

        _, run_call = self.calls_named("run_sweep")[0]
        self.assertGreaterEqual(len(run_call.args), 2)
        sweep_mapping = run_call.args[1]
        self.assertIsInstance(sweep_mapping, ast.Dict)
        self.assertEqual(len(sweep_mapping.keys), 1)
        self.assertIsInstance(sweep_mapping.keys[0], ast.Name)
        self.assertEqual(sweep_mapping.keys[0].id, "SWEEP_VARIABLE")
        self.assertIsInstance(sweep_mapping.values[0], ast.Name)
        self.assertEqual(sweep_mapping.values[0].id, "PLUNGER_VOLTAGES")

    def test_cases_share_one_mapping_and_execution_loop(self):
        case_assignments = self.assignments.get("CASES", [])
        self.assertEqual(len(case_assignments), 1)
        cases_node = case_assignments[0][1]
        self.assertIsInstance(cases_node, ast.Dict)
        case_keys = [ast.literal_eval(key) for key in cases_node.keys]
        self.assertEqual(case_keys, ["empty", "accumulated"])

        case_fields = {}
        case_labels = {}
        case_references = {}
        for key, value in zip(case_keys, cases_node.values, strict=True):
            self.assertIsInstance(value, ast.Dict)
            fields = [ast.literal_eval(field) for field in value.keys]
            case_fields[key] = set(fields)
            label_index = fields.index("label")
            case_labels[key] = ast.literal_eval(value.values[label_index])
            case_references[key] = {
                field: ast.unparse(value.values[fields.index(field)])
                for field in ("input_path", "sweep_root")
            }
        required_fields = {"input_path", "sweep_root", "label"}
        self.assertEqual(case_fields["empty"], required_fields)
        self.assertEqual(case_fields["accumulated"], required_fields)
        self.assertIn("Empty", case_labels["empty"])
        self.assertIn("Accumulated", case_labels["accumulated"])
        self.assertEqual(
            case_references,
            {
                "empty": {
                    "input_path": "EMPTY_QD_INPUT",
                    "sweep_root": "EMPTY_QD_SWEEP_ROOT",
                },
                "accumulated": {
                    "input_path": "ACCUMULATED_QD_INPUT",
                    "sweep_root": "ACCUMULATED_QD_SWEEP_ROOT",
                },
            },
        )

        configured_input_paths = {
            "EMPTY_QD_INPUT": (
                "configs/nextnano_inputs/baseline_single_qd/2d/"
                "Single_Empty_Quantum_Dot_2D.in"
            ),
            "ACCUMULATED_QD_INPUT": (
                "configs/nextnano_inputs/baseline_single_qd/2d/"
                "Single_Quantum_Dot_2D.in"
            ),
        }
        for name, expected_path in configured_input_paths.items():
            with self.subTest(name=name):
                expression = self.assignments[name][0][1]
                self.assertIsInstance(expression, ast.BinOp)
                self.assertIsInstance(expression.op, ast.Div)
                self.assertIsInstance(expression.left, ast.Name)
                self.assertEqual(expression.left.id, "REPO_ROOT")
                self.assertIsInstance(expression.right, ast.Constant)
                self.assertEqual(expression.right.value, expected_path)

        run_guards = [
            node
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "RUN_SWEEP"
        ]
        self.assertEqual(len(run_guards), 1)
        self.assertTrue(contains_call(run_guards[0], "run_sweep"))
        shared_loops = [
            node
            for node in ast.walk(run_guards[0])
            if isinstance(node, ast.For) and contains_call(node, "run_sweep")
        ]
        self.assertEqual(len(shared_loops), 1)
        self.assertIn("CASES.items()", ast.unparse(shared_loops[0].iter))

    def test_sweep_api_arguments_are_explicit(self):
        _, run_call = self.calls_named("run_sweep")[0]
        run_keywords = keyword_map(run_call)
        for keyword in (
            "delete_old_files",
            "delete_input_files",
            "overwrite",
            "show_log",
            "convergenceCheck",
            "parallel_limit",
            "execute_kwargs",
        ):
            self.assertIn(keyword, run_keywords)
        self.assertEqual(
            ast.unparse(run_call.args[0]),
            "case['input_path']",
        )
        self.assertEqual(
            ast.unparse(run_keywords["execute_kwargs"]),
            "{'outputdirectory': str(case['sweep_root'].parent)}",
        )
        self.assertIs(self.assignment_value("SHOW_LOG"), False)

        _, discover_call = self.calls_named("discover_sweep_runs")[0]
        discover_keywords = keyword_map(discover_call)
        self.assertIsInstance(discover_call.args[1], ast.Name)
        self.assertEqual(discover_call.args[1].id, "SWEEP_VARIABLE")
        self.assertEqual(
            {
                "bias",
                "required_outputs",
                "strict",
            },
            set(discover_keywords),
        )
        self.assertEqual(
            ast.unparse(discover_keywords["bias"]),
            "SELECTED_BIAS",
        )
        self.assertEqual(
            ast.unparse(discover_keywords["required_outputs"]),
            "(SELECTED_OUTPUT,)",
        )
        self.assertEqual(
            ast.unparse(discover_keywords["strict"]),
            "STRICT_DISCOVERY",
        )
        self.assertIs(self.assignment_value("STRICT_DISCOVERY"), True)

        _, load_call = self.calls_named("load_sweep_outputs")[0]
        load_keywords = keyword_map(load_call)
        self.assertEqual(set(load_keywords), {"variable"})
        self.assertEqual(ast.unparse(load_call.args[1]), "SELECTED_OUTPUT")
        self.assertEqual(
            ast.unparse(load_keywords["variable"]),
            "SELECTED_VARIABLE",
        )

        plot_calls = [call for _, call in self.calls_named("plot_sweep_lines")]
        interactive_values = []
        for call in plot_calls:
            keywords = keyword_map(call)
            self.assertTrue(
                {
                    "variable",
                    "interactive",
                    "title",
                    "xlim",
                    "ylim",
                    "reference_coord",
                }.issubset(keywords)
            )
            self.assertNotIn("values", keywords)
            interactive_values.append(ast.literal_eval(keywords["interactive"]))
        self.assertCountEqual(interactive_values, [False, True])

        interactive_guards = [
            node
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "SHOW_INTERACTIVE"
        ]
        self.assertEqual(len(interactive_guards), 1)
        self.assertTrue(contains_call(interactive_guards[0], "plot_sweep_lines"))

    def test_optional_execution_validates_paths_without_changing_configuration(self):
        run_guards = [
            node
            for tree in self.trees.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "RUN_SWEEP"
        ]
        self.assertEqual(len(run_guards), 1)
        guard = run_guards[0]
        guard_source = ast.unparse(guard)
        for required_fragment in (
            "CASES.values()",
            "case['input_path'].is_file()",
            "FileNotFoundError",
            "case['sweep_root'].parent.expanduser().resolve()",
            "!= SIMULATION_OUTPUT_ROOT",
            "Every case sweep root must share SIMULATION_OUTPUT_ROOT",
            "sweep.sweep_output_directory",
            "actual_sweep_root != expected_sweep_root",
            "SWEEP_RESULTS[case_name] = sweep",
        ):
            with self.subTest(required_fragment=required_fragment):
                self.assertIn(required_fragment, guard_source)

        run_call = self.calls_named("run_sweep")[0][1]
        input_checks = [
            node
            for node in ast.walk(guard)
            if isinstance(node, ast.Call)
            and call_name(node) == "is_file"
        ]
        self.assertEqual(len(input_checks), 1)
        self.assertLess(input_checks[0].lineno, run_call.lineno)

        forbidden_config_calls = {
            "configure_nextnano",
            "save",
            "set",
            "to_default",
        }
        present_call_names = {
            call_name(node)
            for node in ast.walk(guard)
            if isinstance(node, ast.Call)
        }
        self.assertTrue(forbidden_config_calls.isdisjoint(present_call_names))
        combined_code = "\n".join(self.code_sources.values())
        self.assertNotIn("nn.config", combined_code)
        self.assertNotIn("import nextnanopy", combined_code)

    def test_discovery_loading_and_plotting_are_case_driven(self):
        manifests_node = self.assignments["MANIFESTS"][0][1]
        self.assertIsInstance(manifests_node, ast.DictComp)
        self.assertEqual(
            ast.unparse(manifests_node.generators[0].iter),
            "CASES.items()",
        )
        manifest_discovery_calls = [
            node
            for node in ast.walk(manifests_node)
            if isinstance(node, ast.Call)
            and call_name(node) == "discover_sweep_runs"
        ]
        self.assertEqual(len(manifest_discovery_calls), 1)
        discovery_root = manifest_discovery_calls[0].args[0]
        self.assertIsInstance(discovery_root, ast.Subscript)
        self.assertEqual(ast.unparse(discovery_root), "case['sweep_root']")

        loaded_outputs_node = self.assignments["LOADED_OUTPUTS"][0][1]
        self.assertIsInstance(loaded_outputs_node, ast.DictComp)
        self.assertEqual(
            ast.unparse(loaded_outputs_node.generators[0].iter),
            "CASES",
        )
        loading_calls = [
            node
            for node in ast.walk(loaded_outputs_node)
            if isinstance(node, ast.Call)
            and call_name(node) == "load_sweep_outputs"
        ]
        self.assertEqual(len(loading_calls), 1)
        self.assertEqual(
            ast.unparse(loading_calls[0].args[0]),
            "MANIFESTS[case_name]",
        )

        for index, plot_call in self.calls_named("plot_sweep_lines"):
            self.assertEqual(
                ast.unparse(plot_call.args[0]),
                "LOADED_OUTPUTS[case_name]",
            )
            containing_loops = [
                node
                for node in ast.walk(self.trees[index])
                if isinstance(node, ast.For)
                and any(child is plot_call for child in ast.walk(node))
            ]
            self.assertEqual(len(containing_loops), 1)
            self.assertEqual(
                ast.unparse(containing_loops[0].iter),
                "CASES.items()",
            )

    def test_discovery_summary_and_validation_are_concise(self):
        self.assertEqual(
            self.assignment_value("DISCOVERY_COLUMNS"),
            [
                "sweep_value",
                "run_name",
                "complete",
                "bias_dir",
                "outputs_available",
                "error",
            ],
        )
        self.assertTrue(self.calls_named("display"))

        combined_code = "\n".join(self.code_sources.values())
        for required_check in (
            "manifest.empty",
            'manifest["complete"]',
            'manifest["outputs_available"]',
            'outputs["load_error"]',
        ):
            with self.subTest(required_check=required_check):
                self.assertIn(required_check, combined_code)

    def test_notebook_metadata_paths_and_markdown_are_clean(self):
        for index, cell in self.code_cells:
            self.assertIsNone(cell.get("execution_count"), f"Code cell {index}")
            self.assertEqual(cell.get("outputs"), [], f"Code cell {index}")
            self.assertTrue(cell_source(cell).strip(), f"Empty code cell {index}")

        self.assertTrue(self.cells)
        self.assertFalse(
            self.cells[-1]["cell_type"] == "code"
            and not cell_source(self.cells[-1]).strip()
        )

        combined_source = "\n".join(cell_source(cell) for cell in self.cells)
        self.assertNotIn(
            "/Users/robertjovanov/code/qpu-design-automation-toolkit",
            combined_source,
        )
        self.assertNotIn("sys.path", combined_source)

        repository_path_literals = []
        input_paths = []
        repository_run_literals = []
        for tree in self.trees.values():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if (
                    "qpu-design-automation-toolkit" in node.value
                    and Path(node.value).is_absolute()
                ):
                    repository_path_literals.append(node.value)
                if node.value.endswith(".in"):
                    input_paths.append(node.value)
                if node.value.startswith("runs/"):
                    repository_run_literals.append(node.value)
        self.assertEqual(repository_path_literals, [])
        self.assertEqual(repository_run_literals, [])
        self.assertEqual(len(input_paths), 2)
        for input_path in input_paths:
            self.assertFalse(Path(input_path).is_absolute())
            self.assertTrue(input_path.startswith("configs/"))

        markdown = "\n".join(
            cell_source(cell)
            for cell in self.cells
            if cell["cell_type"] == "markdown"
        ).lower()
        for phrase in (
            "plunger-gate voltage sweep",
            "empty quantum dot",
            "accumulated",
            "optional",
            "disabled by default",
            "potential_1d_x_qd.dat",
            "hh band-edge",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)
