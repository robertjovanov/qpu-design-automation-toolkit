import ast
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT / "notebooks" / "analysis" / "barrier_gate_sweep.ipynb"
)
EXPECTED_BARRIER_VOLTAGES = [
    -2.0,
    -1.8,
    -1.6,
    -1.4,
    -1.2,
    -1.0,
    -0.8,
    -0.6,
    -0.4,
    -0.2,
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
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


class BarrierGateSweepNotebookTests(unittest.TestCase):
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

        cls.assignments = {}
        for index, tree in cls.trees.items():
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    cls.assignments.setdefault(target.id, []).append((index, node.value))

        cls.calls = [
            (index, node)
            for index, tree in cls.trees.items()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]

    def assignment_node(self, name):
        matches = self.assignments.get(name, [])
        self.assertEqual(len(matches), 1, f"Expected one assignment to {name}")
        return matches[0][1]

    def assignment_value(self, name):
        return ast.literal_eval(self.assignment_node(name))

    def calls_named(self, name):
        return [
            (index, call)
            for index, call in self.calls
            if call_name(call) == name
        ]

    def test_no_legacy_code_media_or_local_implementations(self):
        combined_code = "\n".join(self.code_sources.values())
        banned_fragments = (
            "sys.path",
            "importlib.reload",
            "nextnano_analysis",
            "make_xqd_potential_gif",
            "parse_vbg",
            "_parse_vbg",
            "imageio",
            ".gif",
            "mimsave",
            "imwrite",
            "savefig",
            "write_gif",
            "get_writer",
            "SWEEP_5_TO_0_ROOT",
            "OUT_GIF_5_TO_0",
            "NextnanoRun",
            "integrated_density",
            "V_PG",
            "PLUNGER_VOLTAGES",
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

        for stale_mime in ("image/gif", "image/png", "application/javascript"):
            with self.subTest(stale_mime=stale_mime):
                self.assertNotIn(stale_mime, self.raw_notebook)

    def test_installed_imports_and_public_api_calls(self):
        imported_by_module = {}
        direct_imports = set()
        for tree in self.trees.values():
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    imported_by_module.setdefault(node.module, set()).update(
                        alias.name for alias in node.names
                    )
                elif isinstance(node, ast.Import):
                    direct_imports.update(alias.name for alias in node.names)

        self.assertIn("os", direct_imports)
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

    def test_barrier_configuration_and_exact_voltages(self):
        self.assertIs(self.assignment_value("RUN_SWEEP"), False)
        self.assertIs(self.assignment_value("SHOW_INTERACTIVE"), False)
        self.assertEqual(self.assignment_value("SWEEP_VARIABLE"), "V_BG")
        self.assertEqual(
            self.assignment_value("BARRIER_VOLTAGES"),
            EXPECTED_BARRIER_VOLTAGES,
        )
        self.assertEqual(len(EXPECTED_BARRIER_VOLTAGES), 25)
        self.assertEqual(len(set(EXPECTED_BARRIER_VOLTAGES)), 25)

        self.assertEqual(self.assignment_value("SELECTED_BIAS"), "bias_00000")
        self.assertEqual(
            ast.unparse(self.assignment_node("SELECTED_OUTPUT")),
            "Path('potential_1d_x_QD.dat')",
        )
        self.assertIs(self.assignment_value("SELECTED_VARIABLE"), None)
        self.assertIs(self.assignment_value("REFERENCE_COORD"), None)
        self.assertEqual(self.assignment_value("X_LIMITS"), (-200.0, 200.0))
        self.assertIs(self.assignment_value("Y_LIMITS"), None)
        self.assertIs(self.assignment_value("DELETE_OLD_FILES"), True)
        self.assertIs(self.assignment_value("DELETE_INPUT_FILES"), False)
        self.assertIs(self.assignment_value("OVERWRITE"), True)

        _, run_call = self.calls_named("run_sweep")[0]
        self.assertEqual(ast.unparse(run_call.args[0]), "case['input_path']")
        self.assertEqual(
            ast.unparse(run_call.args[1]),
            "{SWEEP_VARIABLE: BARRIER_VOLTAGES}",
        )

    def test_repository_root_searches_current_directory_and_parents(self):
        self.assertEqual(
            ast.unparse(self.assignment_node("START_PATH")),
            "Path.cwd().resolve()",
        )
        root_expression = self.assignment_node("REPO_ROOT")
        self.assertIsInstance(root_expression, ast.Call)
        self.assertIsInstance(root_expression.func, ast.Name)
        self.assertEqual(root_expression.func.id, "next")
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
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "REPO_ROOT is None"
        ]
        self.assertEqual(len(guards), 1)
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(guards[0])))
        self.assertIn("Could not locate the repository root", ast.unparse(guards[0]))

    def test_single_external_output_root_and_native_sweep_names(self):
        output_root_matches = self.assignments.get("SIMULATION_OUTPUT_ROOT", [])
        self.assertEqual(len(output_root_matches), 1)
        uppercase_output_roots = [
            name
            for name in self.assignments
            if name.isupper() and name.endswith("OUTPUT_ROOT")
        ]
        self.assertEqual(uppercase_output_roots, ["SIMULATION_OUTPUT_ROOT"])

        output_root_source = ast.unparse(output_root_matches[0][1])
        for fragment in (
            "os.environ.get('NEXTNANO_OUTPUT_ROOT'",
            "REPO_ROOT.parent / 'qpu-local-outputs' / 'refactoring' / 'runs'",
            ".expanduser()",
            ".resolve()",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, output_root_source)

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

        expected_expressions = {
            "EMPTY_QD_SWEEP_ROOT": (
                "SIMULATION_OUTPUT_ROOT / "
                "f'{EMPTY_QD_INPUT.stem}_sweep__{SWEEP_VARIABLE}'"
            ),
            "ACCUMULATED_QD_SWEEP_ROOT": (
                "SIMULATION_OUTPUT_ROOT / "
                "f'{ACCUMULATED_QD_INPUT.stem}_sweep__{SWEEP_VARIABLE}'"
            ),
        }
        for name, expected_source in expected_expressions.items():
            with self.subTest(name=name):
                expected = ast.parse(expected_source, mode="eval").body
                self.assertEqual(
                    ast.dump(self.assignment_node(name), include_attributes=False),
                    ast.dump(expected, include_attributes=False),
                )

        input_paths = {
            "EMPTY_QD_INPUT": (
                "configs/nextnano_inputs/baseline_single_qd/2d/"
                "Single_Empty_Quantum_Dot_2D.in"
            ),
            "ACCUMULATED_QD_INPUT": (
                "configs/nextnano_inputs/baseline_single_qd/2d/"
                "Single_Quantum_Dot_2D.in"
            ),
        }
        for name, expected_path in input_paths.items():
            with self.subTest(name=name):
                expression = self.assignment_node(name)
                self.assertEqual(ast.unparse(expression.left), "REPO_ROOT")
                self.assertEqual(ast.literal_eval(expression.right), expected_path)

        self.assertEqual(
            {
                f"{Path(path).stem}_sweep__V_BG"
                for path in input_paths.values()
            },
            {
                "Single_Empty_Quantum_Dot_2D_sweep__V_BG",
                "Single_Quantum_Dot_2D_sweep__V_BG",
            },
        )

    def test_cases_share_one_scientifically_distinct_mapping(self):
        cases = self.assignment_node("CASES")
        self.assertIsInstance(cases, ast.Dict)
        keys = [ast.literal_eval(key) for key in cases.keys]
        self.assertEqual(keys, ["empty", "accumulated"])

        references = {}
        labels = {}
        for key, value in zip(keys, cases.values, strict=True):
            fields = [ast.literal_eval(field) for field in value.keys]
            self.assertEqual(set(fields), {"input_path", "sweep_root", "label"})
            references[key] = {
                field: ast.unparse(value.values[fields.index(field)])
                for field in ("input_path", "sweep_root")
            }
            labels[key] = ast.literal_eval(value.values[fields.index("label")])

        self.assertEqual(
            references,
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
        self.assertIn("Empty", labels["empty"])
        self.assertIn("Accumulated", labels["accumulated"])

    def test_optional_execution_is_guarded_shared_and_path_safe(self):
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
        for fragment in (
            "CASES.values()",
            "case['input_path'].is_file()",
            "FileNotFoundError",
            "case['sweep_root'].parent.expanduser().resolve()",
            "!= SIMULATION_OUTPUT_ROOT",
            "CASES.items()",
            "sweep.sweep_output_directory",
            "actual_sweep_root != expected_sweep_root",
            "SWEEP_RESULTS[case_name] = sweep",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, guard_source)

        shared_loops = [
            node
            for node in ast.walk(guard)
            if isinstance(node, ast.For) and contains_call(node, "run_sweep")
        ]
        self.assertEqual(len(shared_loops), 1)
        self.assertEqual(ast.unparse(shared_loops[0].iter), "CASES.items()")

        _, run_call = self.calls_named("run_sweep")[0]
        run_keywords = keyword_map(run_call)
        self.assertTrue(
            {
                "delete_old_files",
                "delete_input_files",
                "overwrite",
                "show_log",
                "convergenceCheck",
                "parallel_limit",
                "execute_kwargs",
            }.issubset(run_keywords)
        )
        self.assertEqual(
            ast.unparse(run_keywords["execute_kwargs"]),
            "{'outputdirectory': str(case['sweep_root'].parent)}",
        )

        combined_code = "\n".join(self.code_sources.values())
        self.assertNotIn("nn.config", combined_code)
        self.assertNotIn("import nextnanopy", combined_code)
        for name in ("configure_nextnano", "to_default", "save"):
            self.assertFalse(self.calls_named(name))

    def test_discovery_loading_and_plotting_are_case_driven(self):
        manifests = self.assignment_node("MANIFESTS")
        self.assertIsInstance(manifests, ast.DictComp)
        self.assertEqual(ast.unparse(manifests.generators[0].iter), "CASES.items()")
        discover_calls = [
            node
            for node in ast.walk(manifests)
            if isinstance(node, ast.Call)
            and call_name(node) == "discover_sweep_runs"
        ]
        self.assertEqual(len(discover_calls), 1)
        discover = discover_calls[0]
        self.assertEqual(ast.unparse(discover.args[0]), "case['sweep_root']")
        self.assertEqual(ast.unparse(discover.args[1]), "SWEEP_VARIABLE")
        discover_keywords = keyword_map(discover)
        self.assertEqual(
            set(discover_keywords),
            {"bias", "required_outputs", "strict"},
        )
        self.assertEqual(ast.unparse(discover_keywords["bias"]), "SELECTED_BIAS")
        self.assertEqual(
            ast.unparse(discover_keywords["required_outputs"]),
            "(SELECTED_OUTPUT,)",
        )
        self.assertEqual(ast.unparse(discover_keywords["strict"]), "STRICT_DISCOVERY")
        self.assertIs(self.assignment_value("STRICT_DISCOVERY"), True)

        loaded = self.assignment_node("LOADED_OUTPUTS")
        self.assertIsInstance(loaded, ast.DictComp)
        self.assertEqual(ast.unparse(loaded.generators[0].iter), "CASES")
        load_calls = [
            node
            for node in ast.walk(loaded)
            if isinstance(node, ast.Call)
            and call_name(node) == "load_sweep_outputs"
        ]
        self.assertEqual(len(load_calls), 1)
        self.assertEqual(ast.unparse(load_calls[0].args[0]), "MANIFESTS[case_name]")
        self.assertEqual(ast.unparse(load_calls[0].args[1]), "SELECTED_OUTPUT")
        self.assertEqual(
            ast.unparse(keyword_map(load_calls[0])["variable"]),
            "SELECTED_VARIABLE",
        )

        plot_calls = self.calls_named("plot_sweep_lines")
        interactive_values = []
        for index, call in plot_calls:
            self.assertEqual(ast.unparse(call.args[0]), "LOADED_OUTPUTS[case_name]")
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
            containing_loops = [
                node
                for node in ast.walk(self.trees[index])
                if isinstance(node, ast.For)
                and any(child is call for child in ast.walk(node))
            ]
            self.assertEqual(len(containing_loops), 1)
            self.assertEqual(ast.unparse(containing_loops[0].iter), "CASES.items()")
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

    def test_summaries_metadata_paths_and_markdown_are_clean(self):
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
        combined_code = "\n".join(self.code_sources.values())
        for fragment in (
            "manifest.empty",
            'manifest["complete"]',
            'manifest["outputs_available"]',
            'outputs["load_error"]',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, combined_code)
        self.assertTrue(self.calls_named("display"))

        self.assertEqual(len(self.cells), 11)
        for index, cell in self.code_cells:
            self.assertIsNone(cell.get("execution_count"), f"Code cell {index}")
            self.assertEqual(cell.get("outputs"), [], f"Code cell {index}")
            self.assertTrue(cell_source(cell).strip(), f"Empty code cell {index}")
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
        repository_run_literals = []
        input_literals = []
        for tree in self.trees.values():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if (
                    "qpu-design-automation-toolkit" in node.value
                    and Path(node.value).is_absolute()
                ):
                    repository_path_literals.append(node.value)
                if node.value.startswith("runs/"):
                    repository_run_literals.append(node.value)
                if node.value.endswith(".in"):
                    input_literals.append(node.value)
        self.assertEqual(repository_path_literals, [])
        self.assertEqual(repository_run_literals, [])
        self.assertEqual(len(input_literals), 2)
        for input_literal in input_literals:
            self.assertFalse(Path(input_literal).is_absolute())
            self.assertTrue(input_literal.startswith("configs/"))

        markdown = "\n".join(
            cell_source(cell)
            for cell in self.cells
            if cell["cell_type"] == "markdown"
        ).lower()
        for phrase in (
            "barrier-gate voltage sweep",
            "empty quantum dot",
            "accumulated",
            "optional",
            "disabled by default",
            "potential_1d_x_qd.dat",
            "hh band-edge",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)
