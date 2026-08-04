from __future__ import annotations

import ast
import copy
import json
import re
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from qd_design import (
    LinearDotArrayDevice,
    TopBarrierLinearDotArrayDevice,
    TwoDDotArrayDevice,
    build_simulation_layout,
    derive_quantum_region,
    make_reference_sige_ge_process_stack,
    write_nextnano_input_from_template,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPO_ROOT
    / "notebooks"
    / "generation"
    / "01_generate_thesis_device_inputs.ipynb"
)
TEMPLATE_PATH = (
    REPO_ROOT
    / "configs"
    / "robert_inputs"
    / "double_qd"
    / "3d"
    / "Double_Quantum_Dot_3D.in"
)
REPOSITORY_GEOMETRY_OUTPUT = REPO_ROOT / "data" / "gds" / "new_generated"
REPOSITORY_INPUT_OUTPUT = (
    REPO_ROOT / "configs" / "robert_inputs" / "new_generated"
)

EXPECTED_COMMON_GEOMETRY = {
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
    "barrier_to_plunger_gap_nm": 20.0,
}

EXPECTED_CASES = {
    "single_qd_bottom_barriers": {
        "builder_class": LinearDotArrayDevice,
        "special_kwargs": {"n_dots": 1, "ohmic_to_barrier_gap_nm": 20.0},
        "counts": {"plunger": 1, "barrier": 2, "ohmic": 2},
        "domain": (-170.0, 170.0, 0.0, 200.0),
        "quantum": (-130.0, 130.0, 20.0, 180.0, -20.0, 5.0),
        "filename": "Single_Quantum_Dot_Bottom_Barriers_3D_from_PHIDL.in",
    },
    "lateral_dqd_bottom_barriers": {
        "builder_class": LinearDotArrayDevice,
        "special_kwargs": {"n_dots": 2, "ohmic_to_barrier_gap_nm": 1000.0},
        "counts": {"plunger": 2, "barrier": 3, "ohmic": 2},
        "domain": (-1240.0, 1240.0, 0.0, 200.0),
        "quantum": (-220.0, 220.0, 20.0, 180.0, -20.0, 5.0),
        "filename": (
            "Lateral_Double_Quantum_Dot_Bottom_Barriers_3D_from_PHIDL.in"
        ),
    },
    "lateral_dqd_top_barriers": {
        "builder_class": TopBarrierLinearDotArrayDevice,
        "special_kwargs": {"n_dots": 2, "ohmic_to_barrier_gap_nm": 1000.0},
        "counts": {"plunger": 2, "barrier": 3, "ohmic": 2},
        "domain": (-1240.0, 1240.0, 0.0, 200.0),
        "quantum": (-220.0, 220.0, 20.0, 180.0, -20.0, 5.0),
        "filename": "Lateral_Double_Quantum_Dot_Top_Barriers_3D_from_PHIDL.in",
    },
    "qd_array_2x2": {
        "builder_class": TwoDDotArrayDevice,
        "special_kwargs": {
            "n_columns": 2,
            "row_gap_nm": 0.0,
            "ohmic_to_barrier_gap_nm": 20.0,
        },
        "counts": {"plunger": 4, "barrier": 6, "ohmic": 4},
        "domain": (-260.0, 260.0, 0.0, 400.0),
        "quantum": (-220.0, 220.0, 20.0, 380.0, -20.0, 5.0),
        "filename": "Quantum_Dot_Array_2x2_3D_from_PHIDL.in",
    },
    "qd_array_2x3": {
        "builder_class": TwoDDotArrayDevice,
        "special_kwargs": {
            "n_columns": 3,
            "row_gap_nm": 0.0,
            "ohmic_to_barrier_gap_nm": 20.0,
        },
        "counts": {"plunger": 6, "barrier": 8, "ohmic": 4},
        "domain": (-350.0, 350.0, 0.0, 400.0),
        "quantum": (-310.0, 310.0, 20.0, 380.0, -20.0, 5.0),
        "filename": "Quantum_Dot_Array_2x3_3D_from_PHIDL.in",
    },
    "single_qd_top_barriers": {
        "builder_class": TopBarrierLinearDotArrayDevice,
        "special_kwargs": {"n_dots": 1, "ohmic_to_barrier_gap_nm": 20.0},
        "counts": {"plunger": 1, "barrier": 2, "ohmic": 2},
        "domain": (-170.0, 170.0, 0.0, 200.0),
        "quantum": (-130.0, 130.0, 20.0, 180.0, -20.0, 5.0),
        "filename": "Single_Quantum_Dot_Top_Barriers_3D_from_PHIDL.in",
    },
    "vertical_dqd_2x1": {
        "builder_class": TwoDDotArrayDevice,
        "special_kwargs": {
            "n_columns": 1,
            "row_gap_nm": 0.0,
            "ohmic_to_barrier_gap_nm": 20.0,
        },
        "counts": {"plunger": 2, "barrier": 4, "ohmic": 4},
        "domain": (-170.0, 170.0, 0.0, 400.0),
        "quantum": (-130.0, 130.0, 20.0, 380.0, -20.0, 5.0),
        "filename": "Vertical_Double_Quantum_Dot_2x1_3D_from_PHIDL.in",
    },
}

EXPECTED_BACKGROUND_Z = {
    "SiGe_body_contact": (-4115.0, -4015.0),
    "SiGe_buffer": (-4015.0, -15.0),
    "Ge_QW": (-15.0, 0.0),
    "SiGe_cap": (0.0, 101.0),
    "Al2O3_dielectric": (101.0, 173.0),
}
EXPECTED_PATTERNED_Z = {
    "barrier": (108.0, 138.0),
    "plunger": (143.0, 173.0),
    "ohmic": (-149.0, 173.0),
}


def _cell_source(notebook, cell_id):
    matches = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise AssertionError(f"Expected one notebook cell named {cell_id!r}")
    source = matches[0].get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _flatten_path_expression(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _flatten_path_expression(node.left) + _flatten_path_expression(
            node.right
        )
    raise AssertionError(f"Unexpected path expression: {ast.dump(node)}")


def _simple_assignment(tree, name):
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one assignment to {name}")
    return matches[0]


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _subscript_name_and_key(node):
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.value.id, node.slice.value
    return None


def _directory_signature(path):
    if not path.exists():
        return None
    return tuple(
        (
            entry.relative_to(path).as_posix(),
            entry.stat().st_size,
            entry.stat().st_mtime_ns,
        )
        for entry in sorted(path.rglob("*"))
        if entry.is_file()
    )


class ThesisDeviceInputNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_by_id = {
            cell["id"]: _cell_source(cls.notebook, cell["id"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        }

        registry_namespace = {
            "LinearDotArrayDevice": LinearDotArrayDevice,
            "TopBarrierLinearDotArrayDevice": TopBarrierLinearDotArrayDevice,
            "TwoDDotArrayDevice": TwoDDotArrayDevice,
        }
        exec(
            compile(
                cls.code_by_id["case-registry"],
                f"{NOTEBOOK_PATH}:case-registry",
                "exec",
            ),
            registry_namespace,
        )
        cls.cases = registry_namespace["CASES"]

        cls.process_stack = make_reference_sige_ge_process_stack()
        cls.built_cases = {}
        for case_key, case in cls.cases.items():
            builder = case["builder_class"](**dict(case["builder_kwargs"]))
            builder.ensure_built()
            layout_spec = builder.layout_spec()
            simulation_layout = build_simulation_layout(
                name=f"{case_key}_test_layout",
                layout_elements=layout_spec,
                process_stack=cls.process_stack,
                x_margin_nm=0.0,
                y_margin_nm=0.0,
            )
            cls.built_cases[case_key] = {
                "builder": builder,
                "layout_spec": layout_spec,
                "simulation_layout": simulation_layout,
                "quantum_region": derive_quantum_region(simulation_layout),
            }

    def test_notebook_is_valid_clean_json(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["kernelspec"]["name"], "rj_thesis_project")
        self.assertTrue(self.notebook["cells"])
        self.assertEqual(
            len({cell["id"] for cell in self.notebook["cells"]}),
            len(self.notebook["cells"]),
        )

        for index, cell in enumerate(self.notebook["cells"]):
            with self.subTest(cell=index):
                self.assertFalse(cell.get("attachments"))
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])

        self.assertTrue(_cell_source(self.notebook, self.notebook["cells"][-1]["id"]).strip())

    def test_exact_output_paths_template_and_directory_creation_location(self):
        tree = ast.parse(self.code_by_id["imports-paths"])
        expected_paths = {
            "GEOMETRY_OUTPUT_DIR": (
                "REPO_ROOT",
                "data",
                "gds",
                "new_generated",
            ),
            "GENERATED_INPUT_DIR": (
                "REPO_ROOT",
                "configs",
                "robert_inputs",
                "new_generated",
            ),
            "TEMPLATE_INPUT_PATH": (
                "REPO_ROOT",
                "configs",
                "robert_inputs",
                "double_qd",
                "3d",
                "Double_Quantum_Dot_3D.in",
            ),
        }
        for name, expected in expected_paths.items():
            self.assertEqual(
                _flatten_path_expression(_simple_assignment(tree, name)),
                expected,
            )

        mkdir_cells = []
        for cell_id, source in self.code_by_id.items():
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "mkdir"
                ):
                    mkdir_cells.append(cell_id)
        self.assertEqual(mkdir_cells, ["guarded-export", "guarded-export"])

    def test_registry_has_exact_builders_parameters_and_unique_filenames(self):
        self.assertEqual(list(self.cases), list(EXPECTED_CASES))
        filenames = []
        for case_key, expected in EXPECTED_CASES.items():
            case = self.cases[case_key]
            kwargs = case["builder_kwargs"]
            with self.subTest(case=case_key):
                self.assertIs(case["builder_class"], expected["builder_class"])
                self.assertIsInstance(case["builder_class"], type)
                self.assertEqual(kwargs["name"], case_key)
                for name, value in EXPECTED_COMMON_GEOMETRY.items():
                    self.assertEqual(kwargs[name], value)
                for name, value in expected["special_kwargs"].items():
                    self.assertEqual(kwargs[name], value)
                self.assertEqual(case["expected_counts"], expected["counts"])
                self.assertEqual(case["expected_domain_nm"], expected["domain"])
                self.assertEqual(case["nextnano_filename"], expected["filename"])
                json.dumps(kwargs)
                filenames.append(case["nextnano_filename"])

        self.assertEqual(len(filenames), len(set(filenames)))
        self.assertEqual(
            self.cases["lateral_dqd_bottom_barriers"]["builder_kwargs"][
                "ohmic_to_barrier_gap_nm"
            ],
            1000.0,
        )
        self.assertEqual(
            self.cases["lateral_dqd_top_barriers"]["builder_kwargs"][
                "ohmic_to_barrier_gap_nm"
            ],
            1000.0,
        )
        vertical = self.cases["vertical_dqd_2x1"]
        self.assertIs(vertical["builder_class"], TwoDDotArrayDevice)
        self.assertEqual(vertical["builder_kwargs"]["n_columns"], 1)
        self.assertNotIn("n_dots", vertical["builder_kwargs"])

    def test_public_pipeline_produces_expected_counts_domains_stack_and_quantum_boxes(self):
        for case_key, expected in EXPECTED_CASES.items():
            record = self.built_cases[case_key]
            layout_spec = record["layout_spec"]
            simulation_layout = record["simulation_layout"]
            domain = simulation_layout.domain
            quantum = record["quantum_region"]
            with self.subTest(case=case_key):
                self.assertEqual(
                    dict(Counter(item["gate_type"] for item in layout_spec)),
                    expected["counts"],
                )
                self.assertEqual(
                    (
                        domain.x_min_nm,
                        domain.x_max_nm,
                        domain.y_min_nm,
                        domain.y_max_nm,
                    ),
                    expected["domain"],
                )
                self.assertEqual((domain.z_min_nm, domain.z_max_nm), (-4115.0, 173.0))
                self.assertEqual(
                    (
                        quantum.x_min_nm,
                        quantum.x_max_nm,
                        quantum.y_min_nm,
                        quantum.y_max_nm,
                        quantum.z_min_nm,
                        quantum.z_max_nm,
                    ),
                    expected["quantum"],
                )
                self.assertTrue(
                    domain.x_min_nm <= quantum.x_min_nm < quantum.x_max_nm <= domain.x_max_nm
                    and domain.y_min_nm <= quantum.y_min_nm < quantum.y_max_nm <= domain.y_max_nm
                    and domain.z_min_nm <= quantum.z_min_nm < quantum.z_max_nm <= domain.z_max_nm
                )
                self.assertEqual(
                    {
                        region.name: (region.z_min_nm, region.z_max_nm)
                        for region in simulation_layout.background_regions
                    },
                    EXPECTED_BACKGROUND_Z,
                )
                self.assertEqual(
                    next(
                        region.contact_name
                        for region in simulation_layout.background_regions
                        if region.name == "SiGe_body_contact"
                    ),
                    "Body",
                )
                self.assertTrue(
                    all(
                        (region.z_min_nm, region.z_max_nm)
                        == EXPECTED_PATTERNED_Z[region.gate_type]
                        for region in simulation_layout.patterned_regions
                    )
                )

    def test_notebook_calls_only_the_public_visualization_and_generation_pipeline(self):
        required = {
            "plot_layout_spec_2d",
            "plot_simulation_layout_3d",
            "make_reference_sige_ge_process_stack",
            "build_simulation_layout",
            "derive_quantum_region",
            "write_nextnano_input_from_template",
        }
        imports_tree = ast.parse(self.code_by_id["imports-paths"])
        qd_imports = {
            alias.name
            for node in ast.walk(imports_tree)
            if isinstance(node, ast.ImportFrom) and node.module == "qd_design"
            for alias in node.names
        }
        self.assertTrue(required <= qd_imports)

        called = {
            _dotted_name(node.func).split(".")[-1]
            for source in self.code_by_id.values()
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
        }
        self.assertTrue(required <= called)

    def test_voltage_overrides_are_derived_generically_from_each_layout(self):
        tree = ast.parse(self.code_by_id["build-cases"])
        matches = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "voltage_overrides"
                for target in node.targets
            )
        ]
        self.assertEqual(len(matches), 1)
        comprehension = matches[0]
        self.assertIsInstance(comprehension, ast.DictComp)
        self.assertEqual(
            _subscript_name_and_key(comprehension.key),
            ("element", "voltage_label"),
        )
        self.assertIsInstance(comprehension.value, ast.IfExp)
        self.assertEqual(
            _subscript_name_and_key(comprehension.value.test.left),
            ("element", "gate_type"),
        )
        self.assertEqual(comprehension.value.test.comparators[0].value, "plunger")
        self.assertEqual(comprehension.value.body.id, "PLUNGER_VOLTAGE_V")
        self.assertEqual(comprehension.value.orelse.id, "OTHER_GATE_VOLTAGE_V")
        self.assertFalse(
            any(
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("V_")
                for node in ast.walk(comprehension)
            )
        )

        expression = ast.Expression(body=copy.deepcopy(comprehension))
        ast.fix_missing_locations(expression)
        for case_key, record in self.built_cases.items():
            overrides = eval(
                compile(expression, f"{NOTEBOOK_PATH}:voltage-overrides", "eval"),
                {
                    "layout_spec": record["layout_spec"],
                    "PLUNGER_VOLTAGE_V": -3.0,
                    "OTHER_GATE_VOLTAGE_V": 0.0,
                },
            )
            with self.subTest(case=case_key):
                self.assertEqual(
                    set(overrides),
                    {element["voltage_label"] for element in record["layout_spec"]},
                )
                for element in record["layout_spec"]:
                    self.assertEqual(
                        overrides[element["voltage_label"]],
                        -3.0 if element["gate_type"] == "plunger" else 0.0,
                    )

    def test_write_guard_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            planned_geometry = temporary_root / "geometry" / "case.gds"
            planned_input = temporary_root / "inputs" / "case.in"
            namespace = {
                "WRITE_FILES": False,
                "TARGET_PATHS_BY_CASE": {
                    "case": {"gds": planned_geometry, "nextnano_input": planned_input}
                },
                "pd": SimpleNamespace(DataFrame=lambda rows: tuple(rows)),
                "display": lambda value: value,
            }
            with redirect_stdout(StringIO()):
                exec(
                    compile(
                        self.code_by_id["guarded-export"],
                        f"{NOTEBOOK_PATH}:guarded-export-dry-run",
                        "exec",
                    ),
                    namespace,
                )
            self.assertEqual(namespace["WRITTEN_PATHS_BY_CASE"], {})
            self.assertFalse(planned_geometry.parent.exists())
            self.assertFalse(planned_input.parent.exists())

    def test_collision_aborts_before_output_directories_are_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            collision = temporary_root / "already-exists.in"
            collision.write_text("existing", encoding="utf-8")
            geometry_output = temporary_root / "geometry" / "new_generated"
            input_output = temporary_root / "inputs" / "new_generated"
            namespace = {
                "WRITE_FILES": True,
                "OVERWRITE_EXISTING": False,
                "all_resolved_targets": [collision],
                "GEOMETRY_OUTPUT_DIR": geometry_output,
                "GENERATED_INPUT_DIR": input_output,
            }
            with self.assertRaisesRegex(FileExistsError, "already-exists.in"):
                exec(
                    compile(
                        self.code_by_id["guarded-export"],
                        f"{NOTEBOOK_PATH}:guarded-export-collision",
                        "exec",
                    ),
                    namespace,
                )
            self.assertFalse(geometry_output.exists())
            self.assertFalse(input_output.exists())

    def test_no_nextnano_execution_import_or_simulation_result_analysis(self):
        trees = [ast.parse(source) for source in self.code_by_id.values()]
        forbidden_modules = {"nextnanopy", "nextnanopp_tools"}
        forbidden_calls = {
            "run_input_file",
            "set_input_variables",
            "execute_input",
            "execute",
            "DataFile",
            "DataFolder",
            "OutputFolder",
            "load_raw_data",
        }
        subprocess_calls = []
        read_receivers = []
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(
                        all(alias.name.split(".")[0] not in forbidden_modules for alias in node.names)
                    )
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden_modules)
                elif isinstance(node, ast.Call):
                    call_name = _dotted_name(node.func)
                    self.assertNotIn(call_name.split(".")[-1], forbidden_calls)
                    if call_name == "subprocess.run":
                        subprocess_calls.append(node)
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "read_text":
                        read_receivers.append(_dotted_name(node.func.value))

        self.assertEqual(len(subprocess_calls), 1)
        self.assertEqual(
            ast.literal_eval(subprocess_calls[0].args[0]),
            ["git", "rev-parse", "HEAD"],
        )
        self.assertEqual(sorted(read_receivers), ["TEMPLATE_INPUT_PATH", "input_path"])

        code_text = "\n".join(self.code_by_id.values()).lower()
        for result_term in (
            "bandedges",
            "wavefunctions",
            "probability_density",
            "simulation_results",
            "output_folder",
        ):
            self.assertNotIn(result_term, code_text)

    def test_manifest_and_generated_input_validation_contracts_are_complete(self):
        preflight_tree = ast.parse(self.code_by_id["preflight"])
        manifest_dicts = [
            node.value
            for node in ast.walk(preflight_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "MANIFESTS_BY_CASE"
                for target in node.targets
            )
        ]
        self.assertEqual(len(manifest_dicts), 1)
        manifest_keys = {
            key.value
            for key in manifest_dicts[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertEqual(
            manifest_keys,
            {
                "case_key",
                "builder_class",
                "builder_parameters",
                "builder_summary",
                "voltage_overrides",
                "template_path",
                "git_commit",
                "simulation_domain",
                "derived_quantum_region",
                "process_stack_description",
                "generated_relative_paths",
            },
        )

        validation_tree = ast.parse(self.code_by_id["written-validation-summary"])
        checks_dicts = [
            node.value
            for node in ast.walk(validation_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "checks"
                for target in node.targets
            )
        ]
        self.assertEqual(len(checks_dicts), 1)
        check_keys = {
            key.value
            for key in checks_dicts[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertEqual(
            check_keys,
            {
                "nonempty_file",
                "all_voltage_labels_defined",
                "all_patterned_regions_represented",
                "polygonal_prism_count_matches",
                "body_contact_present",
                "remove_surface_charge_present",
                "zero_fermi_QW_present",
                "quantum_bounds_match",
                "solver_controls_preserved",
                "generated_run_block_present",
                "all_case_targets_written",
                "no_unresolved_path_collision",
            },
        )

    def test_public_export_pipeline_runs_only_in_a_temporary_directory(self):
        geometry_before = _directory_signature(REPOSITORY_GEOMETRY_OUTPUT)
        inputs_before = _directory_signature(REPOSITORY_INPUT_OUTPUT)
        case_key = "single_qd_bottom_barriers"
        case = self.cases[case_key]
        record = self.built_cases[case_key]
        layout_spec = record["layout_spec"]
        simulation_layout = record["simulation_layout"]
        quantum = record["quantum_region"]
        voltage_overrides = {
            element["voltage_label"]: (
                -3.0 if element["gate_type"] == "plunger" else 0.0
            )
            for element in layout_spec
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory).resolve()
            geometry_dir = temporary_root / "geometry"
            input_dir = temporary_root / "inputs"
            paths = {
                "gds": geometry_dir / f"{case_key}.gds",
                "svg": geometry_dir / f"{case_key}.svg",
                "layout_spec": geometry_dir / f"{case_key}_layout_spec.json",
                "simulation_layout": geometry_dir / f"{case_key}_simulation_layout.json",
                "generation_manifest": geometry_dir / f"{case_key}_generation_manifest.json",
                "nextnano_input": input_dir / case["nextnano_filename"],
            }
            self.assertEqual(len(paths.values()), len(set(paths.values())))
            self.assertTrue(
                all(path.resolve().is_relative_to(temporary_root) for path in paths.values())
            )
            self.assertFalse(any(path.exists() for path in paths.values()))

            geometry_dir.mkdir(parents=True)
            input_dir.mkdir(parents=True)
            record["builder"].write_gds(str(paths["gds"]))
            record["builder"].write_svg(str(paths["svg"]))
            record["builder"].write_layout_spec_json(str(paths["layout_spec"]))
            simulation_layout.write_json(str(paths["simulation_layout"]))
            write_nextnano_input_from_template(
                simulation_layout=simulation_layout,
                template_path=TEMPLATE_PATH,
                output_path=paths["nextnano_input"],
                voltage_overrides=voltage_overrides,
            )

            manifest = {
                "case_key": case_key,
                "builder_class": case["builder_class"].__name__,
                "builder_parameters": dict(case["builder_kwargs"]),
                "builder_summary": record["builder"].summary(),
                "voltage_overrides": voltage_overrides,
                "template_path": TEMPLATE_PATH.relative_to(REPO_ROOT).as_posix(),
                "git_commit": None,
                "simulation_domain": simulation_layout.domain.to_dict(),
                "derived_quantum_region": quantum.to_dict(),
                "process_stack_description": self.process_stack.to_dict(),
                "generated_relative_paths": {
                    name: path.relative_to(temporary_root).as_posix()
                    for name, path in paths.items()
                },
            }
            paths["generation_manifest"].write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )

            for artifact_name, path in paths.items():
                with self.subTest(artifact=artifact_name):
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)

            self.assertIsInstance(json.loads(paths["layout_spec"].read_text()), list)
            serialized_layout = json.loads(paths["simulation_layout"].read_text())
            self.assertEqual(serialized_layout["domain"], simulation_layout.domain.to_dict())
            self.assertEqual(json.loads(paths["generation_manifest"].read_text()), manifest)

            generated = paths["nextnano_input"].read_text(encoding="utf-8")
            template = TEMPLATE_PATH.read_text(encoding="utf-8")
            self.assertTrue(generated.strip())
            for element in layout_spec:
                self.assertRegex(
                    generated,
                    rf"(?m)^\${re.escape(element['voltage_label'])}\s*=",
                )
            for region in simulation_layout.patterned_regions:
                self.assertIn(f"# patterned region: {region.name}", generated)
            self.assertEqual(
                generated.count("polygonal_prism{"),
                sum(len(region.polygon_xy_nm) for region in simulation_layout.patterned_regions),
            )
            self.assertIn("contact{ name = Body }", generated)
            self.assertIn("remove_surface_charge", generated)
            self.assertIn("zero_fermi_QW", generated)
            self.assertIn("run{", generated)
            for solver_name in ("strain", "poisson", "quantum", "quantum_poisson"):
                self.assertIn(f"!WHEN ${solver_name}", generated)
                pattern = rf"(?m)^\${solver_name}\s*=\s*([^#\n]+)"
                self.assertEqual(
                    re.search(pattern, generated).group(1).strip(),
                    re.search(pattern, template).group(1).strip(),
                )

            quantum_match = re.search(
                r'name\s*=\s*"c-Ge_QW"\s*'
                r'x\s*=\s*\[([-+0-9.eE]+),\s*([-+0-9.eE]+)\]\s*'
                r'y\s*=\s*\[([-+0-9.eE]+),\s*([-+0-9.eE]+)\]\s*'
                r'z\s*=\s*\[([-+0-9.eE]+),\s*([-+0-9.eE]+)\]',
                generated,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(quantum_match)
            quantum_values = [float(value) for value in quantum_match.groups()]
            self.assertEqual(
                quantum_values,
                [
                    quantum.x_min_nm,
                    quantum.x_max_nm,
                    quantum.y_min_nm,
                    quantum.y_max_nm,
                    quantum.z_min_nm,
                    quantum.z_max_nm,
                ],
            )

        self.assertEqual(_directory_signature(REPOSITORY_GEOMETRY_OUTPUT), geometry_before)
        self.assertEqual(_directory_signature(REPOSITORY_INPUT_OUTPUT), inputs_before)


if __name__ == "__main__":
    unittest.main()
