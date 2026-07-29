import math
import tempfile
import unittest
from pathlib import Path

import nextnanopp_tools as nnt


EXPECTED_COLUMNS = [
    "sweep_variable",
    "sweep_value",
    "run_root",
    "run_name",
    "value_source",
    "bias_dir",
    "complete",
    "required_outputs",
    "outputs_available",
    "error",
]


class SweepDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_root = Path(self.temporary_directory.name)
        self.sweep_root = self.temporary_root / "device_sweep__V_PG"
        self.sweep_root.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_run(
        self,
        name,
        *,
        variables_text=None,
        biases=("bias_00000",),
        complete=False,
    ):
        run_root = self.sweep_root / name
        run_root.mkdir()
        if variables_text is not None:
            (run_root / "variables_input.txt").write_text(
                variables_text,
                encoding="utf-8",
            )
        for bias_name in biases:
            (run_root / bias_name).mkdir()
        if complete:
            (run_root / "job_done.txt").write_text("done\n", encoding="utf-8")
        return run_root

    def test_metadata_parses_supported_numeric_forms(self):
        cases = {
            "metadata_negative_integer": ("-3", -3.0),
            "metadata_positive_integer": ("+4", 4.0),
            "metadata_decimal": ("2.5", 2.5),
            "metadata_leading_decimal": (".25", 0.25),
            "metadata_scientific": ("-2.5E+3", -2500.0),
        }
        for run_name, (token, _) in cases.items():
            self.make_run(
                run_name,
                variables_text=f"\t$ V_PG = {token} # swept value\n",
            )

        runs = nnt.discover_sweep_runs(self.sweep_root, "V_PG")
        rows = {row.run_name: row for row in runs.itertuples()}

        for run_name, (_, expected) in cases.items():
            with self.subTest(run_name=run_name):
                self.assertEqual(rows[run_name].sweep_value, expected)
                self.assertEqual(rows[run_name].value_source, "variables_input")
                self.assertIsNone(rows[run_name].error)

    def test_metadata_takes_precedence_over_folder_name(self):
        run_root = self.make_run(
            "Device_2D__V_PG_7_",
            variables_text="$V_PG = -3\n",
        )

        row = nnt.discover_sweep_runs(self.sweep_root, "V_PG").iloc[0]

        self.assertEqual(row["sweep_value"], -3.0)
        self.assertEqual(row["value_source"], "variables_input")
        self.assertEqual(row["run_root"], run_root.resolve())
        self.assertIsInstance(row["run_root"], Path)

    def test_metadata_variable_name_matching_is_exact(self):
        self.make_run(
            "Device__V_PG_-4_",
            variables_text="$V_PG1 = 99\n",
        )

        row = nnt.discover_sweep_runs(self.sweep_root, "V_PG").iloc[0]

        self.assertEqual(row["sweep_value"], -4.0)
        self.assertEqual(row["value_source"], "folder_name")

    def test_strict_folder_fallback_parses_supported_numeric_forms(self):
        cases = {
            "Device__V_PG_-3_": -3.0,
            "Device__V_PG_+2.5_": 2.5,
            "Device__V_PG_1e-2_": 0.01,
            "Device__V_PG_4_": 4.0,
        }
        for run_name in cases:
            self.make_run(run_name)

        runs = nnt.discover_sweep_runs(self.sweep_root, "V_PG")
        rows = {row.run_name: row for row in runs.itertuples()}

        for run_name, expected in cases.items():
            with self.subTest(run_name=run_name):
                self.assertEqual(rows[run_name].sweep_value, expected)
                self.assertEqual(rows[run_name].value_source, "folder_name")

    def test_folder_fallback_supports_later_nextnanopy_variable(self):
        self.make_run("sweep_example__ALLOY_0.3_SIZE_80_")

        row = nnt.discover_sweep_runs(self.sweep_root, "SIZE").iloc[0]

        self.assertEqual(row["sweep_value"], 80.0)
        self.assertEqual(row["value_source"], "folder_name")

    def test_folder_fallback_does_not_parse_variable_like_input_stem_text(self):
        self.make_run("Device_2_SIZE_80__V_PG_3_")

        with self.assertRaisesRegex(ValueError, r"could not determine.*SIZE"):
            nnt.discover_sweep_runs(self.sweep_root, "SIZE")

    def test_folder_fallback_requires_native_sweep_delimiter(self):
        self.make_run("Device_2_V_PG_5_")
        self.make_run("V_PG_6_")

        runs = nnt.discover_sweep_runs(
            self.sweep_root,
            "V_PG",
            strict=False,
        )

        self.assertTrue(runs["sweep_value"].isna().all())
        self.assertTrue(runs["error"].str.contains("could not determine").all())

    def test_folder_variable_name_matching_is_exact(self):
        self.make_run("Device__V_PG1_3_")

        with self.assertRaisesRegex(ValueError, r"could not determine.*V_PG"):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_folder_name_never_uses_unrelated_2d_number(self):
        self.make_run("Single_Empty_Quantum_Dot_2D")

        with self.assertRaisesRegex(
            ValueError,
            r"Single_Empty_Quantum_Dot_2D.*V_PG",
        ):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_values_are_sorted_numerically_and_index_is_reset(self):
        for value in ("-2", "1.5", "-10", "0"):
            self.make_run(f"Device__V_PG_{value}_")

        runs = nnt.discover_sweep_runs(self.sweep_root, "V_PG")

        self.assertEqual(runs["sweep_value"].tolist(), [-10.0, -2.0, 0.0, 1.5])
        self.assertEqual(runs.index.tolist(), [0, 1, 2, 3])

    def test_duplicate_values_raise_in_strict_mode(self):
        self.make_run("run_a__V_PG_0_")
        self.make_run("run_b__V_PG_-0.0_")

        with self.assertRaisesRegex(
            ValueError,
            r"Duplicate sweep values.*V_PG.*run_a.*run_b",
        ):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_duplicate_values_mark_every_row_in_non_strict_mode(self):
        self.make_run("run_a__V_PG_0_")
        self.make_run("run_b__V_PG_-0.0_")
        self.make_run("run_c__V_PG_1_")

        runs = nnt.discover_sweep_runs(
            self.sweep_root,
            "V_PG",
            strict=False,
        )

        self.assertEqual(len(runs), 3)
        duplicates = runs[runs["sweep_value"] == 0.0]
        self.assertEqual(len(duplicates), 2)
        for error in duplicates["error"]:
            self.assertIn("Duplicate sweep values", error)
            self.assertIn("run_a", error)
            self.assertIn("run_b", error)
        self.assertIsNone(runs[runs["sweep_value"] == 1.0].iloc[0]["error"])

    def test_missing_value_raises_in_strict_mode(self):
        self.make_run("Device_2D")

        with self.assertRaisesRegex(ValueError, r"could not determine.*V_PG"):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_malformed_metadata_raises_before_folder_fallback_in_strict_mode(self):
        self.make_run(
            "Device__V_PG_5_",
            variables_text="$V_PG = not-a-number\n",
        )

        with self.assertRaisesRegex(ValueError, r"Malformed assignment.*V_PG"):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_missing_equals_is_malformed_exact_variable_metadata(self):
        self.make_run(
            "Device__V_PG_5_",
            variables_text="$V_PG -3\n",
        )

        with self.assertRaisesRegex(
            ValueError,
            r"Device__V_PG_5_.*Malformed assignment.*V_PG",
        ):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_conflicting_metadata_assignments_raise_in_strict_mode(self):
        self.make_run(
            "Device__V_PG_5_",
            variables_text="$V_PG = 1\n$V_PG = 2\n",
        )

        with self.assertRaisesRegex(
            ValueError,
            r"Conflicting duplicate assignments.*V_PG",
        ):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_identical_metadata_assignments_are_allowed(self):
        self.make_run(
            "Device",
            variables_text="$V_PG = 1\n$V_PG = 1.0\n",
        )

        row = nnt.discover_sweep_runs(self.sweep_root, "V_PG").iloc[0]

        self.assertEqual(row["sweep_value"], 1.0)
        self.assertEqual(row["value_source"], "variables_input")

    def test_bad_values_are_retained_and_sorted_last_in_non_strict_mode(self):
        self.make_run("valid__V_PG_-2_")
        self.make_run(
            "recover__V_PG_5_",
            variables_text="$V_PG = malformed\n",
        )
        self.make_run("missing_2D")

        runs = nnt.discover_sweep_runs(
            self.sweep_root,
            "V_PG",
            strict=False,
        )

        self.assertEqual(runs["run_name"].tolist()[:2], ["valid__V_PG_-2_", "recover__V_PG_5_"])
        recovered = runs.iloc[1]
        self.assertEqual(recovered["sweep_value"], 5.0)
        self.assertEqual(recovered["value_source"], "folder_name")
        self.assertIn("Malformed assignment", recovered["error"])
        malformed = runs.iloc[2]
        self.assertTrue(math.isnan(malformed["sweep_value"]))
        self.assertIn("could not determine", malformed["error"])

    def test_conflicting_metadata_is_retained_without_arbitrary_value_in_non_strict_mode(self):
        self.make_run(
            "Device_2D",
            variables_text="$V_PG = 1\n$V_PG = 2\n",
        )

        row = nnt.discover_sweep_runs(
            self.sweep_root,
            "V_PG",
            strict=False,
        ).iloc[0]

        self.assertTrue(math.isnan(row["sweep_value"]))
        self.assertIsNone(row["value_source"])
        self.assertIn("Conflicting duplicate assignments", row["error"])

    def test_completion_requires_job_done_file(self):
        complete = self.make_run("complete__V_PG_-1_", complete=True)
        incomplete = self.make_run("incomplete__V_PG_0_")
        directory_marker = self.make_run("directory__V_PG_1_")
        (directory_marker / "job_done.txt").mkdir()

        runs = nnt.discover_sweep_runs(self.sweep_root, "V_PG")
        completion = dict(zip(runs["run_root"], runs["complete"]))

        self.assertTrue(completion[complete.resolve()])
        self.assertFalse(completion[incomplete.resolve()])
        self.assertFalse(completion[directory_marker.resolve()])

    def test_bias_selection_reuses_existing_get_bias_dir_behaviour(self):
        run_root = self.make_run(
            "Device",
            variables_text="$V_PG = 1\n",
            biases=("bias_00000", "bias_00002"),
        )
        cases = {
            "first": "bias_00000",
            "latest": "bias_00002",
            None: "bias_00002",
            2: "bias_00002",
            1: "bias_00002",
            "bias_00002": "bias_00002",
            "2": "bias_00002",
        }

        for bias, expected_name in cases.items():
            with self.subTest(bias=bias):
                row = nnt.discover_sweep_runs(
                    self.sweep_root,
                    "V_PG",
                    bias=bias,
                ).iloc[0]
                self.assertEqual(row["bias_dir"], (run_root / expected_name).resolve())

    def test_missing_bias_raises_in_strict_mode(self):
        self.make_run(
            "Device",
            variables_text="$V_PG = 1\n",
            biases=(),
        )

        with self.assertRaisesRegex(
            FileNotFoundError,
            r"Device.*could not select bias",
        ):
            nnt.discover_sweep_runs(self.sweep_root, "V_PG")

    def test_missing_bias_is_reported_in_non_strict_mode(self):
        run_root = self.make_run(
            "Device",
            variables_text="$V_PG = 1\n",
            biases=(),
        )
        run_output = run_root / "integrated_density_hole.dat"
        run_output.write_text("value\n", encoding="utf-8")

        row = nnt.discover_sweep_runs(
            self.sweep_root,
            "V_PG",
            required_outputs=("integrated_density_hole.dat",),
            strict=False,
        ).iloc[0]

        self.assertIsNone(row["bias_dir"])
        self.assertIn("could not select bias", row["error"])
        self.assertEqual(
            row["required_outputs"]["integrated_density_hole.dat"],
            run_output.resolve(),
        )
        self.assertTrue(row["outputs_available"])

    def test_required_outputs_use_bias_then_run_root_and_report_missing(self):
        run_root = self.make_run(
            "Device",
            variables_text="$V_PG = 1\n",
        )
        bias_output = run_root / "bias_00000" / "potential_1d_x_QD.dat"
        run_duplicate = run_root / "potential_1d_x_QD.dat"
        run_output = run_root / "integrated_density_hole.dat"
        nested_decoy = run_root / "nested" / "missing.dat"
        bias_output.write_text("bias\n", encoding="utf-8")
        run_duplicate.write_text("run\n", encoding="utf-8")
        run_output.write_text("density\n", encoding="utf-8")
        nested_decoy.parent.mkdir()
        nested_decoy.write_text("must not be discovered\n", encoding="utf-8")

        row = nnt.discover_sweep_runs(
            self.sweep_root,
            "V_PG",
            required_outputs=(
                "potential_1d_x_QD.dat",
                "integrated_density_hole.dat",
                "missing.dat",
            ),
        ).iloc[0]

        self.assertEqual(
            row["required_outputs"],
            {
                "potential_1d_x_QD.dat": bias_output.resolve(),
                "integrated_density_hole.dat": run_output.resolve(),
                "missing.dat": None,
            },
        )
        self.assertFalse(row["outputs_available"])

        available = nnt.discover_sweep_runs(
            self.sweep_root,
            "V_PG",
            required_outputs=(
                "potential_1d_x_QD.dat",
                "integrated_density_hole.dat",
            ),
        ).iloc[0]
        self.assertTrue(available["outputs_available"])

    def test_absolute_required_output_is_checked_directly(self):
        self.make_run(
            "Device",
            variables_text="$V_PG = 1\n",
        )
        absolute_output = self.temporary_root / "shared-output.dat"
        absolute_output.write_text("shared\n", encoding="utf-8")

        row = nnt.discover_sweep_runs(
            str(self.sweep_root),
            "V_PG",
            required_outputs=(absolute_output,),
        ).iloc[0]

        self.assertEqual(
            row["required_outputs"][str(absolute_output)],
            absolute_output.resolve(),
        )
        self.assertTrue(row["outputs_available"])

    def test_empty_root_has_expected_schema_and_ignores_hidden_entries(self):
        (self.sweep_root / ".DS_Store").write_text("", encoding="utf-8")
        (self.sweep_root / "ordinary-file.txt").write_text("", encoding="utf-8")
        (self.sweep_root / ".hidden-run").mkdir()

        runs = nnt.discover_sweep_runs(self.sweep_root, "V_PG")

        self.assertTrue(runs.empty)
        self.assertEqual(runs.columns.tolist(), EXPECTED_COLUMNS)

    def test_invalid_sweep_roots_raise_specific_errors(self):
        missing = self.temporary_root / "missing"
        ordinary_file = self.temporary_root / "not-a-directory"
        ordinary_file.write_text("", encoding="utf-8")

        with self.assertRaises(FileNotFoundError):
            nnt.discover_sweep_runs(str(missing), "V_PG")
        with self.assertRaises(NotADirectoryError):
            nnt.discover_sweep_runs(ordinary_file, "V_PG")

    def test_discover_sweep_runs_is_public(self):
        self.assertIn("discover_sweep_runs", nnt.__all__)


if __name__ == "__main__":
    unittest.main()
