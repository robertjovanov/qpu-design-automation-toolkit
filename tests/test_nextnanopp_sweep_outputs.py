import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import nextnanopp_tools as nnt


DAT_TEXT = """x[nm] Alpha[eV] Beta[eV]
0 1 10
1 2 20
"""

FLD_TEXT = """# AVS/Express field file
ndim = 2
dim1 = 2
dim2 = 2
nspace = 2
veclen = 2
data = double
field = rectilinear
label = Alpha[eV]
label = Beta[eV]
variable 1 file=test.fld filetype=ascii skip=16 offset=0 stride=1
variable 2 file=test.fld filetype=ascii skip=17 offset=0 stride=1
coord 1 file=test.fld filetype=ascii skip=14 offset=0 stride=1
coord 2 file=test.fld filetype=ascii skip=15 offset=0 stride=1
0 1
-1 1
1 2 3 4
10 20 30 40
"""

VTR_TEXT = """<VTKFile type="RectilinearGrid" version="0.1" format="ascii">
<RectilinearGrid WholeExtent="0 1 0 0 0 0">
<Piece Extent="0 1 0 0 0 0">
<Coordinates>
<DataArray type="Float64" Name="X_COORDINATES" format="ascii">0 1</DataArray>
<DataArray type="Float64" Name="Y_COORDINATES" format="ascii">0</DataArray>
<DataArray type="Float64" Name="Z_COORDINATES" format="ascii">-1</DataArray>
</Coordinates>
<PointData>
<DataArray type="Float64" Name="Alpha[eV]" format="ascii">1 2</DataArray>
<DataArray type="Float64" Name="Beta[eV]" format="ascii">10 20</DataArray>
</PointData>
</Piece>
</RectilinearGrid>
</VTKFile>
"""

OUTPUT_COLUMNS = [
    "requested_output",
    "output_path",
    "output_available",
    "dataset",
    "load_error",
]


class SweepOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_run(self, name, *, with_bias=True):
        run_root = self.temporary_root / name
        run_root.mkdir()
        bias_dir = None
        if with_bias:
            bias_dir = run_root / "bias_00000"
            bias_dir.mkdir()
        return run_root, bias_dir

    @staticmethod
    def write_text(path, text=DAT_TEXT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(TypeError, "pandas DataFrame"):
            nnt.load_sweep_outputs([], "output.dat")

        with self.assertRaisesRegex(ValueError, "run_root"):
            nnt.load_sweep_outputs(pd.DataFrame({"bias_dir": []}), "output.dat")

        valid_empty = pd.DataFrame({"run_root": pd.Series(dtype=object)})
        for output in ("", "   ", Path()):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    nnt.load_sweep_outputs(valid_empty, output)

    def test_preserves_manifest_copy_columns_order_and_duplicate_index(self):
        run_a, _ = self.make_run("run_a", with_bias=False)
        run_b, _ = self.make_run("run_b", with_bias=False)
        self.write_text(run_a / "quantity.dat")
        self.write_text(run_b / "quantity.dat")

        runs = pd.DataFrame(
            {
                "sweep_variable": ["V_PG", "V_PG"],
                "sweep_value": [2.0, -1.0],
                "run_root": [run_a, run_b],
                "run_name": ["run_a", "run_b"],
                "value_source": ["variables_input", "folder_name"],
                "bias_dir": [None, None],
                "complete": [True, False],
                "required_outputs": [{"old": None}, {"old": run_b / "old.dat"}],
                "outputs_available": [False, False],
                "error": ["discovery warning", None],
            },
            index=pd.Index([7, 7], name="manifest_row"),
        )
        before = runs.copy(deep=True)
        original_columns = runs.columns.tolist()

        result = nnt.load_sweep_outputs(
            runs,
            "quantity.dat",
            prefer_nextnanopy=False,
        )

        pd.testing.assert_frame_equal(runs, before)
        pd.testing.assert_frame_equal(result[original_columns], runs)
        self.assertEqual(result.index.tolist(), [7, 7])
        self.assertEqual(result["run_name"].tolist(), ["run_a", "run_b"])
        self.assertEqual(result.columns.tolist(), original_columns + OUTPUT_COLUMNS)
        self.assertEqual(result["error"].tolist(), ["discovery warning", None])
        self.assertEqual(result["outputs_available"].tolist(), [False, False])

    def test_relative_resolution_prefers_bias_then_falls_back_to_run_root(self):
        run_bias, bias_dir = self.make_run("bias_precedence")
        bias_output = self.write_text(
            bias_dir / "quantity.dat",
            "x[nm] Value[eV]\n0 1\n",
        )
        self.write_text(
            run_bias / "quantity.dat",
            "x[nm] Value[eV]\n0 99\n",
        )

        run_fallback, fallback_bias = self.make_run("run_fallback")
        fallback_output = self.write_text(
            run_fallback / "quantity.dat",
            "x[nm] Value[eV]\n0 2\n",
        )

        run_no_bias, _ = self.make_run("no_bias", with_bias=False)
        no_bias_output = self.write_text(
            run_no_bias / "quantity.dat",
            "x[nm] Value[eV]\n0 3\n",
        )

        runs = pd.DataFrame(
            {
                "run_root": [run_bias, run_fallback, run_no_bias],
                "bias_dir": [bias_dir, fallback_bias, None],
            }
        )

        result = nnt.load_sweep_outputs(
            runs,
            "quantity.dat",
            prefer_nextnanopy=False,
        )

        expected_paths = [
            bias_output.resolve(),
            fallback_output.resolve(),
            no_bias_output.resolve(),
        ]
        self.assertEqual(result["output_path"].tolist(), expected_paths)
        self.assertTrue(result["output_available"].all())
        self.assertEqual(
            [
                float(dataset.get_variable("Value").value[0])
                for dataset in result["dataset"]
            ],
            [1.0, 2.0, 3.0],
        )
        for output_path, dataset in zip(result["output_path"], result["dataset"]):
            self.assertIsInstance(output_path, Path)
            self.assertEqual(dataset.path, output_path)

    def test_run_root_fallback_works_without_bias_column(self):
        run_root, _ = self.make_run("run_without_bias_column", with_bias=False)
        output_path = self.write_text(run_root / "quantity.dat")
        runs = pd.DataFrame({"run_root": [run_root]})

        result = nnt.load_sweep_outputs(
            runs,
            "quantity.dat",
            prefer_nextnanopy=False,
        )

        self.assertEqual(result.iloc[0]["output_path"], output_path.resolve())
        self.assertIsInstance(result.iloc[0]["dataset"], nnt.OutputDataset)

    def test_absolute_output_uses_only_the_exact_path(self):
        run_root, bias_dir = self.make_run("absolute")
        self.write_text(run_root / "quantity.dat", "x Value\n0 1\n")
        self.write_text(bias_dir / "quantity.dat", "x Value\n0 2\n")
        absolute_output = self.write_text(
            self.temporary_root / "shared.dat",
            "x Value\n0 7\n",
        )
        runs = pd.DataFrame({"run_root": [run_root], "bias_dir": [bias_dir]})

        result = nnt.load_sweep_outputs(
            runs,
            absolute_output,
            prefer_nextnanopy=False,
        )
        row = result.iloc[0]

        self.assertEqual(row["requested_output"], absolute_output)
        self.assertEqual(row["output_path"], absolute_output.resolve())
        self.assertEqual(float(row["dataset"].get_variable("Value").value[0]), 7.0)

    def test_missing_output_is_retained_and_search_is_not_recursive(self):
        run_root, bias_dir = self.make_run("missing")
        self.write_text(bias_dir / "nested" / "quantity.dat")
        self.write_text(run_root / "other" / "quantity.dat")
        runs = pd.DataFrame(
            {"run_root": [run_root], "bias_dir": [bias_dir]},
            index=[41],
        )

        result = nnt.load_sweep_outputs(
            runs,
            "quantity.dat",
            prefer_nextnanopy=False,
        )
        row = result.iloc[0]

        self.assertEqual(result.index.tolist(), [41])
        self.assertFalse(row["output_available"])
        self.assertIsNone(row["output_path"])
        self.assertIsNone(row["dataset"])
        self.assertIn("Output not found", row["load_error"])
        self.assertIn("quantity.dat", row["load_error"])

    def test_load_failure_is_retained_and_does_not_stop_other_rows(self):
        valid_root, _ = self.make_run("valid", with_bias=False)
        failed_root, _ = self.make_run("malformed", with_bias=False)
        missing_root, _ = self.make_run("missing_file", with_bias=False)
        valid_path = self.write_text(valid_root / "quantity.dat")
        failed_path = self.write_text(
            failed_root / "quantity.dat",
            "x Value\nnot-a-number invalid\n",
        )
        runs = pd.DataFrame(
            {
                "run_root": [valid_root, failed_root, missing_root],
                "bias_dir": [None, None, None],
            },
            index=[30, 20, 10],
        )

        result = nnt.load_sweep_outputs(
            runs,
            "quantity.dat",
            prefer_nextnanopy=False,
        )

        self.assertEqual(result.index.tolist(), [30, 20, 10])
        valid = result.iloc[0]
        self.assertEqual(valid["output_path"], valid_path.resolve())
        self.assertTrue(valid["output_available"])
        self.assertIsInstance(valid["dataset"], nnt.OutputDataset)
        self.assertIsNone(valid["load_error"])

        failed = result.iloc[1]
        self.assertEqual(failed["output_path"], failed_path.resolve())
        self.assertTrue(failed["output_available"])
        self.assertIsNone(failed["dataset"])
        self.assertIn("Failed to load", failed["load_error"])

        missing = result.iloc[2]
        self.assertFalse(missing["output_available"])
        self.assertIsNone(missing["output_path"])
        self.assertIsNone(missing["dataset"])
        self.assertIn("Output not found", missing["load_error"])

    def test_real_dat_load_and_variable_validation(self):
        run_root, _ = self.make_run("dat", with_bias=False)
        dat_path = self.write_text(run_root / "multi.dat")
        runs = pd.DataFrame({"run_root": [run_root], "bias_dir": [None]})

        selected = nnt.load_sweep_outputs(
            runs,
            "multi.dat",
            variable="Beta",
            prefer_nextnanopy=False,
        ).iloc[0]

        self.assertEqual(selected["dataset"].source, "custom_dat")
        self.assertEqual(selected["dataset"].path, dat_path.resolve())
        self.assertEqual(selected["dataset"].variable_names, ["Alpha", "Beta"])
        np.testing.assert_allclose(
            selected["dataset"].get_variable("Beta").value,
            [10.0, 20.0],
        )

        missing_variable = nnt.load_sweep_outputs(
            runs,
            "multi.dat",
            variable="Missing",
            prefer_nextnanopy=False,
        ).iloc[0]
        self.assertTrue(missing_variable["output_available"])
        self.assertEqual(missing_variable["output_path"], dat_path.resolve())
        self.assertIsNone(missing_variable["dataset"])
        self.assertIn("Failed to load", missing_variable["load_error"])
        self.assertIn("Missing", missing_variable["load_error"])

    def test_real_fld_load(self):
        run_root, bias_dir = self.make_run("fld")
        fld_path = self.write_text(
            bias_dir / "potential_2d_xz_test.fld",
            FLD_TEXT,
        )
        runs = pd.DataFrame({"run_root": [run_root], "bias_dir": [bias_dir]})

        row = nnt.load_sweep_outputs(
            runs,
            "potential_2d_xz_test.fld",
            variable="Beta",
            prefer_nextnanopy=False,
        ).iloc[0]

        dataset = row["dataset"]
        self.assertEqual(dataset.source, "custom_fld")
        self.assertEqual(dataset.path, fld_path.resolve())
        self.assertEqual(dataset.coord_names, ["x", "z"])
        np.testing.assert_allclose(dataset.get_coord("x").value, [0.0, 1.0])
        np.testing.assert_allclose(dataset.get_coord("z").value, [-1.0, 1.0])
        np.testing.assert_allclose(
            dataset.get_variable("Beta").value,
            [[10.0, 30.0], [20.0, 40.0]],
        )

    def test_real_ascii_vtr_load_from_nested_relative_path(self):
        run_root, bias_dir = self.make_run("vtr")
        relative_path = Path("Quantum/c-Ge_QW/HH/density.vtr")
        vtr_path = self.write_text(bias_dir / relative_path, VTR_TEXT)
        runs = pd.DataFrame({"run_root": [run_root], "bias_dir": [bias_dir]})

        row = nnt.load_sweep_outputs(
            runs,
            relative_path,
            variable="Beta",
            prefer_nextnanopy=False,
        ).iloc[0]

        dataset = row["dataset"]
        self.assertEqual(row["requested_output"], relative_path)
        self.assertEqual(row["output_path"], vtr_path.resolve())
        self.assertEqual(dataset.source, "custom_vtr")
        self.assertEqual(dataset.path, vtr_path.resolve())
        np.testing.assert_allclose(
            dataset.get_variable("Beta").value[:, 0, 0],
            [10.0, 20.0],
        )

    def test_empty_manifest_adds_columns_without_loading(self):
        runs = pd.DataFrame(
            {
                "run_root": pd.Series(dtype=object),
                "bias_dir": pd.Series(dtype=object),
                "error": pd.Series(dtype=object),
                "outputs_available": pd.Series(dtype=bool),
                "required_outputs": pd.Series(dtype=object),
            },
            index=pd.Index([], name="manifest_row"),
        )
        before = runs.copy(deep=True)
        original_columns = runs.columns.tolist()

        with patch.object(
            nnt,
            "load_output_file",
            side_effect=AssertionError("loader must not be called"),
        ) as loader:
            result = nnt.load_sweep_outputs(runs, "quantity.dat")

        loader.assert_not_called()
        pd.testing.assert_frame_equal(runs, before)
        self.assertTrue(result.empty)
        self.assertEqual(result.index.name, "manifest_row")
        self.assertEqual(result.columns.tolist(), original_columns + OUTPUT_COLUMNS)

    def test_prefer_nextnanopy_is_forwarded_and_function_is_public(self):
        run_root, _ = self.make_run("forwarding", with_bias=False)
        output_path = self.write_text(run_root / "quantity.dat")
        dataset = nnt.OutputDataset(
            path=output_path.resolve(),
            coords={
                "x": nnt.AxisData(
                    name="x",
                    value=np.asarray([0.0]),
                    label="x",
                )
            },
            variables={
                "Value": nnt.VariableData(
                    name="Value",
                    value=np.asarray([1.0]),
                    label="Value",
                )
            },
            source="test",
        )
        runs = pd.DataFrame({"run_root": [run_root], "bias_dir": [None]})

        with patch.object(
            nnt,
            "load_output_file",
            return_value=dataset,
        ) as loader:
            row = nnt.load_sweep_outputs(
                runs,
                "quantity.dat",
                variable="Value",
                prefer_nextnanopy=False,
            ).iloc[0]

        loader.assert_called_once_with(
            output_path.resolve(),
            prefer_nextnanopy=False,
        )
        self.assertIs(row["dataset"], dataset)
        self.assertIn("load_sweep_outputs", nnt.__all__)


if __name__ == "__main__":
    unittest.main()
