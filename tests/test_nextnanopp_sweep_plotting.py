import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure as MatplotlibFigure

import nextnanopp_tools as nnt


class SweepPlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def make_dataset(
        tag,
        *,
        x=(0.0, 1.0, 2.0),
        y=(1.0, 2.0, 3.0),
        coord_name="x",
        coord_unit="nm",
        variable_name="Potential",
        variable_unit="eV",
        extra_variables=None,
    ):
        variables = {
            variable_name: nnt.VariableData(
                name=variable_name,
                value=np.asarray(y),
                unit=variable_unit,
                label=f"{variable_name}[{variable_unit}]",
            )
        }
        for name, values, unit in extra_variables or []:
            variables[name] = nnt.VariableData(
                name=name,
                value=np.asarray(values),
                unit=unit,
                label=f"{name}[{unit}]",
            )
        return nnt.OutputDataset(
            path=Path(f"{tag}.dat"),
            coords={
                coord_name: nnt.AxisData(
                    name=coord_name,
                    value=np.asarray(x),
                    unit=coord_unit,
                    label=f"{coord_name}[{coord_unit}]",
                )
            },
            variables=variables,
            source="synthetic-test",
        )

    @staticmethod
    def make_outputs(
        sweep_values,
        datasets,
        *,
        sweep_variable="V_PG",
        include_sweep_variable=True,
        index=None,
    ):
        data = {
            "sweep_value": list(sweep_values),
            "dataset": list(datasets),
        }
        if include_sweep_variable:
            data["sweep_variable"] = [sweep_variable] * len(datasets)
        return pd.DataFrame(data, index=index)

    def test_rejects_non_dataframe_and_missing_required_columns(self):
        with self.assertRaisesRegex(TypeError, "pandas DataFrame"):
            nnt.plot_sweep_lines([])

        with self.assertRaisesRegex(ValueError, "sweep_value"):
            nnt.plot_sweep_lines(pd.DataFrame({"dataset": []}))

        with self.assertRaisesRegex(ValueError, "dataset"):
            nnt.plot_sweep_lines(pd.DataFrame({"sweep_value": []}))

    def test_rejects_empty_and_all_failed_manifests(self):
        empty = pd.DataFrame(
            {
                "sweep_value": pd.Series(dtype=float),
                "dataset": pd.Series(dtype=object),
            }
        )
        with self.assertRaisesRegex(ValueError, "No successfully loaded datasets"):
            nnt.plot_sweep_lines(empty)

        failed = self.make_outputs([0.0, 1.0], [None, None])
        with self.assertRaisesRegex(ValueError, "No successfully loaded datasets"):
            nnt.plot_sweep_lines(failed)

    def test_failed_rows_are_ignored_without_mutating_inputs_or_arrays(self):
        first = self.make_dataset("first", y=(1.0, 3.0, 5.0))
        last = self.make_dataset("last", y=(2.0, 4.0, 6.0))
        outputs = self.make_outputs(
            [2.0, "invalid-but-failed", -1.0],
            [first, None, last],
            index=[7, 7, 9],
        )
        original_columns = outputs.columns.tolist()
        original_index = outputs.index.copy()
        original_sweep_values = outputs["sweep_value"].tolist()
        original_dataset_ids = [id(value) for value in outputs["dataset"]]
        original_arrays = {
            id(dataset): (
                np.asarray(dataset.coords["x"].value).copy(),
                np.asarray(dataset.variables["Potential"].value).copy(),
            )
            for dataset in (first, last)
        }

        figure = nnt.plot_sweep_lines(outputs)

        self.assertEqual(len(figure.axes[0].lines), 2)
        self.assertEqual(outputs.columns.tolist(), original_columns)
        self.assertTrue(outputs.index.equals(original_index))
        self.assertEqual(outputs["sweep_value"].tolist(), original_sweep_values)
        self.assertEqual(
            [id(value) for value in outputs["dataset"]],
            original_dataset_ids,
        )
        for dataset in (first, last):
            x_before, y_before = original_arrays[id(dataset)]
            np.testing.assert_array_equal(dataset.coords["x"].value, x_before)
            np.testing.assert_array_equal(
                dataset.variables["Potential"].value,
                y_before,
            )

    def test_automatic_and_explicit_variable_selection(self):
        single = self.make_dataset("single", y=(3.0, 4.0, 5.0))
        automatic = nnt.plot_sweep_lines(self.make_outputs([0.0], [single]))
        np.testing.assert_array_equal(
            automatic.axes[0].lines[0].get_ydata(),
            [3.0, 4.0, 5.0],
        )

        multiple = self.make_dataset(
            "multiple",
            y=(1.0, 2.0, 3.0),
            extra_variables=[("Charge", (10.0, 20.0, 30.0), "C")],
        )
        with self.assertRaisesRegex(ValueError, "specify variable"):
            nnt.plot_sweep_lines(self.make_outputs([1.0], [multiple]))

        explicit = nnt.plot_sweep_lines(
            self.make_outputs([1.0], [multiple]),
            variable="Charge",
        )
        np.testing.assert_array_equal(
            explicit.axes[0].lines[0].get_ydata(),
            [10.0, 20.0, 30.0],
        )
        self.assertEqual(explicit.axes[0].get_ylabel(), "Charge[C]")

        with self.assertRaisesRegex(ValueError, "exactly") as context:
            nnt.plot_sweep_lines(
                self.make_outputs([1.0], [multiple]),
                variable="Poten",
            )
        self.assertIn("sweep value 1", str(context.exception))

    def test_missing_variable_error_identifies_sweep_value(self):
        first = self.make_dataset("first")
        second = self.make_dataset(
            "second",
            variable_name="Charge",
            variable_unit="C",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"Potential.*sweep value 2",
        ):
            nnt.plot_sweep_lines(
                self.make_outputs([1.0, 2.0], [first, second]),
                variable="Potential",
            )

    def test_default_order_is_numeric(self):
        ten = self.make_dataset("ten", y=(10.0, 10.0, 10.0))
        minus_two = self.make_dataset("minus_two", y=(-2.0, -2.0, -2.0))
        one = self.make_dataset("one", y=(1.0, 1.0, 1.0))
        figure = nnt.plot_sweep_lines(
            self.make_outputs([10.0, -2.0, 1.0], [ten, minus_two, one])
        )

        lines = figure.axes[0].lines
        self.assertEqual(
            [line.get_label() for line in lines],
            ["V_PG = -2", "V_PG = 1", "V_PG = 10"],
        )
        self.assertEqual([line.get_ydata()[0] for line in lines], [-2.0, 1.0, 10.0])

    def test_explicit_filter_preserves_order_and_uses_numeric_tolerance(self):
        minus_one = self.make_dataset("minus_one", y=(-1.0, -1.0, -1.0))
        point_three = self.make_dataset("point_three", y=(0.3, 0.3, 0.3))
        two = self.make_dataset("two", y=(2.0, 2.0, 2.0))
        figure = nnt.plot_sweep_lines(
            self.make_outputs(
                [-1.0, 0.3, 2.0],
                [minus_one, point_three, two],
            ),
            values=[2.0, 0.1 + 0.2],
        )

        lines = figure.axes[0].lines
        self.assertEqual(
            [line.get_label() for line in lines],
            ["V_PG = 2", "V_PG = 0.3"],
        )
        self.assertEqual([line.get_ydata()[0] for line in lines], [2.0, 0.3])

    def test_missing_and_ambiguous_requested_values_are_rejected(self):
        dataset = self.make_dataset("only")
        outputs = self.make_outputs([2.0], [dataset])
        with self.assertRaisesRegex(ValueError, r"8.*9"):
            nnt.plot_sweep_lines(outputs, values=[2.0, 8.0, 9.0])

        duplicate_outputs = self.make_outputs(
            [1.0, 1.0 + 1e-10],
            [self.make_dataset("a"), self.make_dataset("b")],
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            nnt.plot_sweep_lines(duplicate_outputs, values=[1.0])

        with self.assertRaisesRegex(ValueError, "already matched"):
            nnt.plot_sweep_lines(outputs, values=[2.0, 2.0])

        with self.assertRaisesRegex(ValueError, "at least one"):
            nnt.plot_sweep_lines(outputs, values=[])

    def test_rejects_non_1d_datasets_and_invalid_array_shapes(self):
        two_dimensional = nnt.OutputDataset(
            path=Path("two_dimensional.fld"),
            coords={
                "x": nnt.AxisData("x", np.asarray([0.0, 1.0]), "nm"),
                "z": nnt.AxisData("z", np.asarray([-1.0, 0.0]), "nm"),
            },
            variables={
                "Potential": nnt.VariableData(
                    "Potential",
                    np.zeros((2, 2)),
                    "eV",
                )
            },
        )
        zero_dimensional = nnt.OutputDataset(
            path=Path("zero_dimensional.dat"),
            coords={},
            variables={
                "Potential": nnt.VariableData(
                    "Potential",
                    np.asarray([1.0]),
                    "eV",
                )
            },
        )
        bad_coordinate = self.make_dataset(
            "bad_coordinate",
            x=np.asarray([[0.0, 1.0]]),
            y=(1.0, 2.0),
        )
        bad_variable = self.make_dataset(
            "bad_variable",
            x=(0.0, 1.0),
            y=np.asarray([[1.0, 2.0]]),
        )
        bad_length = self.make_dataset(
            "bad_length",
            x=(0.0, 1.0, 2.0),
            y=(1.0, 2.0),
        )

        cases = [
            (two_dimensional, "one-dimensional"),
            (zero_dimensional, "one-dimensional"),
            (bad_coordinate, "Coordinate values"),
            (bad_variable, "Variable values"),
            (bad_length, "lengths differ"),
        ]
        for dataset, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message) as context:
                    nnt.plot_sweep_lines(
                        self.make_outputs([4.0], [dataset])
                    )
                self.assertIn("sweep value 4", str(context.exception))

    def test_rejects_incompatible_axis_and_variable_metadata(self):
        base = self.make_dataset("base")
        incompatible = [
            (
                self.make_dataset("coord_name", coord_name="z"),
                "coordinate",
            ),
            (
                self.make_dataset("coord_unit", coord_unit="um"),
                "coordinate",
            ),
            (
                self.make_dataset("variable_name", variable_name="Charge"),
                "variable",
            ),
            (
                self.make_dataset("variable_unit", variable_unit="meV"),
                "variable",
            ),
        ]
        for dataset, message in incompatible:
            with self.subTest(message=message, path=dataset.path):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"Incompatible {message}.*sweep value 7",
                ):
                    nnt.plot_sweep_lines(
                        self.make_outputs([0.0, 7.0], [base, dataset])
                    )

    def test_different_coordinate_grids_are_preserved_without_interpolation(self):
        first = self.make_dataset(
            "first",
            x=(0.0, 1.0, 2.0),
            y=(1.0, 2.0, 3.0),
        )
        second = self.make_dataset(
            "second",
            x=(0.0, 0.5, 1.5, 3.0),
            y=(4.0, 5.0, 6.0, 7.0),
        )
        figure = nnt.plot_sweep_lines(
            self.make_outputs([0.0, 1.0], [first, second])
        )

        lines = figure.axes[0].lines
        np.testing.assert_array_equal(lines[0].get_xdata(), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(
            lines[1].get_xdata(),
            [0.0, 0.5, 1.5, 3.0],
        )

    def test_static_figure_labels_title_limits_and_no_output_actions(self):
        outputs = self.make_outputs(
            [-3.0, 1.0],
            [self.make_dataset("a"), self.make_dataset("b")],
        )
        with (
            patch("matplotlib.pyplot.show") as show,
            patch.object(MatplotlibFigure, "savefig") as savefig,
            patch("matplotlib.style.use") as style_use,
        ):
            figure = nnt.plot_sweep_lines(
                outputs,
                title="Potential sweep",
                xlim=(-0.5, 2.5),
                ylim=(-2.0, 4.0),
            )

        self.assertIsInstance(figure, MatplotlibFigure)
        self.assertEqual(len(figure.axes), 1)
        axes = figure.axes[0]
        self.assertEqual(
            [line.get_label() for line in axes.lines],
            ["V_PG = -3", "V_PG = 1"],
        )
        self.assertIsNotNone(axes.get_legend())
        self.assertEqual(axes.get_xlabel(), "x[nm]")
        self.assertEqual(axes.get_ylabel(), "Potential[eV]")
        self.assertEqual(axes.get_title(), "Potential sweep")
        np.testing.assert_allclose(axes.get_xlim(), (-0.5, 2.5))
        np.testing.assert_allclose(axes.get_ylim(), (-2.0, 4.0))
        show.assert_not_called()
        savefig.assert_not_called()
        style_use.assert_not_called()

        fallback = nnt.plot_sweep_lines(
            self.make_outputs(
                [3.0],
                [self.make_dataset("fallback")],
                include_sweep_variable=False,
            )
        )
        self.assertEqual(fallback.axes[0].lines[0].get_label(), "sweep = 3")

    def test_interactive_figure_matches_static_and_is_not_displayed_or_written(self):
        outputs = self.make_outputs(
            [2.0, 0.0],
            [
                self.make_dataset("two", y=(20.0, 21.0, 22.0)),
                self.make_dataset("zero", y=(0.0, 1.0, 2.0)),
            ],
        )
        static = nnt.plot_sweep_lines(
            outputs,
            values=[2.0, 0.0],
            title="Selected sweep",
            xlim=(-1.0, 3.0),
            ylim=(-5.0, 25.0),
        )
        with (
            patch.object(
                nnt,
                "_display_plotly_figure",
                side_effect=AssertionError("must not display"),
            ),
            patch.object(go.Figure, "show", side_effect=AssertionError("must not show")),
            patch.object(
                go.Figure,
                "write_html",
                side_effect=AssertionError("must not write"),
            ),
            patch.object(
                go.Figure,
                "write_image",
                side_effect=AssertionError("must not write"),
            ),
        ):
            interactive = nnt.plot_sweep_lines(
                outputs,
                values=[2.0, 0.0],
                interactive=True,
                title="Selected sweep",
                xlim=(-1.0, 3.0),
                ylim=(-5.0, 25.0),
            )

        self.assertIsInstance(interactive, go.Figure)
        lines = static.axes[0].lines
        self.assertEqual(len(lines), len(interactive.data))
        self.assertEqual(
            [line.get_label() for line in lines],
            [trace.name for trace in interactive.data],
        )
        for line, trace in zip(lines, interactive.data, strict=True):
            np.testing.assert_array_equal(line.get_xdata(), np.asarray(trace.x))
            np.testing.assert_array_equal(line.get_ydata(), np.asarray(trace.y))
            self.assertIn(trace.name.split(" = ")[0], trace.hovertemplate)
        self.assertEqual(interactive.layout.title.text, "Selected sweep")
        self.assertEqual(interactive.layout.xaxis.title.text, "x[nm]")
        self.assertEqual(interactive.layout.yaxis.title.text, "Potential[eV]")
        self.assertEqual(tuple(interactive.layout.xaxis.range), (-1.0, 3.0))
        self.assertEqual(tuple(interactive.layout.yaxis.range), (-5.0, 25.0))

    def test_reference_subtraction_is_per_curve_and_does_not_mutate_data(self):
        first = self.make_dataset(
            "first",
            x=(0.0, 1.0, 2.0),
            y=(5.0, 8.0, 13.0),
        )
        second = self.make_dataset(
            "second",
            x=(0.0, 1.5, 3.0),
            y=(10.0, 14.0, 19.0),
        )
        original_first = first.variables["Potential"].value.copy()
        original_second = second.variables["Potential"].value.copy()
        outputs = self.make_outputs([0.0, 1.0], [first, second])

        static = nnt.plot_sweep_lines(outputs, reference_coord=1.4)
        interactive = nnt.plot_sweep_lines(
            outputs,
            reference_coord=1.4,
            interactive=True,
        )

        expected = ([-3.0, 0.0, 5.0], [-4.0, 0.0, 5.0])
        for line, expected_y in zip(static.axes[0].lines, expected, strict=True):
            np.testing.assert_array_equal(line.get_ydata(), expected_y)
        for trace, expected_y in zip(interactive.data, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(trace.y), expected_y)
        self.assertEqual(
            [line.get_gid() for line in static.axes[0].lines],
            ["reference_coord=1", "reference_coord=1.5"],
        )
        self.assertEqual(
            [trace.meta["reference_coord"] for trace in interactive.data],
            [1.0, 1.5],
        )
        self.assertIn("nearest reference", interactive.data[0].hovertemplate)
        self.assertIn("relative", static.axes[0].get_ylabel().lower())
        self.assertIn("relative", interactive.layout.yaxis.title.text.lower())
        np.testing.assert_array_equal(
            first.variables["Potential"].value,
            original_first,
        )
        np.testing.assert_array_equal(
            second.variables["Potential"].value,
            original_second,
        )

    def test_rejects_invalid_dataset_and_numeric_metadata(self):
        with self.assertRaisesRegex(TypeError, r"sweep value 1.*OutputDataset"):
            nnt.plot_sweep_lines(self.make_outputs([1.0], [object()]))

        with self.assertRaisesRegex(ValueError, "must be numeric"):
            nnt.plot_sweep_lines(
                self.make_outputs(["not-a-number"], [self.make_dataset("bad")])
            )

        with self.assertRaisesRegex(ValueError, "reference_coord must be finite"):
            nnt.plot_sweep_lines(
                self.make_outputs([0.0], [self.make_dataset("reference")]),
                reference_coord=np.inf,
            )

    def test_public_export_and_docstring(self):
        self.assertIn("plot_sweep_lines", nnt.__all__)
        docstring = nnt.plot_sweep_lines.__doc__ or ""
        for phrase in (
            "load_sweep_outputs",
            "variable",
            "values",
            "reference_coord",
            "Matplotlib",
            "Plotly",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, docstring)
