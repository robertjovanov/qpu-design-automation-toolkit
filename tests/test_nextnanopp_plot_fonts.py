import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import nextnanopp_tools as nnt


BROWSER_PLOT_APIS = (
    "plot_bias_volume_3d",
    "plot_bias_volume_linecut",
    "plot_bias_volume_slice",
    "plot_convergence",
    "plot_integrated_density_hole",
    "plot_quantum_density_volume_3d",
    "plot_quantum_density_volume_linecut",
    "plot_quantum_density_volume_slice",
    "plot_quantum_energy_spectrum",
    "plot_quantum_occupation",
    "plot_quantum_probability_volume_3d",
    "plot_quantum_probability_volume_linecut",
    "plot_quantum_probability_volume_slice",
    "plot_total_charges",
    "plot_vtr_slice",
)


class PlotFontTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def convergence_table():
        return pd.DataFrame(
            {
                "Iteration": [1, 2, 3],
                "Poisson Residual": [1.0, 0.1, 0.01],
                "Quantum Residual": [0.5, 0.05, 0.005],
            }
        )

    @staticmethod
    def volume_dataset():
        return nnt.OutputDataset(
            path=Path("synthetic.vtr"),
            coords={
                "x": nnt.AxisData(
                    name="x",
                    value=np.asarray([0.0, 1.0]),
                    unit="nm",
                    label="x[nm]",
                ),
                "y": nnt.AxisData(
                    name="y",
                    value=np.asarray([0.0, 2.0]),
                    unit="nm",
                    label="y[nm]",
                ),
                "z": nnt.AxisData(
                    name="z",
                    value=np.asarray([-1.0, 1.0]),
                    unit="nm",
                    label="z[nm]",
                ),
            },
            variables={
                "Potential": nnt.VariableData(
                    name="Potential",
                    value=np.arange(8.0).reshape(2, 2, 2),
                    unit="eV",
                    label="Potential[eV]",
                )
            },
            source="synthetic-test",
        )

    @staticmethod
    def axis_font_sizes(axes):
        legend = axes.get_legend()
        return {
            "title": axes.title.get_fontsize(),
            "xlabel": axes.xaxis.label.get_fontsize(),
            "ylabel": axes.yaxis.label.get_fontsize(),
            "xticks": tuple(label.get_fontsize() for label in axes.get_xticklabels()),
            "yticks": tuple(label.get_fontsize() for label in axes.get_yticklabels()),
            "legend": ()
            if legend is None
            else tuple(text.get_fontsize() for text in legend.get_texts()),
        }

    def test_browser_plot_apis_have_backward_compatible_default(self):
        self.assertIn("apply_plot_font_scale", nnt.__all__)
        helper_parameter = inspect.signature(nnt.apply_plot_font_scale).parameters[
            "font_scale"
        ]
        self.assertEqual(helper_parameter.default, 1.0)
        self.assertEqual(helper_parameter.kind, inspect.Parameter.KEYWORD_ONLY)

        for api_name in BROWSER_PLOT_APIS:
            with self.subTest(api=api_name):
                parameter = inspect.signature(getattr(nnt, api_name)).parameters[
                    "font_scale"
                ]
                self.assertEqual(parameter.default, 1.0)
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_omitted_scale_matches_explicit_one_without_mutating_rcparams(self):
        rc_keys = (
            "font.size",
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
            "figure.titlesize",
        )
        rc_before = {key: matplotlib.rcParams[key] for key in rc_keys}

        with patch.object(
            nnt,
            "read_convergence_table",
            return_value=self.convergence_table(),
        ):
            omitted = nnt.plot_convergence("unused-bias-directory")
            explicit = nnt.plot_convergence(
                "unused-bias-directory",
                font_scale=1.0,
            )

        omitted_axes = omitted.axes[0]
        explicit_axes = explicit.axes[0]
        self.assertEqual(
            self.axis_font_sizes(omitted_axes),
            self.axis_font_sizes(explicit_axes),
        )
        self.assertEqual(omitted_axes.title.get_fontsize(), 12.0)
        self.assertEqual(omitted_axes.xaxis.label.get_fontsize(), 10.0)
        self.assertTrue(
            all(label.get_fontsize() == 10.0 for label in omitted_axes.get_xticklabels())
        )
        self.assertTrue(
            all(
                text.get_fontsize() == 10.0
                for text in omitted_axes.get_legend().get_texts()
            )
        )
        self.assertEqual(len(omitted_axes.lines), len(explicit_axes.lines))
        for omitted_line, explicit_line in zip(
            omitted_axes.lines,
            explicit_axes.lines,
        ):
            np.testing.assert_array_equal(
                omitted_line.get_xdata(),
                explicit_line.get_xdata(),
            )
            np.testing.assert_array_equal(
                omitted_line.get_ydata(),
                explicit_line.get_ydata(),
            )
        self.assertEqual(
            {key: matplotlib.rcParams[key] for key in rc_keys},
            rc_before,
        )

    def test_font_scale_must_be_positive_and_finite(self):
        figure, _ = plt.subplots()
        for invalid in (0.0, -0.5, np.nan, np.inf, -np.inf):
            with self.subTest(font_scale=invalid):
                with self.assertRaisesRegex(ValueError, "font_scale"):
                    nnt.apply_plot_font_scale(figure, font_scale=invalid)

        with patch.object(
            nnt,
            "read_convergence_table",
            return_value=self.convergence_table(),
        ):
            with self.assertRaisesRegex(ValueError, "font_scale"):
                nnt.plot_convergence(
                    "unused-bias-directory",
                    font_scale=0.0,
                )

    def test_scaled_matplotlib_title_labels_ticks_and_legend(self):
        with patch.object(
            nnt,
            "read_convergence_table",
            return_value=self.convergence_table(),
        ):
            figure = nnt.plot_convergence(
                "unused-bias-directory",
                font_scale=1.5,
            )

        axes = figure.axes[0]
        self.assertEqual(axes.title.get_fontsize(), 18.0)
        self.assertEqual(axes.xaxis.label.get_fontsize(), 15.0)
        self.assertEqual(axes.yaxis.label.get_fontsize(), 15.0)
        self.assertTrue(
            all(label.get_fontsize() == 15.0 for label in axes.get_xticklabels())
        )
        self.assertTrue(
            all(label.get_fontsize() == 15.0 for label in axes.get_yticklabels())
        )
        self.assertTrue(
            all(
                text.get_fontsize() == 15.0
                for text in axes.get_legend().get_texts()
            )
        )

    def test_scaled_matplotlib_colorbar_and_linecut_annotation(self):
        dataset = self.volume_dataset()
        with patch.object(nnt, "_load_vtr_dataset", return_value=dataset):
            slice_figure = nnt.plot_vtr_slice(
                "synthetic.vtr",
                variable="Potential",
                slice_axis="y",
                slice_value=0.0,
                font_scale=1.5,
            )
            linecut_figure = nnt.plot_vtr_linecut(
                "synthetic.vtr",
                variable="Potential",
                axis="x",
                fixed_coords={"y": 0.0, "z": -1.0},
                font_scale=1.5,
            )

        plot_axes, colorbar_axes = slice_figure.axes
        self.assertEqual(plot_axes.title.get_fontsize(), 18.0)
        self.assertEqual(plot_axes.xaxis.label.get_fontsize(), 15.0)
        self.assertTrue(
            all(label.get_fontsize() == 15.0 for label in plot_axes.get_xticklabels())
        )
        self.assertEqual(colorbar_axes.yaxis.label.get_fontsize(), 15.0)
        self.assertTrue(
            all(
                label.get_fontsize() == 15.0
                for label in colorbar_axes.get_yticklabels()
            )
        )

        linecut_axes = linecut_figure.axes[0]
        self.assertEqual(len(linecut_axes.texts), 1)
        self.assertEqual(linecut_axes.texts[0].get_fontsize(), 13.5)

    def test_scaled_plotly_3d_scene_and_colorbar_fonts_without_output(self):
        dataset = self.volume_dataset()
        with (
            patch.object(
                nnt,
                "resolve_bias_output_file",
                return_value=Path("synthetic.vtr"),
            ),
            patch.object(nnt, "_load_vtr_dataset", return_value=dataset),
            patch.object(
                nnt,
                "_display_plotly_figure",
                side_effect=lambda figure: figure,
            ) as display,
            patch.object(go.Figure, "show") as show,
            patch.object(go.Figure, "write_html") as write_html,
            patch.object(go.Figure, "write_image") as write_image,
        ):
            figure = nnt.plot_bias_volume_3d(
                "unused-run-directory",
                "potential",
                variable="Potential",
                title="Synthetic 3D volume",
                max_points=8,
                font_scale=1.5,
            )

        display.assert_called_once_with(figure)
        show.assert_not_called()
        write_html.assert_not_called()
        write_image.assert_not_called()
        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(figure.layout.font.size, 18.0)
        self.assertEqual(figure.layout.title.font.size, 25.5)
        for scene_axis in (
            figure.layout.scene.xaxis,
            figure.layout.scene.yaxis,
            figure.layout.scene.zaxis,
        ):
            self.assertEqual(scene_axis.title.font.size, 21.0)
            self.assertEqual(scene_axis.tickfont.size, 18.0)
        colorbar = figure.data[0].colorbar
        self.assertEqual(colorbar.title.font.size, 21.0)
        self.assertEqual(colorbar.tickfont.size, 18.0)


if __name__ == "__main__":
    unittest.main()
