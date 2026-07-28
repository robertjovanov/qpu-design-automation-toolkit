import unittest

import numpy as np

from qd_design import (
    SquareDotArrayDevice,
    TopBarrierLinearDotArrayDevice,
    TwoDDotArrayDevice,
    build_simulation_layout,
    make_reference_sige_ge_process_stack,
)


def _elements_of_type(builder, gate_type):
    return [
        element
        for element in builder.layout_spec()
        if element["gate_type"] == gate_type
    ]


class TwoDDotArrayDeviceTests(unittest.TestCase):
    def test_2x3_is_composed_from_two_same_side_1x3_rows(self):
        array = TwoDDotArrayDevice(n_columns=3)
        device = array.ensure_built()

        self.assertEqual(array.n_dots, 6)
        self.assertIsInstance(
            array.row_builders["bottom"],
            TopBarrierLinearDotArrayDevice,
        )
        self.assertIsInstance(
            array.row_builders["top"],
            TopBarrierLinearDotArrayDevice,
        )
        self.assertEqual(array.refs["bottom_row"].rotation % 360, 180)
        self.assertEqual(array.refs["top_row"].rotation % 360, 0)
        np.testing.assert_allclose(
            device.bbox,
            [[-350.0, 0.0], [350.0, 400.0]],
            atol=1e-12,
        )

        spec = array.layout_spec()
        self.assertEqual(len(spec), 18)
        self.assertEqual(len(_elements_of_type(array, "plunger")), 6)
        self.assertEqual(len(_elements_of_type(array, "barrier")), 8)
        self.assertEqual(len(_elements_of_type(array, "ohmic")), 4)

        names = [element["name"] for element in spec]
        voltage_labels = [element["voltage_label"] for element in spec]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(voltage_labels), len(set(voltage_labels)))
        self.assertEqual(
            {element["name"] for element in _elements_of_type(array, "plunger")},
            {f"P{index}" for index in range(1, 7)},
        )
        self.assertEqual(
            {element["name"] for element in _elements_of_type(array, "barrier")},
            {f"B{index}" for index in range(1, 9)},
        )
        self.assertEqual(
            {element["name"] for element in _elements_of_type(array, "ohmic")},
            {"OC_L_1", "OC_R_1", "OC_L_2", "OC_R_2"},
        )

    def test_rows_keep_plungers_and_barriers_on_the_same_outer_side(self):
        array = TwoDDotArrayDevice(n_columns=3)
        elements = {element["name"]: element for element in array.layout_spec()}

        for name in ("P1", "P2", "P3", "B1", "B2", "B3", "B4"):
            self.assertEqual(elements[name]["y_min_nm"], 0.0)
        for name in ("P4", "P5", "P6", "B5", "B6", "B7", "B8"):
            self.assertEqual(elements[name]["y_max_nm"], 400.0)

    def test_row_gap_is_inserted_between_the_two_complete_rows(self):
        array = TwoDDotArrayDevice(n_columns=3, row_gap_nm=25.0)
        elements = {element["name"]: element for element in array.layout_spec()}

        self.assertEqual(array.row_pitch_y_nm, 225.0)
        self.assertEqual(array.facing_plunger_gap_nm, 125.0)
        self.assertEqual(elements["OC_R_1"]["y_max_nm"], 200.0)
        self.assertEqual(elements["OC_R_2"]["y_min_nm"], 225.0)
        self.assertEqual(elements["P6"]["y_max_nm"], 425.0)

    def test_layout_spec_matches_the_row_transforms_point_for_point(self):
        array = TwoDDotArrayDevice(n_columns=3)
        array.ensure_built()
        combined = {element["name"]: element for element in array.layout_spec()}

        bottom_source = sorted(
            _elements_of_type(array.row_builders["bottom"], "plunger"),
            key=lambda element: element["center_x_nm"],
            reverse=True,
        )
        top_source = sorted(
            _elements_of_type(array.row_builders["top"], "plunger"),
            key=lambda element: element["center_x_nm"],
        )

        for index, source in enumerate(bottom_source, start=1):
            expected = [
                [(-x, array.device_y_size_nm - y) for x, y in polygon]
                for polygon in source["polygon_xy_nm"]
            ]
            np.testing.assert_allclose(
                combined[f"P{index}"]["polygon_xy_nm"],
                expected,
                atol=1e-12,
            )

        for index, source in enumerate(top_source, start=4):
            expected = [
                [(x, y + array.row_pitch_y_nm) for x, y in polygon]
                for polygon in source["polygon_xy_nm"]
            ]
            np.testing.assert_allclose(
                combined[f"P{index}"]["polygon_xy_nm"],
                expected,
                atol=1e-12,
            )

    def test_2x2_geometry_matches_the_existing_square_array(self):
        composed = {
            element["name"]: element
            for element in TwoDDotArrayDevice(n_columns=2).layout_spec()
        }
        existing = {
            element["name"]: element
            for element in SquareDotArrayDevice().layout_spec()
        }

        self.assertEqual(set(composed), set(existing))
        for name in composed:
            with self.subTest(name=name):
                for bound in ("x_min_nm", "x_max_nm", "y_min_nm", "y_max_nm"):
                    self.assertAlmostEqual(
                        composed[name][bound],
                        existing[name][bound],
                    )

                composed_vertices = sorted(composed[name]["polygon_xy_nm"][0])
                existing_vertices = sorted(existing[name]["polygon_xy_nm"][0])
                np.testing.assert_allclose(
                    composed_vertices,
                    existing_vertices,
                    atol=1e-12,
                )

    def test_2x3_simulation_layout_keeps_all_unique_polygonal_regions(self):
        simulation = build_simulation_layout(
            name="two_d_dot_array_test_3d",
            layout_elements=TwoDDotArrayDevice(n_columns=3).layout_spec(),
            process_stack=make_reference_sige_ge_process_stack(),
            x_margin_nm=0.0,
            y_margin_nm=0.0,
        )

        self.assertEqual(len(simulation.patterned_regions), 18)
        self.assertEqual(
            len({region.name for region in simulation.patterned_regions}),
            18,
        )
        self.assertEqual(
            len({region.voltage_label for region in simulation.patterned_regions}),
            18,
        )
        plunger_vertex_counts = [
            [len(polygon) for polygon in region.polygon_xy_nm]
            for region in simulation.patterned_regions
            if region.gate_type == "plunger"
        ]
        self.assertEqual(plunger_vertex_counts, [[10]] * 6)

    def test_optional_screening_gates_are_relabelled_per_physical_dot(self):
        array = TwoDDotArrayDevice(
            n_columns=3,
            include_screening_gates=True,
        )
        screening = _elements_of_type(array, "screening")

        self.assertEqual([element["name"] for element in screening], [
            "SG1",
            "SG2",
            "SG3",
            "SG4",
            "SG5",
            "SG6",
        ])
        self.assertEqual(
            [element["voltage_label"] for element in screening],
            [f"V_SG{index}" for index in range(1, 7)],
        )
        for index, element in enumerate(screening, start=1):
            self.assertIn(f"P{index}", element["notes"])

    def test_invalid_row_geometry_is_rejected(self):
        invalid_builders = [
            TwoDDotArrayDevice(n_columns=0),
            TwoDDotArrayDevice(n_columns=1.5),
            TwoDDotArrayDevice(n_columns=True),
            TwoDDotArrayDevice(row_gap_nm=-1.0),
            TwoDDotArrayDevice(plunger_head_height_nm=160.0),
            TwoDDotArrayDevice(barrier_length_nm=201.0),
            TwoDDotArrayDevice(ohmic_length_nm=201.0),
        ]

        for builder in invalid_builders:
            with self.subTest(builder=builder):
                with self.assertRaises(ValueError):
                    builder.ensure_built()


if __name__ == "__main__":
    unittest.main()
