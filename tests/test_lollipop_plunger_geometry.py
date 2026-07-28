import unittest

import numpy as np

from qd_design import (
    CANONICAL_LOLLIPOP_PLUNGER,
    LinearDotArrayDevice,
    SingleDotDevice,
    SquareDotArrayDevice,
    TopBarrierLinearDotArrayDevice,
    TwoDDotArrayDevice,
    VerticallyStackedDoubleDotDevice,
)
from qd_design.visualization import polygonal_prism_mesh_arrays


def _plunger_polygons(builder):
    return {
        element["name"]: element["polygon_xy_nm"]
        for element in builder.layout_spec()
        if element["gate_type"] == "plunger"
    }


class LollipopPlungerGeometryTests(unittest.TestCase):
    def test_device_builders_share_canonical_plunger_defaults(self):
        canonical = CANONICAL_LOLLIPOP_PLUNGER
        builders = [
            SingleDotDevice(),
            LinearDotArrayDevice(),
            TopBarrierLinearDotArrayDevice(),
            TwoDDotArrayDevice(),
            VerticallyStackedDoubleDotDevice(),
            SquareDotArrayDevice(),
        ]

        for builder in builders:
            self.assertEqual(builder.plunger_body_width_nm, canonical.body_width_nm)
            self.assertEqual(builder.plunger_body_length_nm, canonical.body_length_nm)
            self.assertEqual(
                builder.plunger_head_top_width_nm,
                canonical.head_top_width_nm,
            )
            self.assertEqual(
                builder.plunger_head_max_width_nm,
                canonical.head_max_width_nm,
            )
            self.assertEqual(builder.plunger_head_height_nm, canonical.head_height_nm)
            self.assertEqual(
                builder.plunger_upper_taper_height_nm,
                canonical.upper_taper_height_nm,
            )
            self.assertEqual(
                builder.plunger_lower_taper_height_nm,
                canonical.lower_taper_height_nm,
            )

    def test_top_barrier_array_reuses_reference_lollipop_polygons_exactly(self):
        reference = _plunger_polygons(LinearDotArrayDevice(n_dots=2))
        top_barrier = _plunger_polygons(TopBarrierLinearDotArrayDevice(n_dots=2))

        self.assertEqual(top_barrier, reference)
        self.assertEqual([len(polygon) for polygon in top_barrier["P1"]], [10])
        self.assertEqual([len(polygon) for polygon in top_barrier["P2"]], [10])

    def test_only_barrier_approach_side_changes_between_linear_builders(self):
        reference = LinearDotArrayDevice(n_dots=2)
        top_barrier = TopBarrierLinearDotArrayDevice(n_dots=2)
        reference.ensure_built()
        top_barrier.ensure_built()

        self.assertEqual(
            [(ref.ymin, ref.ymax) for ref in reference.refs["barriers"]],
            [(0.0, 140.0)] * 3,
        )
        self.assertEqual(
            [(ref.ymin, ref.ymax) for ref in top_barrier.refs["barriers"]],
            [(60.0, 200.0)] * 3,
        )

    def test_3d_renderer_extrudes_the_lollipop_outline_not_its_bounding_box(self):
        polygon = _plunger_polygons(TopBarrierLinearDotArrayDevice(n_dots=2))["P1"][0]
        vertices, faces = polygonal_prism_mesh_arrays(polygon, 110.0, 115.0)

        self.assertEqual(vertices.shape, (20, 3))
        self.assertGreater(faces.shape[0], 12)
        self.assertEqual(len(np.unique(vertices[:10, 0])), 7)
        self.assertEqual(len(np.unique(vertices[:10, 1])), 5)


if __name__ == "__main__":
    unittest.main()
