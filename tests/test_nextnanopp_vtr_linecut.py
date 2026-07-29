import tempfile
import unittest
from pathlib import Path

import numpy as np

import nextnanopp_tools as nnt


SYNTHETIC_VTR = """<VTKFile type="RectilinearGrid" version="0.1" format="ascii">
<RectilinearGrid WholeExtent="1 3 1 2 1 2">
<Piece Extent="1 3 1 2 1 2">
<Coordinates>
<DataArray type="Float64" Name="X_COORDINATES" format="ascii">0 1 2</DataArray>
<DataArray type="Float64" Name="Y_COORDINATES" format="ascii">0 10</DataArray>
<DataArray type="Float64" Name="Z_COORDINATES" format="ascii">-1 1</DataArray>
</Coordinates>
<PointData>
<DataArray type="Float64" Name="Value[arb]" format="ascii">
0 1 2 10 11 12 100 101 102 110 111 112
</DataArray>
</PointData>
</Piece>
</RectilinearGrid>
</VTKFile>
"""


class VtrLinecutTests(unittest.TestCase):
    def test_loads_contiguous_x_line_without_full_volume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "synthetic.vtr"
            path.write_text(SYNTHETIC_VTR, encoding="utf-8")

            line = nnt.load_vtr_linecut(
                path,
                variable="Value",
                axis="x",
                fixed_coords={"y": 10.0, "z": 1.0},
            )

        np.testing.assert_allclose(line["axis"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(line["values"], [110.0, 111.0, 112.0])
        self.assertEqual(line["chosen_coords"], {"y": 10.0, "z": 1.0})
        self.assertEqual(line["chosen_indices"], {"y": 1, "z": 1})

    def test_loads_noncontiguous_y_plane_with_potential_unit_label(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "synthetic.vtr"
            path.write_text(
                SYNTHETIC_VTR.replace("Value[arb]", "Potential[V]"),
                encoding="utf-8",
            )

            plane = nnt.load_vtr_plane(
                path,
                variable="Potential",
                slice_axis="y",
                slice_value=10.0,
            )

        np.testing.assert_allclose(plane["x"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(plane["y"], [-1.0, 1.0])
        np.testing.assert_allclose(
            plane["values"],
            [[10.0, 110.0], [11.0, 111.0], [12.0, 112.0]],
        )
        self.assertEqual(plane["variable"].name, "Potential")
        self.assertEqual(plane["variable"].unit, "V")
        self.assertEqual(plane["variable"].label, "Potential[V]")
        self.assertEqual(plane["slice_coordinate"], 10.0)

    def test_loads_noncontiguous_y_plane_with_hh_unit_label(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "synthetic.vtr"
            path.write_text(
                SYNTHETIC_VTR.replace("Value[arb]", "HH[eV]"),
                encoding="utf-8",
            )

            plane = nnt.load_vtr_plane(
                path,
                variable="HH",
                slice_axis="y",
                slice_value=10.0,
            )

        np.testing.assert_allclose(
            plane["values"],
            [[10.0, 110.0], [11.0, 111.0], [12.0, 112.0]],
        )
        self.assertEqual(plane["variable"].name, "HH")
        self.assertEqual(plane["variable"].unit, "eV")
        self.assertEqual(plane["variable"].label, "HH[eV]")

    def test_loads_noncontiguous_z_line_with_hh_unit_label(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "synthetic.vtr"
            path.write_text(
                SYNTHETIC_VTR.replace("Value[arb]", "HH[eV]"),
                encoding="utf-8",
            )

            line = nnt.load_vtr_linecut(
                path,
                variable="HH",
                axis="z",
                fixed_coords={"x": 1.0, "y": 10.0},
            )

        np.testing.assert_allclose(line["axis"], [-1.0, 1.0])
        np.testing.assert_allclose(line["values"], [11.0, 111.0])
        self.assertEqual(line["variable"].name, "HH")
        self.assertEqual(line["variable"].unit, "eV")
        self.assertEqual(line["variable"].label, "HH[eV]")
        self.assertEqual(line["chosen_coords"], {"x": 1.0, "y": 10.0})
        self.assertEqual(line["chosen_indices"], {"x": 1, "y": 1})


if __name__ == "__main__":
    unittest.main()
