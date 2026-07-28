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


if __name__ == "__main__":
    unittest.main()
