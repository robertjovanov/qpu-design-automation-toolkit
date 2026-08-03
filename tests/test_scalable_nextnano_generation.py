import copy
import re
import tempfile
import unittest
from pathlib import Path

from qd_design import (
    LinearDotArrayDevice,
    SquareDotArrayDevice,
    build_adaptive_z_grid_lines,
    build_lateral_grid_lines,
    build_simulation_layout,
    derive_active_device_bounds,
    derive_quantum_region,
    make_reference_sige_ge_process_stack,
    make_sige_ge_process_stack,
    render_structure_block,
    write_nextnano_input_from_template,
)
from qd_design.export.nextnano_writer import (
    render_output_block,
    render_quantum_block,
    render_run_block,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    REPO_ROOT
    / "configs"
    / "robert_inputs"
    / "double_qd"
    / "3d"
    / "Double_Quantum_Dot_3D.in"
)
TEMPLATE_SPACING_VALUES = {
    f"${name}": float(value)
    for name, value in re.findall(
        r"(?m)^\$(d[xyz][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)",
        TEMPLATE_PATH.read_text(encoding="utf-8"),
    )
}


def _double_dot(*, ohmic_gap_nm=1000.0):
    return LinearDotArrayDevice(
        name="scalable_double_dot",
        n_dots=2,
        ohmic_to_barrier_gap_nm=ohmic_gap_nm,
    )


def _layout(device, *, process_stack=None, layout_elements=None):
    return build_simulation_layout(
        name=f"{device.name}_simulation",
        layout_elements=(
            device.layout_spec() if layout_elements is None else layout_elements
        ),
        process_stack=(
            process_stack or make_reference_sige_ge_process_stack()
        ),
    )


def _block_span(text, block_name):
    match = re.search(rf"(?m)^[ \t]*{re.escape(block_name)}[ \t]*\{{", text)
    if match is None:
        raise AssertionError(f"Missing block: {block_name}")

    brace_start = text.find("{", match.start())
    depth = 0
    for index in range(brace_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1
    raise AssertionError(f"Unterminated block: {block_name}")


def _extract_block(text, block_name):
    start, end = _block_span(text, block_name)
    return text[start:end]


def _region_after_comment(text, comment):
    suffix = text[text.index(comment) :]
    start, end = _block_span(suffix, "region")
    return suffix[start:end]


def _xyz_bounds(text):
    bounds = {}
    for axis in "xyz":
        match = re.search(
            rf"(?m)^[ \t]*{axis}[ \t]*=[ \t]*"
            r"\[([-+0-9.eE]+),[ \t]*([-+0-9.eE]+)\]",
            text,
        )
        if match is None:
            raise AssertionError(f"Missing {axis}-bounds")
        bounds[axis] = (float(match.group(1)), float(match.group(2)))
    return bounds


def _background(layout, name):
    matches = [
        region for region in layout.background_regions if region.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one background named {name!r}")
    return matches[0]


def _vertices(region_text):
    return [
        (float(x), float(y))
        for x, y in re.findall(
            r"vertex\s*\{\s*x\s*=\s*\[([-+0-9.eE]+)\]"
            r"\s*y\s*=\s*\[([-+0-9.eE]+)\]",
            region_text,
            flags=re.DOTALL,
        )
    ]


def _write(layout, *, template_path=TEMPLATE_PATH, voltage_overrides=None):
    temporary_directory = tempfile.TemporaryDirectory()
    output_path = Path(temporary_directory.name) / "generated.in"
    write_nextnano_input_from_template(
        simulation_layout=layout,
        template_path=template_path,
        output_path=output_path,
        voltage_overrides=voltage_overrides,
    )
    return temporary_directory, output_path.read_text(encoding="utf-8")


def _assert_grid_invariants(test_case, layout, grid_lines, axis):
    positions = [position for position, _ in grid_lines]
    lower = getattr(layout.domain, f"{axis}_min_nm")
    upper = getattr(layout.domain, f"{axis}_max_nm")
    test_case.assertEqual(positions, sorted(positions))
    test_case.assertEqual(len(positions), len(set(positions)))
    test_case.assertTrue(all(lower <= position <= upper for position in positions))
    for _, spacing in grid_lines:
        test_case.assertIn(spacing, TEMPLATE_SPACING_VALUES)
        test_case.assertGreater(TEMPLATE_SPACING_VALUES[spacing], 0.0)


class StandardProcessAndDqdRegressionTests(unittest.TestCase):
    def setUp(self):
        self.device = _double_dot()
        self.layout = _layout(self.device)

    def test_standard_vertical_geometry_is_process_derived_and_shared(self):
        body = _background(self.layout, "SiGe_body_contact")
        buffer = _background(self.layout, "SiGe_buffer")
        quantum_well = _background(self.layout, "Ge_QW")
        cap = _background(self.layout, "SiGe_cap")
        quantum = derive_quantum_region(self.layout)

        self.assertEqual((body.z_min_nm, body.z_max_nm), (-4115.0, -4015.0))
        self.assertEqual((buffer.z_min_nm, buffer.z_max_nm), (-4015.0, -15.0))
        self.assertEqual(body.z_max_nm, buffer.z_min_nm)
        self.assertEqual(body.contact_name, "Body")
        self.assertEqual((quantum_well.z_min_nm, quantum_well.z_max_nm), (-15.0, 0.0))
        self.assertEqual((cap.z_min_nm, cap.z_max_nm), (0.0, 101.0))
        self.assertEqual((quantum.z_min_nm, quantum.z_max_nm), (-20.0, 5.0))

        ohmics = [
            region
            for region in self.layout.patterned_regions
            if region.gate_type == "ohmic"
        ]
        self.assertEqual(len(ohmics), 2)
        self.assertTrue(
            all((region.z_min_nm, region.z_max_nm) == (-149.0, 173.0) for region in ohmics)
        )

        structure = render_structure_block(self.layout)
        surface = _xyz_bounds(
            _region_after_comment(
                structure,
                "# auxiliary fermi_hole contact: remove_surface_charge",
            )
        )
        zero_fermi = _xyz_bounds(
            _region_after_comment(
                structure,
                "# auxiliary fermi_hole contact: zero_fermi_QW",
            )
        )
        self.assertEqual(surface["z"], (cap.z_min_nm, cap.z_max_nm))
        self.assertEqual(
            (surface["x"], surface["y"]),
            (
                (self.layout.domain.x_min_nm, self.layout.domain.x_max_nm),
                (self.layout.domain.y_min_nm, self.layout.domain.y_max_nm),
            ),
        )
        self.assertEqual(zero_fermi["z"], (quantum.z_min_nm, quantum.z_max_nm))
        self.assertEqual(zero_fermi["x"], surface["x"])
        self.assertEqual(zero_fermi["y"], surface["y"])

    def test_validated_lateral_dqd_generation(self):
        domain = self.layout.domain
        active = derive_active_device_bounds(self.layout)
        quantum = derive_quantum_region(self.layout)
        self.assertEqual(
            (domain.x_min_nm, domain.x_max_nm, domain.y_min_nm, domain.y_max_nm),
            (-1240.0, 1240.0, 0.0, 200.0),
        )
        self.assertEqual((active.x_min_nm, active.x_max_nm), (-200.0, 200.0))
        self.assertEqual(
            quantum.to_dict(),
            {
                "x_min_nm": -220.0,
                "x_max_nm": 220.0,
                "y_min_nm": 20.0,
                "y_max_nm": 180.0,
                "z_min_nm": -20.0,
                "z_max_nm": 5.0,
            },
        )

        polygons_before = {
            region.name: copy.deepcopy(region.polygon_xy_nm)
            for region in self.layout.patterned_regions
            if region.gate_type == "plunger"
        }
        temporary_directory, generated = _write(
            self.layout,
            voltage_overrides={"V_P1": -1.65, "V_P2": -1.65},
        )
        self.addCleanup(temporary_directory.cleanup)

        self.assertRegex(generated, r"(?m)^\$V_P1\s*= -1\.65 # \(V\) generated$")
        self.assertRegex(generated, r"(?m)^\$V_P2\s*= -1\.65 # \(V\) generated$")
        self.assertEqual(
            _xyz_bounds(_extract_block(generated, "quantum")),
            {"x": (-220.0, 220.0), "y": (20.0, 180.0), "z": (-20.0, 5.0)},
        )

        x_lines, y_lines = build_lateral_grid_lines(self.layout)
        x_grid = dict(x_lines)
        y_grid = dict(y_lines)
        self.assertEqual(
            {
                spacing: TEMPLATE_SPACING_VALUES[spacing]
                for spacing in ("$dx", "$dx_QD", "$dx_coarse", "$dy", "$dy_QD")
            },
            {
                "$dx": 10.0,
                "$dx_QD": 5.0,
                "$dx_coarse": 100.0,
                "$dy": 5.0,
                "$dy_QD": 5.0,
            },
        )
        for position in (-1200.0, -350.0, 350.0, 1200.0):
            self.assertEqual(x_grid[position], "$dx_coarse")
        for position in (
            -220.0,
            -200.0,
            -180.0,
            -160.0,
            -140.0,
            -90.0,
            -40.0,
            -20.0,
            0.0,
            20.0,
            40.0,
            90.0,
            140.0,
            160.0,
            180.0,
            200.0,
            220.0,
        ):
            self.assertEqual(x_grid[position], "$dx_QD")
        self.assertEqual(x_grid[-1240.0], "$dx")
        self.assertEqual(x_grid[1240.0], "$dx")
        self.assertEqual(y_grid[0.0], "$dy_QD")
        self.assertEqual(y_grid[20.0], "$dy_QD")
        self.assertEqual(y_grid[180.0], "$dy_QD")
        self.assertEqual(y_grid[200.0], "$dy_QD")
        _assert_grid_invariants(self, self.layout, x_lines, "x")
        _assert_grid_invariants(self, self.layout, y_lines, "y")
        _assert_grid_invariants(
            self,
            self.layout,
            build_adaptive_z_grid_lines(self.layout),
            "z",
        )

        for region in self.layout.patterned_regions:
            if region.gate_type != "plunger":
                continue
            self.assertEqual(region.polygon_xy_nm, polygons_before[region.name])
            rendered_region = _region_after_comment(
                generated,
                f"# patterned region: {region.name}",
            )
            rendered_vertices = _vertices(rendered_region)
            self.assertEqual(len(rendered_vertices), len(region.polygon_xy_nm[0]))
            for rendered_vertex, source_vertex in zip(
                rendered_vertices,
                region.polygon_xy_nm[0],
            ):
                self.assertAlmostEqual(rendered_vertex[0], source_vertex[0])
                self.assertAlmostEqual(rendered_vertex[1], source_vertex[1])


class ChangedProcessStackTests(unittest.TestCase):
    def test_changed_stack_moves_every_dependent_vertical_coordinate(self):
        process_stack = make_sige_ge_process_stack(
            ge_qw_thickness_nm=24.0,
            sige_cap_thickness_nm=111.0,
            al2o3_total_thickness_nm=90.0,
            barrier_gate_bottom_z_nm=118.0,
            barrier_gate_thickness_nm=33.0,
            plunger_gate_bottom_z_nm=158.0,
            plunger_gate_thickness_nm=37.0,
        )
        layout = _layout(_double_dot(ohmic_gap_nm=20.0), process_stack=process_stack)
        quantum_well = _background(layout, "Ge_QW")
        cap = _background(layout, "SiGe_cap")
        quantum = derive_quantum_region(layout)
        ohmics = [
            region for region in layout.patterned_regions if region.gate_type == "ohmic"
        ]

        self.assertEqual((quantum_well.z_min_nm, quantum_well.z_max_nm), (-24.0, 0.0))
        self.assertEqual((quantum.z_min_nm, quantum.z_max_nm), (-29.0, 5.0))
        self.assertEqual((cap.z_min_nm, cap.z_max_nm), (0.0, 111.0))
        self.assertEqual(layout.domain.z_max_nm, 201.0)
        self.assertTrue(
            all((region.z_min_nm, region.z_max_nm) == (-139.0, 201.0) for region in ohmics)
        )

        structure = render_structure_block(layout)
        zero_fermi = _xyz_bounds(
            _region_after_comment(
                structure,
                "# auxiliary fermi_hole contact: zero_fermi_QW",
            )
        )
        surface = _xyz_bounds(
            _region_after_comment(
                structure,
                "# auxiliary fermi_hole contact: remove_surface_charge",
            )
        )
        self.assertEqual(zero_fermi["z"], (-29.0, 5.0))
        self.assertEqual(surface["z"], (0.0, 111.0))
        self.assertIn('section2D{ name = "xy_QW" z = -12 }', render_output_block(layout))

        z_grid = dict(build_adaptive_z_grid_lines(layout))
        for position in (-139.0, -29.0, -24.0, 5.0, 110.0, 111.0, 118.0, 151.0, 158.0, 195.0, 201.0):
            self.assertIn(position, z_grid)
        for old_literal in (-149.0, -20.0, -15.0, 100.0, 101.0, 108.0, 138.0, 143.0, 173.0):
            self.assertNotIn(old_literal, z_grid)

    def test_legacy_ohmic_parameter_keeps_device_top_semantics(self):
        stack = make_sige_ge_process_stack(ohmic_depth_from_device_top_nm=275.0)
        ohmic_rule = stack.gate_rule_for_layer_name("ohmic")
        self.assertEqual((ohmic_rule.z_min_nm, ohmic_rule.z_max_nm), (-102.0, 173.0))

        with self.assertRaisesRegex(ValueError, "Specify only one"):
            make_sige_ge_process_stack(
                ohmic_penetration_from_semiconductor_surface_nm=250.0,
                ohmic_depth_from_device_top_nm=275.0,
            )


class GeneralityAndConfigurationTests(unittest.TestCase):
    def test_existing_square_array_uses_same_policy_without_long_gap(self):
        device = SquareDotArrayDevice(name="square_2x2_scalable")
        layout = _layout(device)
        quantum = derive_quantum_region(layout)
        self.assertEqual(
            (
                layout.domain.x_min_nm,
                layout.domain.x_max_nm,
                layout.domain.y_min_nm,
                layout.domain.y_max_nm,
            ),
            (-260.0, 260.0, 0.0, 400.0),
        )
        self.assertEqual(
            quantum.to_dict(),
            {
                "x_min_nm": -220.0,
                "x_max_nm": 220.0,
                "y_min_nm": 20.0,
                "y_max_nm": 380.0,
                "z_min_nm": -20.0,
                "z_max_nm": 5.0,
            },
        )
        ohmics = [
            region for region in layout.patterned_regions if region.gate_type == "ohmic"
        ]
        self.assertEqual(len(ohmics), 4)
        self.assertTrue(
            all(
                region.x_max_nm <= quantum.x_min_nm
                or region.x_min_nm >= quantum.x_max_nm
                for region in ohmics
            )
        )

        x_lines, y_lines = build_lateral_grid_lines(layout)
        self.assertNotIn("$dx_coarse", {spacing for _, spacing in x_lines})
        _assert_grid_invariants(self, layout, x_lines, "x")
        _assert_grid_invariants(self, layout, y_lines, "y")
        temporary_directory, generated = _write(layout)
        self.addCleanup(temporary_directory.cleanup)
        self.assertEqual(
            _xyz_bounds(_extract_block(generated, "quantum")),
            {"x": (-220.0, 220.0), "y": (20.0, 380.0), "z": (-20.0, 5.0)},
        )

    def test_translated_asymmetric_layout_is_not_origin_or_symmetry_dependent(self):
        device = _double_dot(ohmic_gap_nm=20.0)
        translated = copy.deepcopy(device.layout_spec())
        dx_nm = 137.0
        dy_nm = -43.0
        for element in translated:
            extra_x_nm = 500.0 if element["name"] == "OC_R" else 0.0
            for key in ("x_min_nm", "x_max_nm", "center_x_nm"):
                element[key] += dx_nm + extra_x_nm
            for key in ("y_min_nm", "y_max_nm", "center_y_nm"):
                element[key] += dy_nm
            element["polygon_xy_nm"] = [
                [
                    (x + dx_nm + extra_x_nm, y + dy_nm)
                    for x, y in polygon
                ]
                for polygon in element["polygon_xy_nm"]
            ]

        layout = _layout(device, layout_elements=translated)
        active = derive_active_device_bounds(layout)
        quantum = derive_quantum_region(layout)
        self.assertEqual((active.x_min_nm, active.x_max_nm), (-63.0, 337.0))
        self.assertEqual(
            quantum.to_dict(),
            {
                "x_min_nm": -83.0,
                "x_max_nm": 357.0,
                "y_min_nm": -23.0,
                "y_max_nm": 137.0,
                "z_min_nm": -20.0,
                "z_max_nm": 5.0,
            },
        )
        self.assertNotEqual(
            0.5 * (quantum.x_min_nm + quantum.x_max_nm),
            0.5 * (layout.domain.x_min_nm + layout.domain.x_max_nm),
        )
        x_grid = dict(build_lateral_grid_lines(layout)[0])
        self.assertEqual(x_grid[487.0], "$dx_coarse")
        self.assertEqual(x_grid[857.0], "$dx_coarse")
        self.assertNotIn(-213.0, x_grid)

    def test_layout_without_ohmics_clamps_quantum_padding_and_stays_valid(self):
        device = _double_dot(ohmic_gap_nm=20.0)
        no_ohmics = [
            element
            for element in device.layout_spec()
            if element["gate_type"] != "ohmic"
        ]
        layout = _layout(device, layout_elements=no_ohmics)
        quantum = derive_quantum_region(layout)
        self.assertEqual(
            (quantum.x_min_nm, quantum.x_max_nm),
            (layout.domain.x_min_nm, layout.domain.x_max_nm),
        )
        x_lines, _ = build_lateral_grid_lines(layout)
        self.assertNotIn("$dx_coarse", {spacing for _, spacing in x_lines})
        _assert_grid_invariants(self, layout, x_lines, "x")

    def test_remote_y_ohmic_clips_quantum_box_without_device_names(self):
        device = _double_dot(ohmic_gap_nm=20.0)
        layout_elements = copy.deepcopy(device.layout_spec())
        top_ohmic = copy.deepcopy(
            next(element for element in layout_elements if element["name"] == "OC_L")
        )
        top_ohmic["name"] = "TOP_CONTACT"
        top_ohmic["voltage_label"] = "V_TOP_CONTACT"
        for key in ("x_min_nm", "x_max_nm", "center_x_nm"):
            top_ohmic[key] += 240.0
        for key in ("y_min_nm", "y_max_nm", "center_y_nm"):
            top_ohmic[key] += 400.0
        top_ohmic["polygon_xy_nm"] = [
            [(x + 240.0, y + 400.0) for x, y in polygon]
            for polygon in top_ohmic["polygon_xy_nm"]
        ]
        layout_elements.append(top_ohmic)

        layout = _layout(device, layout_elements=layout_elements)
        quantum = derive_quantum_region(layout)
        self.assertEqual(
            (
                layout.domain.y_min_nm,
                layout.domain.y_max_nm,
                quantum.y_min_nm,
                quantum.y_max_nm,
            ),
            (0.0, 600.0, 20.0, 400.0),
        )
        excluded_regions = [
            region for region in layout.patterned_regions if region.gate_type == "ohmic"
        ]
        self.assertEqual(len(excluded_regions), 3)
        self.assertTrue(
            all(
                not (
                    max(quantum.x_min_nm, region.x_min_nm)
                    < min(quantum.x_max_nm, region.x_max_nm)
                    and max(quantum.y_min_nm, region.y_min_nm)
                    < min(quantum.y_max_nm, region.y_max_nm)
                )
                for region in excluded_regions
            )
        )

    def test_standalone_z_grid_does_not_require_active_lateral_gates(self):
        device = _double_dot(ohmic_gap_nm=20.0)
        only_ohmics = [
            element
            for element in device.layout_spec()
            if element["gate_type"] == "ohmic"
        ]
        layout = _layout(device, layout_elements=only_ohmics)
        z_grid = dict(build_adaptive_z_grid_lines(layout))
        self.assertEqual(z_grid[-20.0], "$dz_QW_coarse")
        self.assertEqual(z_grid[-15.0], "$dz_QW_fine")
        self.assertEqual(z_grid[5.0], "$dz_QW_coarse")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "preserved_blocks.in"
            write_nextnano_input_from_template(
                simulation_layout=layout,
                template_path=TEMPLATE_PATH,
                output_path=output_path,
                replace_output=False,
                replace_contacts=False,
                replace_structure=True,
                replace_grid=False,
                replace_quantum=False,
                replace_run=False,
            )
            self.assertTrue(output_path.is_file())
            generated = output_path.read_text(encoding="utf-8")
            zero_fermi = _xyz_bounds(
                _region_after_comment(
                    generated,
                    "# auxiliary fermi_hole contact: zero_fermi_QW",
                )
            )
            self.assertEqual(zero_fermi["z"], (-20.0, 5.0))

    def test_voltage_and_solver_modes_remain_run_configuration(self):
        layout = _layout(_double_dot(ohmic_gap_nm=20.0))
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            self_consistent_template = temporary_directory / "self_consistent.in"
            self_consistent_template.write_text(
                re.sub(
                    r"(?m)^\$quantum_poisson\s*=\s*0\s*$",
                    "$quantum_poisson = 1",
                    template,
                ),
                encoding="utf-8",
            )
            poisson_path = temporary_directory / "poisson.in"
            self_consistent_path = temporary_directory / "quantum_poisson.in"
            write_nextnano_input_from_template(
                simulation_layout=layout,
                template_path=TEMPLATE_PATH,
                output_path=poisson_path,
                voltage_overrides={"V_P1": -2.25, "V_P2": -2.25},
            )
            write_nextnano_input_from_template(
                simulation_layout=layout,
                template_path=self_consistent_template,
                output_path=self_consistent_path,
                voltage_overrides={"V_P1": -1.65, "V_P2": -1.65},
            )
            poisson_text = poisson_path.read_text(encoding="utf-8")
            self_consistent_text = self_consistent_path.read_text(encoding="utf-8")

        expected_poisson = {
            "strain": 1,
            "poisson": 1,
            "quantum": 0,
            "quantum_poisson": 0,
        }
        expected_self_consistent = {**expected_poisson, "quantum_poisson": 1}
        for text, expected in (
            (poisson_text, expected_poisson),
            (self_consistent_text, expected_self_consistent),
        ):
            actual = {
                name: int(
                    re.search(rf"(?m)^\${name}\s*=\s*([01])\s*$", text).group(1)
                )
                for name in expected
            }
            self.assertEqual(actual, expected)
            self.assertEqual(_extract_block(text, "run"), render_run_block())

        self.assertRegex(poisson_text, r"(?m)^\$V_P1\s*= -2\.25 # \(V\) generated$")
        self.assertRegex(
            self_consistent_text,
            r"(?m)^\$V_P1\s*= -1\.65 # \(V\) generated$",
        )
        self.assertEqual(
            _extract_block(poisson_text, "quantum"),
            _extract_block(self_consistent_text, "quantum"),
        )


if __name__ == "__main__":
    unittest.main()
