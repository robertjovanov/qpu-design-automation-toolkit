import re
import tempfile
import unittest
from pathlib import Path

from qd_design import (
    AdaptiveZGridPolicy,
    LinearDotArrayDevice,
    build_adaptive_z_grid_lines,
    build_simulation_layout,
    make_reference_sige_ge_process_stack,
    make_sige_ge_process_stack,
    render_grid_block,
    write_nextnano_input_from_template,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUAL_TEMPLATE_PATH = (
    REPO_ROOT
    / "configs"
    / "robert_inputs"
    / "double_qd"
    / "3d"
    / "Double_Quantum_Dot_3D.in"
)
TRACKED_GENERATED_PATH = (
    REPO_ROOT
    / "configs"
    / "robert_inputs"
    / "generated"
    / "Double_Quantum_Dot_3D_from_PHIDL.in"
)


def _representative_device(*, include_screening_gates=False):
    """Match the maintained non-notebook generation workflow in notebook 03."""
    return LinearDotArrayDevice(
        name="double_dot_from_phidl",
        n_dots=2,
        device_y_size_nm=200.0,
        ohmic_width_nm=40.0,
        ohmic_length_nm=200.0,
        barrier_width_nm=40.0,
        barrier_length_nm=140.0,
        plunger_body_width_nm=40.0,
        plunger_body_length_nm=50.0,
        plunger_head_top_width_nm=60.0,
        plunger_head_max_width_nm=100.0,
        plunger_head_height_nm=100.0,
        plunger_upper_taper_height_nm=25.0,
        plunger_lower_taper_height_nm=25.0,
        ohmic_to_barrier_gap_nm=20.0,
        barrier_to_plunger_gap_nm=20.0,
        include_screening_gates=include_screening_gates,
    )


def _simulation_layout(
    *,
    process_stack=None,
    include_screening_gates=False,
    include_ohmics=True,
):
    device = _representative_device(
        include_screening_gates=include_screening_gates,
    )
    layout_spec = device.layout_spec()
    if not include_ohmics:
        layout_spec = [
            element
            for element in layout_spec
            if element["gate_type"] != "ohmic"
        ]

    return build_simulation_layout(
        name="double_dot_simulation_layout",
        layout_elements=layout_spec,
        process_stack=process_stack or make_reference_sige_ge_process_stack(),
        x_margin_nm=0.0,
        y_margin_nm=0.0,
    )


def _block_span(text, block_name):
    match = re.search(
        rf"(?m)^[ \t]*{re.escape(block_name)}[ \t]*\{{",
        text,
    )
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


def _without_grid_block(text):
    start, end = _block_span(text, "grid")
    return text[:start] + "<GRID_BLOCK>" + text[end:]


class AdaptiveNextnanoZGridTests(unittest.TestCase):
    def test_default_grid_has_the_expected_adaptive_hierarchy(self):
        layout = _simulation_layout()

        self.assertEqual(
            build_adaptive_z_grid_lines(layout),
            [
                (-4015.0, "$dz_buffer_coarse"),
                (-4010.0, "$dz_buffer_coarse"),
                (-1015.0, "$dz_buffer_medium"),
                (-165.0, "$dz_buffer_fine"),
                (-77.0, "$dz_oxide_gates_medium"),
                (-20.0, "$dz_QW_coarse"),
                (-15.0, "$dz_QW_fine"),
                (0.0, "$dz_QW_fine"),
                (5.0, "$dz_QW_coarse"),
                (100.0, "$dz_cap_fine"),
                (101.0, "$dz_cap_fine"),
                (108.0, "$dz_oxide_gates_medium"),
                (138.0, "$dz_oxide_gates_medium"),
                (143.0, "$dz_oxide_gates_medium"),
                (173.0, "$dz_oxide_gates_medium"),
            ],
        )

        grid_block = render_grid_block(layout)
        self.assertIn(
            "line{ pos = -4015 spacing = $dz_buffer_coarse }",
            grid_block,
        )
        self.assertNotIn(
            "line{ pos = -4015 spacing = $dz_oxide_gates_medium }",
            grid_block,
        )

    def test_qw_boundaries_and_margins_follow_the_actual_qw(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                ge_qw_thickness_nm=24.0,
            )
        )
        policy = AdaptiveZGridPolicy(quantum_margin_nm=7.0)
        grid = dict(build_adaptive_z_grid_lines(layout, policy=policy))

        self.assertEqual(grid[-31.0], "$dz_QW_coarse")
        self.assertEqual(grid[-24.0], "$dz_QW_fine")
        self.assertEqual(grid[0.0], "$dz_QW_fine")
        self.assertEqual(grid[7.0], "$dz_QW_coarse")
        self.assertNotIn(-20.0, grid)
        self.assertNotIn(-15.0, grid)
        self.assertNotIn(5.0, grid)

        rendered = render_grid_block(layout, z_grid_policy=policy)
        self.assertIn(
            "line{ pos = -31 spacing = $dz_QW_coarse }",
            rendered,
        )

    def test_buffer_extent_and_refinement_depths_follow_stack_and_policy(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                sige_buffer_thickness_nm=5200.0,
            )
        )
        default_grid = dict(build_adaptive_z_grid_lines(layout))

        # Extra buffer thickness moves the lower domain boundary while the
        # physical refinement depths remain anchored to the QW-facing interface.
        self.assertEqual(default_grid[-5215.0], "$dz_buffer_coarse")
        self.assertEqual(default_grid[-1015.0], "$dz_buffer_medium")
        self.assertEqual(default_grid[-165.0], "$dz_buffer_fine")
        self.assertNotIn(-4015.0, default_grid)

        policy = AdaptiveZGridPolicy(
            buffer_medium_refinement_depth_nm=1200.0,
            buffer_fine_refinement_depth_nm=225.0,
        )
        grid = dict(build_adaptive_z_grid_lines(layout, policy=policy))

        self.assertEqual(grid[-5215.0], "$dz_buffer_coarse")
        self.assertEqual(grid[-1215.0], "$dz_buffer_medium")
        self.assertEqual(grid[-240.0], "$dz_buffer_fine")
        self.assertNotIn(-1015.0, grid)
        self.assertNotIn(-165.0, grid)

    def test_cap_refinement_follows_cap_thickness_and_policy(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                sige_cap_thickness_nm=121.0,
            )
        )
        policy = AdaptiveZGridPolicy(cap_interface_fine_thickness_nm=2.5)
        grid = dict(build_adaptive_z_grid_lines(layout, policy=policy))

        self.assertEqual(grid[118.5], "$dz_cap_fine")
        self.assertEqual(grid[121.0], "$dz_cap_fine")
        self.assertEqual(grid[193.0], "$dz_oxide_gates_medium")
        self.assertNotIn(100.0, grid)
        self.assertNotIn(101.0, grid)

    def test_material_and_patterned_interfaces_follow_changed_gate_geometry(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                al2o3_total_thickness_nm=80.0,
                barrier_gate_bottom_z_nm=112.0,
                barrier_gate_thickness_nm=27.0,
                plunger_gate_bottom_z_nm=148.0,
                plunger_gate_thickness_nm=31.0,
                ohmic_depth_from_device_top_nm=275.0,
            )
        )
        grid = dict(build_adaptive_z_grid_lines(layout))

        for region in layout.background_regions:
            with self.subTest(background_region=region.name):
                self.assertIn(region.z_min_nm, grid)
                self.assertIn(region.z_max_nm, grid)

        for region in layout.patterned_regions:
            with self.subTest(patterned_region=region.name):
                self.assertEqual(
                    grid[region.z_min_nm],
                    "$dz_oxide_gates_medium",
                )
                self.assertEqual(
                    grid[region.z_max_nm],
                    "$dz_oxide_gates_medium",
                )

        self.assertIn(112.0, grid)
        self.assertIn(139.0, grid)
        self.assertIn(148.0, grid)
        self.assertIn(179.0, grid)
        self.assertIn(-94.0, grid)
        self.assertIn(181.0, grid)
        self.assertNotIn(108.0, grid)
        self.assertNotIn(143.0, grid)

    def test_screening_gate_interfaces_are_included(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                screening_gate_bottom_z_nm=121.0,
                screening_gate_thickness_nm=9.0,
            ),
            include_screening_gates=True,
        )
        grid = dict(build_adaptive_z_grid_lines(layout))

        screening_regions = [
            region
            for region in layout.patterned_regions
            if region.gate_type == "screening"
        ]
        self.assertEqual(len(screening_regions), 2)
        self.assertEqual(grid[121.0], "$dz_oxide_gates_medium")
        self.assertEqual(grid[130.0], "$dz_oxide_gates_medium")

    def test_layout_without_ohmics_keeps_a_valid_upper_buffer_grid(self):
        layout = _simulation_layout(include_ohmics=False)
        grid = dict(build_adaptive_z_grid_lines(layout))

        self.assertFalse(
            any(region.gate_type == "ohmic" for region in layout.patterned_regions)
        )
        self.assertNotIn(-77.0, grid)
        self.assertEqual(grid[-165.0], "$dz_buffer_fine")
        self.assertEqual(grid[-20.0], "$dz_QW_coarse")
        self.assertEqual(grid[-15.0], "$dz_QW_fine")

    def test_entries_are_sorted_in_domain_and_deduplicate_to_finest_spacing(self):
        layout = _simulation_layout()
        grid_lines = build_adaptive_z_grid_lines(layout)
        positions = [position for position, _ in grid_lines]
        grid = dict(grid_lines)

        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(positions), len(set(positions)))
        self.assertTrue(
            all(
                layout.domain.z_min_nm <= position <= layout.domain.z_max_nm
                for position in positions
            )
        )
        self.assertEqual(grid[0.0], "$dz_QW_fine")
        self.assertEqual(grid[101.0], "$dz_cap_fine")
        self.assertEqual(grid[173.0], "$dz_oxide_gates_medium")

    def test_representative_generation_changes_only_the_z_grid(self):
        layout = _simulation_layout()
        voltage_overrides = {
            "V_P1": -3.0,
            "V_P2": -3.0,
            "V_B1": 0.0,
            "V_B2": 0.0,
            "V_B3": 0.0,
            "V_OC_L": 0.0,
            "V_OC_R": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / "generated.in"
            write_nextnano_input_from_template(
                simulation_layout=layout,
                template_path=MANUAL_TEMPLATE_PATH,
                output_path=generated_path,
                voltage_overrides=voltage_overrides,
            )
            generated_text = generated_path.read_text(encoding="utf-8")

        tracked_text = TRACKED_GENERATED_PATH.read_text(encoding="utf-8")
        generated_grid = _extract_block(generated_text, "grid")
        tracked_grid = _extract_block(tracked_text, "grid")

        self.assertEqual(
            _extract_block(generated_grid, "xgrid"),
            _extract_block(tracked_grid, "xgrid"),
        )
        self.assertEqual(
            _extract_block(generated_grid, "ygrid"),
            _extract_block(tracked_grid, "ygrid"),
        )

        for block_name in ("output", "contacts", "structure", "quantum", "run"):
            with self.subTest(block=block_name):
                self.assertEqual(
                    _extract_block(generated_text, block_name),
                    _extract_block(tracked_text, block_name),
                )

        self.assertEqual(
            _without_grid_block(generated_text),
            _without_grid_block(tracked_text),
        )


if __name__ == "__main__":
    unittest.main()
