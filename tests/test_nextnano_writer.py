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
    render_structure_block,
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
    x_margin_nm=0.0,
    y_margin_nm=0.0,
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
        x_margin_nm=x_margin_nm,
        y_margin_nm=y_margin_nm,
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


def _without_blocks(text, block_names):
    result = text
    for block_name in block_names:
        start, end = _block_span(result, block_name)
        result = result[:start] + f"<{block_name.upper()}_BLOCK>" + result[end:]
    return result


def _region_after_comment(text, comment):
    comment_start = text.index(comment)
    suffix = text[comment_start:]
    start, end = _block_span(suffix, "region")
    return suffix[start:end]


def _background_region(layout, name):
    matches = [
        region for region in layout.background_regions if region.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one background region named {name!r}; found {len(matches)}."
        )
    return matches[0]


def _cuboid_bounds(region_block):
    bounds = {}
    for axis in ("x", "y", "z"):
        match = re.search(
            rf"(?m)^[ \t]*{axis}[ \t]*=[ \t]*"
            r"\[([-+0-9.eE]+),[ \t]*([-+0-9.eE]+)\]",
            region_block,
        )
        if match is None:
            raise AssertionError(f"Missing {axis}-bounds in region block.")
        bounds[axis] = (float(match.group(1)), float(match.group(2)))
    return bounds


class RemoveSurfaceChargeGeometryTests(unittest.TestCase):
    def test_default_contact_matches_the_complete_cap_and_domain(self):
        layout = _simulation_layout()
        quantum_well = _background_region(layout, "Ge_QW")
        cap = _background_region(layout, "SiGe_cap")
        dielectric = _background_region(layout, "Al2O3_dielectric")
        structure = render_structure_block(layout)
        cap_block = _region_after_comment(
            structure,
            "# background: SiGe_cap",
        )
        contact_block = _region_after_comment(
            structure,
            "# auxiliary fermi_hole contact: remove_surface_charge",
        )

        self.assertEqual(
            _cuboid_bounds(contact_block),
            {
                "x": (layout.domain.x_min_nm, layout.domain.x_max_nm),
                "y": (layout.domain.y_min_nm, layout.domain.y_max_nm),
                "z": (0.0, 101.0),
            },
        )
        self.assertEqual(
            _cuboid_bounds(contact_block),
            _cuboid_bounds(cap_block),
        )
        self.assertEqual(cap.z_min_nm, quantum_well.z_max_nm)
        self.assertEqual(cap.z_max_nm, dielectric.z_min_nm)
        self.assertIn(
            "contact{ name = remove_surface_charge }",
            contact_block,
        )

        # The auxiliary region overlaps the cap to assign the contact. It
        # neither replaces nor duplicates the modeled cap material.
        self.assertNotIn("binary{", contact_block)
        self.assertNotIn("ternary_constant{", contact_block)
        self.assertIn(
            'ternary_constant{ name = "Si(x)Ge(1-x)" alloy_x = 0.15 }',
            cap_block,
        )
        self.assertNotIn("contact{", cap_block)

    def test_contact_follows_configured_cap_thickness_and_qw_reference(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                ge_qw_thickness_nm=24.0,
                sige_cap_thickness_nm=121.0,
            )
        )
        quantum_well = _background_region(layout, "Ge_QW")
        cap = _background_region(layout, "SiGe_cap")
        dielectric = _background_region(layout, "Al2O3_dielectric")
        contact_block = _region_after_comment(
            render_structure_block(layout),
            "# auxiliary fermi_hole contact: remove_surface_charge",
        )

        self.assertEqual(
            (quantum_well.z_min_nm, quantum_well.z_max_nm),
            (-24.0, 0.0),
        )
        self.assertEqual((cap.z_min_nm, cap.z_max_nm), (0.0, 121.0))
        self.assertEqual(
            _cuboid_bounds(contact_block)["z"],
            (cap.z_min_nm, cap.z_max_nm),
        )
        self.assertEqual(cap.z_min_nm, quantum_well.z_max_nm)
        self.assertEqual(cap.z_max_nm, dielectric.z_min_nm)
        self.assertEqual(
            cap.z_max_nm - cap.z_min_nm,
            121.0,
        )

    def test_contact_follows_lateral_simulation_domain_bounds(self):
        layout = _simulation_layout(
            x_margin_nm=12.5,
            y_margin_nm=35.0,
        )
        cap = _background_region(layout, "SiGe_cap")
        contact_block = _region_after_comment(
            render_structure_block(layout),
            "# auxiliary fermi_hole contact: remove_surface_charge",
        )
        contact_bounds = _cuboid_bounds(contact_block)

        self.assertEqual(
            contact_bounds["x"],
            (layout.domain.x_min_nm, layout.domain.x_max_nm),
        )
        self.assertEqual(
            contact_bounds["y"],
            (layout.domain.y_min_nm, layout.domain.y_max_nm),
        )
        self.assertEqual(
            contact_bounds,
            {
                "x": (-272.5, 272.5),
                "y": (-35.0, 235.0),
                "z": (cap.z_min_nm, cap.z_max_nm),
            },
        )

    def test_contact_requires_the_semantically_named_cap_region(self):
        layout = _simulation_layout()
        layout.background_regions = [
            region
            for region in layout.background_regions
            if region.name != "SiGe_cap"
        ]

        with self.assertRaisesRegex(
            ValueError,
            "exactly one background region named 'SiGe_cap'; found 0",
        ):
            render_structure_block(layout)


class BodyContactGeometryTests(unittest.TestCase):
    def test_default_body_contact_is_a_distinct_full_domain_sige_layer(self):
        layout = _simulation_layout()
        body = _background_region(layout, "SiGe_body_contact")
        buffer = _background_region(layout, "SiGe_buffer")

        self.assertEqual((body.z_min_nm, body.z_max_nm), (-4115.0, -4015.0))
        self.assertEqual(body.z_max_nm - body.z_min_nm, 100.0)
        self.assertEqual(
            (buffer.z_min_nm, buffer.z_max_nm),
            (-4015.0, -15.0),
        )
        self.assertEqual(buffer.z_max_nm - buffer.z_min_nm, 4000.0)

        self.assertEqual(body.z_max_nm, buffer.z_min_nm)
        self.assertLess(body.z_min_nm, buffer.z_min_nm)
        self.assertEqual(
            (
                body.x_min_nm,
                body.x_max_nm,
                body.y_min_nm,
                body.y_max_nm,
            ),
            (
                layout.domain.x_min_nm,
                layout.domain.x_max_nm,
                layout.domain.y_min_nm,
                layout.domain.y_max_nm,
            ),
        )
        self.assertEqual(body.material, buffer.material)
        self.assertEqual(body.alloy_x, buffer.alloy_x)
        self.assertEqual(body.contact_name, "Body")
        self.assertEqual(layout.domain.z_min_nm, body.z_min_nm)
        self.assertEqual(layout.domain.z_max_nm, 173.0)

        structure = render_structure_block(layout)
        body_block = _region_after_comment(
            structure,
            "# background: SiGe_body_contact",
        )
        self.assertIn("x = [-260, 260]", body_block)
        self.assertIn("y = [0, 200]", body_block)
        self.assertIn("z = [-4115, -4015]", body_block)
        self.assertIn(
            'ternary_constant{ name = "Si(x)Ge(1-x)" alloy_x = 0.15 }',
            body_block,
        )
        self.assertIn("contact{ name = Body }", body_block)

    def test_body_contact_thickness_changes_only_its_bottom_and_domain_minimum(self):
        default_layout = _simulation_layout()
        configured_layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                body_contact_thickness_nm=50.0,
            )
        )
        default_body = _background_region(
            default_layout,
            "SiGe_body_contact",
        )
        configured_body = _background_region(
            configured_layout,
            "SiGe_body_contact",
        )

        self.assertEqual(
            (configured_body.z_min_nm, configured_body.z_max_nm),
            (-4065.0, -4015.0),
        )
        self.assertEqual(configured_body.z_max_nm, default_body.z_max_nm)
        self.assertEqual(
            configured_body.z_min_nm,
            default_body.z_min_nm + 50.0,
        )
        self.assertEqual(
            configured_layout.domain.z_min_nm,
            configured_body.z_min_nm,
        )
        self.assertEqual(
            configured_layout.domain.z_max_nm,
            default_layout.domain.z_max_nm,
        )
        self.assertEqual(
            [
                region.to_dict()
                for region in configured_layout.background_regions
                if region.name != "SiGe_body_contact"
            ],
            [
                region.to_dict()
                for region in default_layout.background_regions
                if region.name != "SiGe_body_contact"
            ],
        )
        self.assertEqual(
            configured_layout.patterned_regions,
            default_layout.patterned_regions,
        )
        self.assertEqual(
            (
                configured_layout.domain.x_min_nm,
                configured_layout.domain.x_max_nm,
                configured_layout.domain.y_min_nm,
                configured_layout.domain.y_max_nm,
            ),
            (
                default_layout.domain.x_min_nm,
                default_layout.domain.x_max_nm,
                default_layout.domain.y_min_nm,
                default_layout.domain.y_max_nm,
            ),
        )

    def test_buffer_thickness_moves_the_body_contact_chain_consistently(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                sige_buffer_thickness_nm=5200.0,
            )
        )
        body = _background_region(layout, "SiGe_body_contact")
        buffer = _background_region(layout, "SiGe_buffer")

        self.assertEqual(
            (buffer.z_min_nm, buffer.z_max_nm),
            (-5215.0, -15.0),
        )
        self.assertEqual(
            (body.z_min_nm, body.z_max_nm),
            (-5315.0, -5215.0),
        )
        self.assertEqual(body.z_max_nm, buffer.z_min_nm)
        self.assertEqual(layout.domain.z_min_nm, body.z_min_nm)

    def test_qw_thickness_moves_the_buffer_and_body_chain_consistently(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                ge_qw_thickness_nm=24.0,
            )
        )
        body = _background_region(layout, "SiGe_body_contact")
        buffer = _background_region(layout, "SiGe_buffer")
        quantum_well = _background_region(layout, "Ge_QW")

        self.assertEqual(
            (quantum_well.z_min_nm, quantum_well.z_max_nm),
            (-24.0, 0.0),
        )
        self.assertEqual(
            (buffer.z_min_nm, buffer.z_max_nm),
            (-4024.0, -24.0),
        )
        self.assertEqual(
            (body.z_min_nm, body.z_max_nm),
            (-4124.0, -4024.0),
        )
        self.assertEqual(body.z_max_nm, buffer.z_min_nm)
        self.assertEqual(buffer.z_max_nm, quantum_well.z_min_nm)

    def test_body_contact_thickness_must_be_positive(self):
        for invalid_thickness in (0.0, -1.0):
            with self.subTest(body_contact_thickness_nm=invalid_thickness):
                with self.assertRaisesRegex(
                    ValueError,
                    "body_contact_thickness_nm must be positive",
                ):
                    make_sige_ge_process_stack(
                        body_contact_thickness_nm=invalid_thickness,
                    )

    def test_body_contact_reuses_the_configured_buffer_alloy(self):
        layout = _simulation_layout(
            process_stack=make_sige_ge_process_stack(
                sige_alloy_x=0.22,
            )
        )
        body = _background_region(layout, "SiGe_body_contact")
        buffer = _background_region(layout, "SiGe_buffer")

        self.assertEqual((body.material, body.alloy_x), ("SiGe", 0.22))
        self.assertEqual(
            (body.material, body.alloy_x),
            (buffer.material, buffer.alloy_x),
        )
        body_block = _region_after_comment(
            render_structure_block(layout),
            "# background: SiGe_body_contact",
        )
        self.assertIn("alloy_x = 0.22", body_block)


class AdaptiveNextnanoZGridTests(unittest.TestCase):
    def test_default_grid_has_the_expected_adaptive_hierarchy(self):
        layout = _simulation_layout()

        self.assertEqual(
            build_adaptive_z_grid_lines(layout),
            [
                (-4115.0, "$dz_buffer_coarse"),
                (-4015.0, "$dz_buffer_coarse"),
                (-1015.0, "$dz_buffer_medium"),
                (-165.0, "$dz_buffer_fine"),
                (-149.0, "$dz_oxide_gates_medium"),
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
        self.assertEqual(default_grid[-5315.0], "$dz_buffer_coarse")
        self.assertEqual(default_grid[-5215.0], "$dz_buffer_coarse")
        self.assertEqual(default_grid[-1015.0], "$dz_buffer_medium")
        self.assertEqual(default_grid[-165.0], "$dz_buffer_fine")
        self.assertNotIn(-4015.0, default_grid)

        policy = AdaptiveZGridPolicy(
            buffer_medium_refinement_depth_nm=1200.0,
            buffer_fine_refinement_depth_nm=225.0,
        )
        grid = dict(build_adaptive_z_grid_lines(layout, policy=policy))

        self.assertEqual(grid[-5315.0], "$dz_buffer_coarse")
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
        self.assertNotIn(-149.0, grid)
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

    def test_representative_generation_changes_only_expected_geometry(self):
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

        for block_name in ("output", "contacts", "run"):
            with self.subTest(block=block_name):
                self.assertEqual(
                    _extract_block(generated_text, block_name),
                    _extract_block(tracked_text, block_name),
                )

        self.assertEqual(
            _without_blocks(generated_text, ("structure", "grid", "quantum")),
            _without_blocks(tracked_text, ("structure", "grid", "quantum")),
        )

        unchanged_region_comments = [
            "# background: SiGe_buffer",
            "# background: Ge_QW",
            "# background: SiGe_cap",
            "# background: Al2O3_dielectric",
            *[
                f"# patterned region: {region.name}"
                for region in layout.patterned_regions
                if region.gate_type != "ohmic"
            ],
        ]
        for comment in unchanged_region_comments:
            with self.subTest(region=comment):
                self.assertEqual(
                    _region_after_comment(generated_text, comment),
                    _region_after_comment(tracked_text, comment),
                )

        old_surface_contact = _region_after_comment(
            tracked_text,
            "# auxiliary fermi_hole contact: remove_surface_charge",
        )
        new_surface_contact = _region_after_comment(
            generated_text,
            "# auxiliary fermi_hole contact: remove_surface_charge",
        )
        self.assertIn("z = [100, 101]", old_surface_contact)
        self.assertIn("z = [0, 101]", new_surface_contact)
        self.assertEqual(
            old_surface_contact.replace(
                "z = [100, 101]",
                "z = [0, 101]",
            ),
            new_surface_contact,
        )

        old_body = _region_after_comment(
            tracked_text,
            "# auxiliary contact: Body",
        )
        new_body = _region_after_comment(
            generated_text,
            "# background: SiGe_body_contact",
        )
        self.assertIn("z = [-4015, -4010]", old_body)
        self.assertNotIn("ternary_constant", old_body)
        self.assertIn("z = [-4115, -4015]", new_body)
        self.assertIn(
            'ternary_constant{ name = "Si(x)Ge(1-x)" alloy_x = 0.15 }',
            new_body,
        )
        self.assertIn("contact{ name = Body }", new_body)

        old_zero_fermi = _region_after_comment(
            tracked_text,
            "# auxiliary fermi_hole contact: zero_fermi_QW",
        )
        new_zero_fermi = _region_after_comment(
            generated_text,
            "# auxiliary fermi_hole contact: zero_fermi_QW",
        )
        self.assertEqual(_cuboid_bounds(old_zero_fermi)["z"], (-15.0, 0.0))
        self.assertEqual(_cuboid_bounds(new_zero_fermi)["z"], (-20.0, 5.0))

        for name in ("OC_L", "OC_R"):
            old_ohmic = _region_after_comment(
                tracked_text,
                f"# patterned region: {name}",
            )
            new_ohmic = _region_after_comment(
                generated_text,
                f"# patterned region: {name}",
            )
            self.assertIn("z = [-77, 173]", old_ohmic)
            self.assertEqual(
                old_ohmic.replace("z = [-77, 173]", "z = [-149, 173]"),
                new_ohmic,
            )

        quantum_bounds = _cuboid_bounds(
            _extract_block(generated_text, "quantum")
        )
        self.assertEqual(quantum_bounds["x"], (-220.0, 220.0))
        self.assertEqual(quantum_bounds["y"], (20.0, 180.0))
        self.assertEqual(quantum_bounds["z"], (-20.0, 5.0))

        self.assertIn(
            "line{ pos = -220 spacing = $dx_QD }",
            generated_grid,
        )
        self.assertIn(
            "line{ pos = 220 spacing = $dx_QD }",
            generated_grid,
        )
        self.assertNotIn("spacing = $dx_coarse", generated_grid)


if __name__ == "__main__":
    unittest.main()
