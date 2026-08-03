from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .simulation_layout import (
    BackgroundRegion,
    PatternedRegion,
    QuantumRegion,
    QuantumRegionPolicy,
    SimulationLayout,
    derive_active_device_bounds,
    derive_quantum_region,
)


@dataclass(frozen=True)
class AdaptiveZGridPolicy:
    """Geometry-relative refinement distances for the nextnano z-grid."""

    quantum_margin_nm: float = 5.0
    buffer_medium_refinement_depth_nm: float = 1000.0
    buffer_fine_refinement_depth_nm: float = 150.0
    cap_interface_fine_thickness_nm: float = 1.0

    def __post_init__(self) -> None:
        distances = {
            "quantum_margin_nm": self.quantum_margin_nm,
            "buffer_medium_refinement_depth_nm": self.buffer_medium_refinement_depth_nm,
            "buffer_fine_refinement_depth_nm": self.buffer_fine_refinement_depth_nm,
            "cap_interface_fine_thickness_nm": self.cap_interface_fine_thickness_nm,
        }
        for name, value in distances.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        if (
            self.buffer_medium_refinement_depth_nm
            <= self.buffer_fine_refinement_depth_nm
        ):
            raise ValueError(
                "buffer_medium_refinement_depth_nm must be greater than "
                "buffer_fine_refinement_depth_nm."
            )


@dataclass(frozen=True)
class LateralMeshPolicy:
    """Geometry-aware lateral grid anchors and nextnano spacing variables."""

    active_transition_margin_nm: float = 150.0
    minimum_coarse_gap_nm: float = 300.0
    ordinary_x_spacing: str = "$dx"
    active_x_spacing: str = "$dx_QD"
    coarse_x_spacing: str = "$dx_coarse"
    ordinary_y_spacing: str = "$dy"
    active_y_spacing: str = "$dy_QD"

    def __post_init__(self) -> None:
        if self.active_transition_margin_nm < 0:
            raise ValueError("active_transition_margin_nm must be non-negative.")
        if self.minimum_coarse_gap_nm <= 0:
            raise ValueError("minimum_coarse_gap_nm must be positive.")

        spacing_fields = {
            "ordinary_x_spacing": self.ordinary_x_spacing,
            "active_x_spacing": self.active_x_spacing,
            "coarse_x_spacing": self.coarse_x_spacing,
            "ordinary_y_spacing": self.ordinary_y_spacing,
            "active_y_spacing": self.active_y_spacing,
        }
        for name, value in spacing_fields.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")


_Z_SPACING_FINENESS = {
    "$dz_buffer_coarse": 0,
    "$dz_buffer_medium": 1,
    "$dz_buffer_fine": 2,
    "$dz_oxide_gates_medium": 3,
    "$dz_QW_coarse": 4,
    "$dz_QW_fine": 5,
    "$dz_cap_fine": 6,
}

def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def _indent(text: str, n: int = 1) -> str:
    pad = "    " * n
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def _quote(value: str) -> str:
    return f'"{value}"'


def _find_block_span(text: str, block_name: str) -> Tuple[int, int]:
    """
    Find a top-level nextnano block like structure{...}, contacts{...}, quantum{...}.
    This avoids accidentally matching nested calls like !WHEN $quantum quantum{}.
    """
    import re

    pattern = re.compile(rf"(?m)^[ \t]*{block_name}[ \t]*\{{")
    match = pattern.search(text)

    if match is None:
        raise ValueError(f"Could not find top-level block '{block_name}' in template.")

    start = match.start()
    brace_start = text.find("{", start)

    depth = 0
    for i in range(brace_start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1

    raise ValueError(f"Could not find end of block '{block_name}'.")


def replace_block(text: str, block_name: str, replacement: str) -> str:
    start, end = _find_block_span(text, block_name)
    return text[:start] + replacement + text[end:]


def _voltage_var_name(region: PatternedRegion) -> str:
    if region.voltage_label:
        return region.voltage_label

    return f"V_{region.name}"


def _default_voltage_for_region(region: PatternedRegion) -> float:
    if region.gate_type == "plunger":
        return -3.0
    return 0.0


def render_voltage_variables(
    simulation_layout: SimulationLayout,
    voltage_overrides: Optional[Dict[str, float]] = None,
) -> str:
    voltage_overrides = voltage_overrides or {}

    lines = [
        "#--------------------------------------",
        "#--- generated gate voltage values ---",
        "#--------------------------------------",
    ]

    seen = set()

    for region in simulation_layout.patterned_regions:
        var = _voltage_var_name(region)
        if var in seen:
            continue

        value = voltage_overrides.get(var, _default_voltage_for_region(region))
        lines.append(f"${var:<16} = {_fmt(value)} # (V) generated")
        seen.add(var)

    return "\n".join(lines) + "\n"


def render_polygonal_prism(points_xy: List[Tuple[float, float]], z_min_nm: float, z_max_nm: float) -> str:
    """
    Render a nextnano++ polygonal_prism block.

    Syntax follows nextnanopy's InputAssistant.region_polygonal_prism:
    polygonal_prism{
        z = [z_min, z_max]
        vertex{
            x = [x_i]
            y = [y_i]
        }
        ...
    }
    """
    lines = [
        "polygonal_prism{",
        f"    z = [{_fmt(z_min_nm)}, {_fmt(z_max_nm)}]",
    ]

    for x, y in points_xy:
        lines.extend(
            [
                "    vertex{",
                f"        x = [{_fmt(x)}]",
                f"        y = [{_fmt(y)}]",
                "    }",
            ]
        )

    lines.append("}")
    return "\n".join(lines)


def _bbox_polygon(region: PatternedRegion) -> List[Tuple[float, float]]:
    return [
        (region.x_min_nm, region.y_min_nm),
        (region.x_max_nm, region.y_min_nm),
        (region.x_max_nm, region.y_max_nm),
        (region.x_min_nm, region.y_max_nm),
    ]


def _material_block(material: str, alloy_x: Optional[float] = None) -> str:
    """
    Map our material labels to nextnano region material blocks.
    """
    if material == "SiGe":
        if alloy_x is None:
            raise ValueError("SiGe material requires alloy_x.")
        return f'ternary_constant{{ name = "Si(x)Ge(1-x)" alloy_x = {_fmt(alloy_x)} }}'

    if material == "Ge":
        return "binary{ name = Ge }"

    if material in {"Al2O3", "Sapphire_zb"}:
        return "binary{ name = Sapphire_zb }"

    # Metal regions in the reference file are represented by contact blocks only.
    # Keep this as a comment rather than adding an unsupported binary material.
    if material in {"Al", "aluminium", "aluminum"}:
        return "# metal region defined through contact"

    return f'binary{{ name = {material} }}'


def render_contacts_block(simulation_layout: SimulationLayout) -> str:
    lines = [
        "contacts{",
        "    vacuum_level = $vacuum_level",
    ]

    for region in simulation_layout.patterned_regions:
        var = _voltage_var_name(region)

        if region.gate_type == "ohmic":
            work_function = "$work_function_OC"
        else:
            work_function = "$work_function_Gate"

        lines.append(
            f"    schottky{{ name = {_quote(region.name)} bias = 0 "
            f"work_function = {work_function} - ${var} }}"
        )

    # Keep useful template contacts.
    lines.append("    ohmic{ name = Body bias = 0.0 shift = 0.0 }")
    lines.append("    fermi_hole{ name = remove_surface_charge bias = $V_remove_surface_charge }")
    lines.append("    fermi_hole{ name = zero_fermi_QW bias = 0.0 }")
    lines.append("}")

    return "\n".join(lines)

def render_auxiliary_contact_regions(
    simulation_layout: SimulationLayout,
    *,
    quantum_region: Optional[QuantumRegion] = None,
    quantum_region_policy: Optional[QuantumRegionPolicy] = None,
) -> str:
    """
    Add non-layout auxiliary contact regions.

    These are not PHIDL gates. They are simulation-control contacts:
    - remove_surface_charge: full-cap contact
    - zero_fermi_QW: QW Fermi-level reference contact

    The surface-charge contact follows the modeled SiGe cap.  The QW
    reference contact uses the same vertical bounds as the quantum solver,
    while retaining the full lateral simulation domain.
    """
    d = simulation_layout.domain
    cap = _named_background_region(simulation_layout, "SiGe_cap")
    if quantum_region is None:
        quantum_region_policy = quantum_region_policy or QuantumRegionPolicy()
        quantum_well = _named_background_region(simulation_layout, "Ge_QW")
        quantum_z_min_nm = (
            quantum_well.z_min_nm
            - quantum_region_policy.quantum_well_z_padding_nm
        )
        quantum_z_max_nm = (
            quantum_well.z_max_nm
            + quantum_region_policy.quantum_well_z_padding_nm
        )
    else:
        quantum_z_min_nm = quantum_region.z_min_nm
        quantum_z_max_nm = quantum_region.z_max_nm

    return f"""
    # auxiliary fermi_hole contact: remove_surface_charge
    region{{
        cuboid{{
            x = [{_fmt(d.x_min_nm)}, {_fmt(d.x_max_nm)}]
            y = [{_fmt(d.y_min_nm)}, {_fmt(d.y_max_nm)}]
            z = [{_fmt(cap.z_min_nm)}, {_fmt(cap.z_max_nm)}]
        }}
        contact{{ name = remove_surface_charge }}
    }}

    # auxiliary fermi_hole contact: zero_fermi_QW
    region{{
        cuboid{{
            x = [{_fmt(d.x_min_nm)}, {_fmt(d.x_max_nm)}]
            y = [{_fmt(d.y_min_nm)}, {_fmt(d.y_max_nm)}]
            z = [{_fmt(quantum_z_min_nm)}, {_fmt(quantum_z_max_nm)}]
        }}
        contact{{ name = zero_fermi_QW }}
    }}
"""


def render_structure_block(
    simulation_layout: SimulationLayout,
    *,
    quantum_region: Optional[QuantumRegion] = None,
    quantum_region_policy: Optional[QuantumRegionPolicy] = None,
) -> str:
    d = simulation_layout.domain

    lines = [
        "structure{",
        "    output_contact_index{ boxes = no }",
        "    output_impurities{ boxes = no }",
        "    output_region_index{ boxes = no }",
        "    output_material_index{ boxes = no }",
        "",
    ]

    # Background layers first.
    for bg in simulation_layout.background_regions:
        material_block = _material_block(bg.material, bg.alloy_x)

        region_lines = [
            f"    # background: {bg.name}",
            "    region{",
            "        cuboid{",
            f"            x = [{_fmt(d.x_min_nm)}, {_fmt(d.x_max_nm)}]",
            f"            y = [{_fmt(d.y_min_nm)}, {_fmt(d.y_max_nm)}]",
            f"            z = [{_fmt(bg.z_min_nm)}, {_fmt(bg.z_max_nm)}]",
            "        }",
            f"        {material_block}",
        ]

        if bg.name == "Ge_QW":
            region_lines.append("        integrate{ hole_density{} }")

        if bg.contact_name is not None:
            region_lines.append(f"        contact{{ name = {bg.contact_name} }}")

        region_lines.append("    }")
        region_lines.append("")

        lines.extend(region_lines)

    # Auxiliary contacts not coming from PHIDL.
    lines.append(
        render_auxiliary_contact_regions(
            simulation_layout,
            quantum_region=quantum_region,
            quantum_region_policy=quantum_region_policy,
        )
    )
    lines.append("")

    # Patterned override regions.
    for region in simulation_layout.patterned_regions:
        polygons = region.polygon_xy_nm or [_bbox_polygon(region)]

        for idx, polygon in enumerate(polygons):
            suffix = "" if len(polygons) == 1 else f"_{idx + 1}"

            prism = render_polygonal_prism(
                points_xy=polygon,
                z_min_nm=region.z_min_nm,
                z_max_nm=region.z_max_nm,
            )

            region_lines = [
                f"    # patterned region: {region.name}{suffix}",
                "    region{",
                _indent(prism, 2),
                f"        contact{{ name = {_quote(region.name)} }}",
                "    }",
                "",
            ]
            lines.extend(region_lines)

    lines.append("}")

    return "\n".join(lines)


def render_output_block(simulation_layout: SimulationLayout) -> str:
    d = simulation_layout.domain

    quantum_well = _named_background_region(simulation_layout, "Ge_QW")
    z_qw_mid = 0.5 * (quantum_well.z_min_nm + quantum_well.z_max_nm)
    y_mid = 0.5 * (d.y_min_nm + d.y_max_nm)

    return f"""output{{
    only_sections = no

    format2D = AvsAscii_one_file
    format3D = VTKAscii

    set_origin{{ x = 0.0 y = 0.0 z = 0.0 }}

    section2D{{ name = "xy_QW" z = {_fmt(z_qw_mid)} }}
    section2D{{ name = "xz_QW" y = {_fmt(y_mid)} }}
    section2D{{ name = "xy_full_top" z = {_fmt(d.z_max_nm)} }}
}}"""


def _named_background_region(
    simulation_layout: SimulationLayout,
    name: str,
) -> BackgroundRegion:
    matches = [
        region
        for region in simulation_layout.background_regions
        if region.name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Nextnano generation requires exactly one background region "
            f"named '{name}'; found {len(matches)}."
        )
    return matches[0]


def build_adaptive_z_grid_lines(
    simulation_layout: SimulationLayout,
    policy: Optional[AdaptiveZGridPolicy] = None,
    *,
    quantum_region: Optional[QuantumRegion] = None,
) -> List[Tuple[float, str]]:
    """Return sorted, deduplicated ``(position_nm, spacing_variable)`` pairs."""
    policy = policy or AdaptiveZGridPolicy()
    d = simulation_layout.domain

    buffer = _named_background_region(simulation_layout, "SiGe_buffer")
    quantum_well = _named_background_region(simulation_layout, "Ge_QW")
    cap = _named_background_region(simulation_layout, "SiGe_cap")
    quantum_z_min_nm = (
        quantum_region.z_min_nm
        if quantum_region is not None
        else quantum_well.z_min_nm - policy.quantum_margin_nm
    )
    quantum_z_max_nm = (
        quantum_region.z_max_nm
        if quantum_region is not None
        else quantum_well.z_max_nm + policy.quantum_margin_nm
    )

    requirements: Dict[float, str] = {}
    domain_min = float(_fmt(d.z_min_nm))
    domain_max = float(_fmt(d.z_max_nm))

    def add(position_nm: float, spacing: str) -> None:
        if spacing not in _Z_SPACING_FINENESS:
            raise ValueError(f"Unknown nextnano z-grid spacing variable: {spacing}")

        # Canonicalize exactly as the writer does so distinct float expressions
        # cannot produce duplicate textual positions.
        position = float(_fmt(position_nm))
        if position == 0:
            position = 0.0
        if position < domain_min or position > domain_max:
            return

        current = requirements.get(position)
        if (
            current is None
            or _Z_SPACING_FINENESS[spacing] > _Z_SPACING_FINENESS[current]
        ):
            requirements[position] = spacing

    # Domain and continuous material interfaces.
    add(d.z_min_nm, "$dz_buffer_coarse")
    add(d.z_max_nm, "$dz_oxide_gates_medium")

    for region in simulation_layout.background_regions:
        if region.contact_name == "Body":
            add(region.z_min_nm, "$dz_buffer_coarse")
            add(region.z_max_nm, "$dz_buffer_coarse")
        elif region.name == buffer.name:
            add(region.z_min_nm, "$dz_buffer_coarse")
            add(region.z_max_nm, "$dz_QW_fine")
        elif region.name == quantum_well.name:
            add(region.z_min_nm, "$dz_QW_fine")
            add(region.z_max_nm, "$dz_QW_fine")
        elif region.name == cap.name:
            add(region.z_min_nm, "$dz_QW_fine")
            add(region.z_max_nm, "$dz_cap_fine")
        else:
            add(region.z_min_nm, "$dz_oxide_gates_medium")
            add(region.z_max_nm, "$dz_oxide_gates_medium")

    # Deep-to-upper buffer hierarchy, measured down from the QW-facing
    # buffer interface so added substrate thickness remains coarsely resolved.
    buffer_medium_z = (
        buffer.z_max_nm - policy.buffer_medium_refinement_depth_nm
    )
    buffer_fine_z = buffer.z_max_nm - policy.buffer_fine_refinement_depth_nm
    if buffer.z_min_nm < buffer_medium_z < buffer.z_max_nm:
        add(buffer_medium_z, "$dz_buffer_medium")
    if buffer.z_min_nm < buffer_fine_z < buffer.z_max_nm:
        add(buffer_fine_z, "$dz_buffer_fine")

    # QW interfaces and the shared quantum-region bounds.
    add(quantum_z_min_nm, "$dz_QW_coarse")
    add(quantum_well.z_min_nm, "$dz_QW_fine")
    add(quantum_well.z_max_nm, "$dz_QW_fine")
    add(quantum_z_max_nm, "$dz_QW_coarse")

    # Resolve the final cap slice and the cap/dielectric interface.
    cap_fine_start_z = max(
        cap.z_min_nm,
        cap.z_max_nm - policy.cap_interface_fine_thickness_nm,
    )
    add(cap_fine_start_z, "$dz_cap_fine")
    add(cap.z_max_nm, "$dz_cap_fine")

    # Every actual patterned-region interface is a required grid line.
    for region in simulation_layout.patterned_regions:
        add(region.z_min_nm, "$dz_oxide_gates_medium")
        add(region.z_max_nm, "$dz_oxide_gates_medium")

    return sorted(requirements.items())


def build_lateral_grid_lines(
    simulation_layout: SimulationLayout,
    *,
    quantum_region: Optional[QuantumRegion] = None,
    quantum_region_policy: Optional[QuantumRegionPolicy] = None,
    policy: Optional[LateralMeshPolicy] = None,
) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]:
    """Build sorted, unique geometry-aware x/y grid-line specifications."""
    policy = policy or LateralMeshPolicy()
    quantum_region_policy = quantum_region_policy or QuantumRegionPolicy()
    quantum_region = quantum_region or derive_quantum_region(
        simulation_layout,
        policy=quantum_region_policy,
    )
    d = simulation_layout.domain
    active = derive_active_device_bounds(
        simulation_layout,
        excluded_gate_types=quantum_region_policy.excluded_gate_types,
    )

    x_requirements: Dict[float, Tuple[str, int]] = {}
    y_requirements: Dict[float, Tuple[str, int]] = {}
    x_domain_min = float(_fmt(d.x_min_nm))
    x_domain_max = float(_fmt(d.x_max_nm))
    y_domain_min = float(_fmt(d.y_min_nm))
    y_domain_max = float(_fmt(d.y_max_nm))

    def add(
        requirements: Dict[float, Tuple[str, int]],
        position_nm: float,
        spacing: str,
        priority: int,
        domain_min_nm: float,
        domain_max_nm: float,
    ) -> None:
        position = float(_fmt(position_nm))
        if position == 0:
            position = 0.0
        if position < domain_min_nm or position > domain_max_nm:
            return

        current = requirements.get(position)
        if current is None or priority > current[1]:
            requirements[position] = (spacing, priority)

    def add_x(position_nm: float, spacing: str, priority: int) -> None:
        add(
            x_requirements,
            position_nm,
            spacing,
            priority,
            x_domain_min,
            x_domain_max,
        )

    def add_y(position_nm: float, spacing: str, priority: int) -> None:
        add(
            y_requirements,
            position_nm,
            spacing,
            priority,
            y_domain_min,
            y_domain_max,
        )

    # Domain bounds use the ordinary mesh.  Fine quantum/active anchors win
    # any x collision (important when a layout has no ohmics).
    add_x(d.x_min_nm, policy.ordinary_x_spacing, 1)
    add_x(d.x_max_nm, policy.ordinary_x_spacing, 1)
    add_y(d.y_min_nm, policy.ordinary_y_spacing, 1)
    add_y(d.y_max_nm, policy.ordinary_y_spacing, 1)

    add_x(quantum_region.x_min_nm, policy.active_x_spacing, 2)
    add_x(quantum_region.x_max_nm, policy.active_x_spacing, 2)
    add_y(quantum_region.y_min_nm, policy.active_y_spacing, 2)
    add_y(quantum_region.y_max_nm, policy.active_y_spacing, 2)

    excluded_gate_types = set(quantum_region_policy.excluded_gate_types)
    active_regions = [
        region
        for region in simulation_layout.patterned_regions
        if region.gate_type not in excluded_gate_types
    ]
    excluded_regions = [
        region
        for region in simulation_layout.patterned_regions
        if region.gate_type in excluded_gate_types
    ]

    for region in active_regions:
        for position in (
            region.x_min_nm,
            0.5 * (region.x_min_nm + region.x_max_nm),
            region.x_max_nm,
        ):
            add_x(position, policy.active_x_spacing, 2)

    transition_margin = policy.active_transition_margin_nm
    left_transition = active.x_min_nm - transition_margin
    right_transition = active.x_max_nm + transition_margin

    laterally_relevant_excluded = [
        region
        for region in excluded_regions
        if max(quantum_region.y_min_nm, region.y_min_nm)
        < min(quantum_region.y_max_nm, region.y_max_nm)
    ]
    left_inner_edges = [
        region.x_max_nm
        for region in laterally_relevant_excluded
        if region.x_max_nm <= active.x_min_nm
    ]
    right_inner_edges = [
        region.x_min_nm
        for region in laterally_relevant_excluded
        if region.x_min_nm >= active.x_max_nm
    ]
    left_gap_boundary = max(left_inner_edges, default=d.x_min_nm)
    right_gap_boundary = min(right_inner_edges, default=d.x_max_nm)
    left_gap_is_long = (
        active.x_min_nm - left_gap_boundary >= policy.minimum_coarse_gap_nm
    )
    right_gap_is_long = (
        right_gap_boundary - active.x_max_nm >= policy.minimum_coarse_gap_nm
    )

    # Coarse transition anchors are useful only when a real interval remains
    # outside the quantum box.  This omits/clamps them for short gaps.
    if (
        left_gap_is_long
        and left_gap_boundary < left_transition < quantum_region.x_min_nm
    ):
        add_x(left_transition, policy.coarse_x_spacing, 0)
    if (
        right_gap_is_long
        and quantum_region.x_max_nm < right_transition < right_gap_boundary
    ):
        add_x(right_transition, policy.coarse_x_spacing, 0)

    for region in excluded_regions:
        center_x = 0.5 * (region.x_min_nm + region.x_max_nm)
        add_x(center_x, policy.ordinary_x_spacing, 1)

        if region.x_max_nm <= active.x_min_nm:
            gap_nm = active.x_min_nm - region.x_max_nm
            add_x(region.x_min_nm, policy.ordinary_x_spacing, 1)
            if (
                gap_nm >= policy.minimum_coarse_gap_nm
                and region.x_max_nm < left_transition
            ):
                add_x(region.x_max_nm, policy.coarse_x_spacing, 0)
            else:
                add_x(region.x_max_nm, policy.ordinary_x_spacing, 1)
        elif region.x_min_nm >= active.x_max_nm:
            gap_nm = region.x_min_nm - active.x_max_nm
            if (
                gap_nm >= policy.minimum_coarse_gap_nm
                and region.x_min_nm > right_transition
            ):
                add_x(region.x_min_nm, policy.coarse_x_spacing, 0)
            else:
                add_x(region.x_min_nm, policy.ordinary_x_spacing, 1)
            add_x(region.x_max_nm, policy.ordinary_x_spacing, 1)
        else:
            add_x(region.x_min_nm, policy.ordinary_x_spacing, 1)
            add_x(region.x_max_nm, policy.ordinary_x_spacing, 1)

    # Y remains uniformly fine for the reference profile, but its anchors are
    # still derived from the domain, quantum box, and every layout feature.
    for region in simulation_layout.patterned_regions:
        for position in (
            region.y_min_nm,
            0.5 * (region.y_min_nm + region.y_max_nm),
            region.y_max_nm,
        ):
            add_y(position, policy.active_y_spacing, 2)

    x_lines = [
        (position, spacing_and_priority[0])
        for position, spacing_and_priority in sorted(x_requirements.items())
    ]
    y_lines = [
        (position, spacing_and_priority[0])
        for position, spacing_and_priority in sorted(y_requirements.items())
    ]
    return x_lines, y_lines


def render_grid_block(
    simulation_layout: SimulationLayout,
    *,
    quantum_region: Optional[QuantumRegion] = None,
    quantum_region_policy: Optional[QuantumRegionPolicy] = None,
    lateral_mesh_policy: Optional[LateralMeshPolicy] = None,
    z_grid_policy: Optional[AdaptiveZGridPolicy] = None,
) -> str:
    quantum_region_policy = _resolve_quantum_region_policy(
        quantum_region_policy,
        z_grid_policy,
    )
    quantum_region = quantum_region or derive_quantum_region(
        simulation_layout,
        policy=quantum_region_policy,
    )
    x_lines, y_lines = build_lateral_grid_lines(
        simulation_layout,
        quantum_region=quantum_region,
        quantum_region_policy=quantum_region_policy,
        policy=lateral_mesh_policy,
    )

    def render_axis(
        axis: str,
        values: List[Tuple[float, str]],
    ) -> List[str]:
        lines = [f"    {axis}grid{{"]
        for value, spacing in values:
            lines.append(f"        line{{ pos = {_fmt(value)} spacing = {spacing} }}")
        lines.append("    }")
        return lines

    lines = ["grid{"]
    lines.extend(render_axis("x", x_lines))
    lines.extend(render_axis("y", y_lines))
    lines.append("    zgrid{")
    for position, spacing in build_adaptive_z_grid_lines(
        simulation_layout,
        policy=z_grid_policy,
        quantum_region=quantum_region,
    ):
        lines.append(
            f"        line{{ pos = {_fmt(position)} spacing = {spacing} }}"
        )
    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def render_run_block() -> str:
    return """run{
    !WHEN $strain             strain{}
    !WHEN $poisson            poisson{}
    !WHEN $quantum            quantum{}
    !WHEN $quantum_poisson    quantum_poisson{}
}"""


def render_quantum_block(
    simulation_layout: SimulationLayout,
    *,
    quantum_region_name: str = "c-Ge_QW",
    quantum_region: Optional[QuantumRegion] = None,
    quantum_region_policy: Optional[QuantumRegionPolicy] = None,
) -> str:
    quantum_region = quantum_region or derive_quantum_region(
        simulation_layout,
        policy=quantum_region_policy,
    )

    return f"""quantum{{
    region{{
        name = "{quantum_region_name}"
        x = [{_fmt(quantum_region.x_min_nm)}, {_fmt(quantum_region.x_max_nm)}]
        y = [{_fmt(quantum_region.y_min_nm)}, {_fmt(quantum_region.y_max_nm)}]
        z = [{_fmt(quantum_region.z_min_nm)}, {_fmt(quantum_region.z_max_nm)}]

        boundary{{
            x = neumann
            y = neumann
            z = neumann
        }}

        HH{{ num_ev = $n_c_Ge_QW_HH }}
        LH{{ num_ev = $n_c_Ge_QW_LH }}

        !IF($decomposition)
            quantize_z{{}}
            output_wavefunctions{{
                max_num = 100
                amplitudes = no
                probabilities = yes
            }}
        !ELSE
            output_wavefunctions{{
                max_num = 100
                amplitudes = no
                probabilities = yes
                all_k_points = no
                energy_shift = shifted
            }}
        !ENDIF

        output_quantum_densities{{}}
    }}
}}"""


def _resolve_quantum_region_policy(
    quantum_region_policy: Optional[QuantumRegionPolicy],
    z_grid_policy: Optional[AdaptiveZGridPolicy],
) -> QuantumRegionPolicy:
    """Bridge the legacy z-grid margin into the shared quantum policy."""
    if quantum_region_policy is None:
        if z_grid_policy is None:
            return QuantumRegionPolicy()
        return QuantumRegionPolicy(
            quantum_well_z_padding_nm=z_grid_policy.quantum_margin_nm,
        )

    if (
        z_grid_policy is not None
        and quantum_region_policy.quantum_well_z_padding_nm
        != z_grid_policy.quantum_margin_nm
    ):
        raise ValueError(
            "QuantumRegionPolicy.quantum_well_z_padding_nm and "
            "AdaptiveZGridPolicy.quantum_margin_nm must match when both "
            "policies are supplied."
        )
    return quantum_region_policy


def write_nextnano_input_from_template(
    *,
    simulation_layout: SimulationLayout,
    template_path: str | Path,
    output_path: str | Path,
    voltage_overrides: Optional[Dict[str, float]] = None,
    replace_output: bool = True,
    replace_contacts: bool = True,
    replace_structure: bool = True,
    replace_grid: bool = True,
    replace_quantum: bool = True,
    replace_run: bool = True,
    quantum_region_policy: Optional[QuantumRegionPolicy] = None,
    lateral_mesh_policy: Optional[LateralMeshPolicy] = None,
    z_grid_policy: Optional[AdaptiveZGridPolicy] = None,
) -> Path:
    """
    Write a nextnano input file from a reference template.

    The writer keeps the solver/material database parts of the template,
    but replaces generated geometry-dependent blocks.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)
    quantum_region: Optional[QuantumRegion] = None
    if replace_grid or replace_quantum:
        quantum_region_policy = _resolve_quantum_region_policy(
            quantum_region_policy,
            z_grid_policy,
        )
        quantum_region = derive_quantum_region(
            simulation_layout,
            policy=quantum_region_policy,
        )

    text = template_path.read_text(encoding="utf-8")

    generated_vars = render_voltage_variables(
        simulation_layout,
        voltage_overrides=voltage_overrides,
    )

    text = generated_vars + "\n" + text

    if replace_output:
        text = replace_block(text, "output", render_output_block(simulation_layout))

    if replace_contacts:
        text = replace_block(text, "contacts", render_contacts_block(simulation_layout))

    if replace_structure:
        text = replace_block(
            text,
            "structure",
            render_structure_block(
                simulation_layout,
                quantum_region=quantum_region,
                quantum_region_policy=quantum_region_policy,
            ),
        )

    if replace_grid:
        text = replace_block(
            text,
            "grid",
            render_grid_block(
                simulation_layout,
                quantum_region=quantum_region,
                quantum_region_policy=quantum_region_policy,
                lateral_mesh_policy=lateral_mesh_policy,
                z_grid_policy=z_grid_policy,
            ),
        )

    if replace_quantum:
        text = replace_block(
            text,
            "quantum",
            render_quantum_block(
                simulation_layout,
                quantum_region=quantum_region,
            ),
        )

    if replace_run:
        text = replace_block(text, "run", render_run_block())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return output_path
