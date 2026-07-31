from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .simulation_layout import BackgroundRegion, PatternedRegion, SimulationLayout


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

def render_auxiliary_contact_regions(simulation_layout: SimulationLayout) -> str:
    """
    Add non-layout auxiliary contact regions.

    These are not PHIDL gates. They are simulation-control contacts:
    - remove_surface_charge: cap/oxide interface contact
    - zero_fermi_QW: QW Fermi-level reference contact

    Coordinate convention:
    - z = 0 nm is the top of the Ge QW
    - Ge QW spans z = [-15, 0]
    - SiGe cap spans z = [0, 101]
    - oxide starts at z = 101
    """
    d = simulation_layout.domain

    # Thin region at the SiGe-cap / Al2O3 interface.
    remove_surface_z_min = 100.0
    remove_surface_z_max = 101.0

    # QW reference region.
    qw_z_min = -15.0
    qw_z_max = 0.0

    return f"""
    # auxiliary fermi_hole contact: remove_surface_charge
    region{{
        cuboid{{
            x = [{_fmt(d.x_min_nm)}, {_fmt(d.x_max_nm)}]
            y = [{_fmt(d.y_min_nm)}, {_fmt(d.y_max_nm)}]
            z = [{_fmt(remove_surface_z_min)}, {_fmt(remove_surface_z_max)}]
        }}
        contact{{ name = remove_surface_charge }}
    }}

    # auxiliary fermi_hole contact: zero_fermi_QW
    region{{
        cuboid{{
            x = [{_fmt(d.x_min_nm)}, {_fmt(d.x_max_nm)}]
            y = [{_fmt(d.y_min_nm)}, {_fmt(d.y_max_nm)}]
            z = [{_fmt(qw_z_min)}, {_fmt(qw_z_max)}]
        }}
        contact{{ name = zero_fermi_QW }}
    }}
"""


def render_structure_block(simulation_layout: SimulationLayout) -> str:
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
    lines.append(render_auxiliary_contact_regions(simulation_layout))
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

    z_qw_mid = -7.5
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
            f"Adaptive nextnano z-grid requires exactly one background region "
            f"named '{name}'; found {len(matches)}."
        )
    return matches[0]


def build_adaptive_z_grid_lines(
    simulation_layout: SimulationLayout,
    policy: Optional[AdaptiveZGridPolicy] = None,
) -> List[Tuple[float, str]]:
    """Return sorted, deduplicated ``(position_nm, spacing_variable)`` pairs."""
    policy = policy or AdaptiveZGridPolicy()
    d = simulation_layout.domain

    buffer = _named_background_region(simulation_layout, "SiGe_buffer")
    quantum_well = _named_background_region(simulation_layout, "Ge_QW")
    cap = _named_background_region(simulation_layout, "SiGe_cap")

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

    # QW interfaces and the quantum-region refinement margin.
    add(
        quantum_well.z_min_nm - policy.quantum_margin_nm,
        "$dz_QW_coarse",
    )
    add(quantum_well.z_min_nm, "$dz_QW_fine")
    add(quantum_well.z_max_nm, "$dz_QW_fine")
    add(
        quantum_well.z_max_nm + policy.quantum_margin_nm,
        "$dz_QW_coarse",
    )

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


def render_grid_block(
    simulation_layout: SimulationLayout,
    *,
    z_grid_policy: Optional[AdaptiveZGridPolicy] = None,
) -> str:
    d = simulation_layout.domain

    x_lines = {d.x_min_nm, d.x_max_nm}
    y_lines = {d.y_min_nm, d.y_max_nm}

    for region in simulation_layout.patterned_regions:
        x_lines.update([region.x_min_nm, region.x_max_nm])
        y_lines.update([region.y_min_nm, region.y_max_nm])

    def render_axis(axis: str, values: Iterable[float], spacing: str) -> List[str]:
        lines = [f"    {axis}grid{{"]
        for value in sorted(values):
            lines.append(f"        line{{ pos = {_fmt(value)} spacing = {spacing} }}")
        lines.append("    }")
        return lines

    lines = ["grid{"]
    lines.extend(render_axis("x", x_lines, "$dx_QD"))
    lines.extend(render_axis("y", y_lines, "$dy_QD"))
    lines.append("    zgrid{")
    for position, spacing in build_adaptive_z_grid_lines(
        simulation_layout,
        policy=z_grid_policy,
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
    z_min_nm: float = -20.0,
    z_max_nm: float = 5.0,
) -> str:
    d = simulation_layout.domain

    return f"""quantum{{
    region{{
        name = "{quantum_region_name}"
        x = [{_fmt(d.x_min_nm)}, {_fmt(d.x_max_nm)}]
        y = [{_fmt(d.y_min_nm)}, {_fmt(d.y_max_nm)}]
        z = [{_fmt(z_min_nm)}, {_fmt(z_max_nm)}]

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
    z_grid_policy: Optional[AdaptiveZGridPolicy] = None,
) -> Path:
    """
    Write a nextnano input file from a reference template.

    The writer keeps the solver/material database parts of the template,
    but replaces generated geometry-dependent blocks.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

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
        text = replace_block(text, "structure", render_structure_block(simulation_layout))

    if replace_grid:
        text = replace_block(
            text,
            "grid",
            render_grid_block(
                simulation_layout,
                z_grid_policy=z_grid_policy,
            ),
        )

    if replace_quantum:
        text = replace_block(text, "quantum", render_quantum_block(simulation_layout))

    if replace_run:
        text = replace_block(text, "run", render_run_block())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return output_path
