from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .simulation_layout import PatternedRegion, SimulationLayout


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
    - Body: bottom/back contact
    - remove_surface_charge: cap/oxide interface contact
    - zero_fermi_QW: QW Fermi-level reference contact

    Coordinate convention:
    - z = 0 nm is the top of the Ge QW
    - Ge QW spans z = [-15, 0]
    - SiGe cap spans z = [0, 101]
    - oxide starts at z = 101
    """
    d = simulation_layout.domain

    # Thin bottom body contact at the bottom of the buffer.
    body_z_min = d.z_min_nm
    body_z_max = d.z_min_nm + 5.0

    # Thin region at the SiGe-cap / Al2O3 interface.
    remove_surface_z_min = 100.0
    remove_surface_z_max = 101.0

    # QW reference region.
    qw_z_min = -15.0
    qw_z_max = 0.0

    return f"""
    # auxiliary contact: Body
    region{{
        cuboid{{
            x = [{_fmt(d.x_min_nm)}, {_fmt(d.x_max_nm)}]
            y = [{_fmt(d.y_min_nm)}, {_fmt(d.y_max_nm)}]
            z = [{_fmt(body_z_min)}, {_fmt(body_z_max)}]
        }}
        contact{{ name = Body }}
    }}

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


def render_grid_block(simulation_layout: SimulationLayout) -> str:
    d = simulation_layout.domain

    x_lines = {d.x_min_nm, d.x_max_nm}
    y_lines = {d.y_min_nm, d.y_max_nm}
    z_lines = {d.z_min_nm, d.z_max_nm, -15.0, 0.0, 101.0}

    for region in simulation_layout.patterned_regions:
        x_lines.update([region.x_min_nm, region.x_max_nm])
        y_lines.update([region.y_min_nm, region.y_max_nm])
        z_lines.update([region.z_min_nm, region.z_max_nm])

    def render_axis(axis: str, values: Iterable[float], spacing: str) -> List[str]:
        lines = [f"    {axis}grid{{"]
        for value in sorted(values):
            lines.append(f"        line{{ pos = {_fmt(value)} spacing = {spacing} }}")
        lines.append("    }")
        return lines

    lines = ["grid{"]
    lines.extend(render_axis("x", x_lines, "$dx_QD"))
    lines.extend(render_axis("y", y_lines, "$dy_QD"))
    lines.extend(render_axis("z", z_lines, "$dz_oxide_gates_medium"))
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
        text = replace_block(text, "grid", render_grid_block(simulation_layout))

    if replace_quantum:
        text = replace_block(text, "quantum", render_quantum_block(simulation_layout))

    if replace_run:
        text = replace_block(text, "run", render_run_block())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return output_path