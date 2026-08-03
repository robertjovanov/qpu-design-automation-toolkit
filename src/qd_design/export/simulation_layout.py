from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json

from .layout_spec import PlacedElement
from .process_stack import MaterialLayer, ProcessStack

PolygonXY = List[tuple[float, float]]


@dataclass
class SimulationDomain:
    """
    Full simulation domain bounds in nm.
    """

    x_min_nm: float
    x_max_nm: float
    y_min_nm: float
    y_max_nm: float
    z_min_nm: float
    z_max_nm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackgroundRegion:
    """
    Continuous layer spanning the full x-y simulation domain.
    """

    name: str
    material: str
    x_min_nm: float
    x_max_nm: float
    y_min_nm: float
    y_max_nm: float
    z_min_nm: float
    z_max_nm: float
    alloy_x: Optional[float] = None
    notes: Optional[str] = None
    contact_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternedRegion:
    """
    Local 3D region obtained by extruding a 2D layout element according to the process stack.
    """

    name: str
    gate_type: str
    layer_name: str
    material: str
    voltage_label: Optional[str]

    x_min_nm: float
    x_max_nm: float
    y_min_nm: float
    y_max_nm: float
    z_min_nm: float
    z_max_nm: float

    gds_layer: int
    gds_datatype: int

    source_rule_name: str
    params: Dict[str, Any]
    polygon_xy_nm: List[PolygonXY]
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationLayout:
    """
    Combined 3D simulation-ready description of a device.

    Contains:
    - simulation domain
    - background material layers
    - patterned override regions
    """

    name: str
    z_reference_name: str
    domain: SimulationDomain
    background_regions: List[BackgroundRegion] = field(default_factory=list)
    patterned_regions: List[PatternedRegion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "z_reference_name": self.z_reference_name,
            "domain": self.domain.to_dict(),
            "background_regions": [region.to_dict() for region in self.background_regions],
            "patterned_regions": [region.to_dict() for region in self.patterned_regions],
        }

    def write_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass(frozen=True)
class LateralBounds:
    """Axis-aligned lateral bounds in layout coordinates."""

    x_min_nm: float
    x_max_nm: float
    y_min_nm: float
    y_max_nm: float

    def __post_init__(self) -> None:
        if self.x_max_nm <= self.x_min_nm:
            raise ValueError("Lateral x-bounds must be non-empty.")
        if self.y_max_nm <= self.y_min_nm:
            raise ValueError("Lateral y-bounds must be non-empty.")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class QuantumRegion:
    """Computed solver region shared by all nextnano geometry consumers."""

    x_min_nm: float
    x_max_nm: float
    y_min_nm: float
    y_max_nm: float
    z_min_nm: float
    z_max_nm: float

    def __post_init__(self) -> None:
        if self.x_max_nm <= self.x_min_nm:
            raise ValueError("Quantum-region x-bounds must be non-empty.")
        if self.y_max_nm <= self.y_min_nm:
            raise ValueError("Quantum-region y-bounds must be non-empty.")
        if self.z_max_nm <= self.z_min_nm:
            raise ValueError("Quantum-region z-bounds must be non-empty.")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class QuantumRegionPolicy:
    """Geometry-relative policy for deriving the quantum solver region."""

    active_x_padding_nm: float = 20.0
    domain_y_inset_nm: float = 20.0
    quantum_well_z_padding_nm: float = 5.0
    excluded_gate_types: Tuple[str, ...] = ("ohmic",)

    def __post_init__(self) -> None:
        if self.active_x_padding_nm < 0:
            raise ValueError("active_x_padding_nm must be non-negative.")
        if self.domain_y_inset_nm < 0:
            raise ValueError("domain_y_inset_nm must be non-negative.")
        if self.quantum_well_z_padding_nm <= 0:
            raise ValueError("quantum_well_z_padding_nm must be positive.")


def derive_active_device_bounds(
    simulation_layout: SimulationLayout,
    *,
    excluded_gate_types: Sequence[str] = ("ohmic",),
) -> LateralBounds:
    """Return the bounding box of patterned gates participating in the device."""
    excluded = set(excluded_gate_types)
    active_regions = [
        region
        for region in simulation_layout.patterned_regions
        if region.gate_type not in excluded
    ]
    if not active_regions:
        raise ValueError(
            "Cannot derive an active-device region: no relevant non-excluded "
            "patterned gates were found."
        )

    return LateralBounds(
        x_min_nm=min(region.x_min_nm for region in active_regions),
        x_max_nm=max(region.x_max_nm for region in active_regions),
        y_min_nm=min(region.y_min_nm for region in active_regions),
        y_max_nm=max(region.y_max_nm for region in active_regions),
    )


def derive_quantum_region(
    simulation_layout: SimulationLayout,
    policy: Optional[QuantumRegionPolicy] = None,
) -> QuantumRegion:
    """Derive a contained quantum box from active gates and the named Ge QW."""
    policy = policy or QuantumRegionPolicy()
    domain = simulation_layout.domain
    active = derive_active_device_bounds(
        simulation_layout,
        excluded_gate_types=policy.excluded_gate_types,
    )

    quantum_wells = [
        region
        for region in simulation_layout.background_regions
        if region.name == "Ge_QW"
    ]
    if len(quantum_wells) != 1:
        raise ValueError(
            "Quantum-region generation requires exactly one background region "
            f"named 'Ge_QW'; found {len(quantum_wells)}."
        )
    quantum_well = quantum_wells[0]

    excluded_gate_types = set(policy.excluded_gate_types)
    excluded_regions = [
        patterned_region
        for patterned_region in simulation_layout.patterned_regions
        if patterned_region.gate_type in excluded_gate_types
    ]
    y_min_nm = domain.y_min_nm + policy.domain_y_inset_nm
    y_max_nm = domain.y_max_nm - policy.domain_y_inset_nm
    left_exclusion_edges = [
        patterned_region.x_max_nm
        for patterned_region in excluded_regions
        if (
            patterned_region.x_max_nm <= active.x_min_nm
            and max(y_min_nm, patterned_region.y_min_nm)
            < min(y_max_nm, patterned_region.y_max_nm)
        )
    ]
    right_exclusion_edges = [
        patterned_region.x_min_nm
        for patterned_region in excluded_regions
        if (
            patterned_region.x_min_nm >= active.x_max_nm
            and max(y_min_nm, patterned_region.y_min_nm)
            < min(y_max_nm, patterned_region.y_max_nm)
        )
    ]
    x_min_nm = max(
        domain.x_min_nm,
        active.x_min_nm - policy.active_x_padding_nm,
    )
    x_max_nm = min(
        domain.x_max_nm,
        active.x_max_nm + policy.active_x_padding_nm,
    )
    if left_exclusion_edges:
        x_min_nm = max(x_min_nm, max(left_exclusion_edges))
    if right_exclusion_edges:
        x_max_nm = min(x_max_nm, min(right_exclusion_edges))

    lower_exclusion_edges = [
        patterned_region.y_max_nm
        for patterned_region in excluded_regions
        if (
            patterned_region.y_max_nm <= active.y_min_nm
            and max(x_min_nm, patterned_region.x_min_nm)
            < min(x_max_nm, patterned_region.x_max_nm)
        )
    ]
    upper_exclusion_edges = [
        patterned_region.y_min_nm
        for patterned_region in excluded_regions
        if (
            patterned_region.y_min_nm >= active.y_max_nm
            and max(x_min_nm, patterned_region.x_min_nm)
            < min(x_max_nm, patterned_region.x_max_nm)
        )
    ]

    if lower_exclusion_edges:
        y_min_nm = max(y_min_nm, max(lower_exclusion_edges))
    if upper_exclusion_edges:
        y_max_nm = min(y_max_nm, min(upper_exclusion_edges))

    region = QuantumRegion(
        x_min_nm=x_min_nm,
        x_max_nm=x_max_nm,
        y_min_nm=y_min_nm,
        y_max_nm=y_max_nm,
        z_min_nm=(
            quantum_well.z_min_nm - policy.quantum_well_z_padding_nm
        ),
        z_max_nm=(
            quantum_well.z_max_nm + policy.quantum_well_z_padding_nm
        ),
    )

    contained = (
        domain.x_min_nm <= region.x_min_nm < region.x_max_nm <= domain.x_max_nm
        and domain.y_min_nm <= region.y_min_nm < region.y_max_nm <= domain.y_max_nm
        and domain.z_min_nm <= region.z_min_nm < region.z_max_nm <= domain.z_max_nm
    )
    if not contained:
        raise ValueError(
            "The derived quantum region must be non-empty and lie inside the "
            "simulation domain."
        )

    overlapping_excluded_regions = [
        patterned_region.name
        for patterned_region in excluded_regions
        if (
            max(region.x_min_nm, patterned_region.x_min_nm)
            < min(region.x_max_nm, patterned_region.x_max_nm)
            and max(region.y_min_nm, patterned_region.y_min_nm)
            < min(region.y_max_nm, patterned_region.y_max_nm)
        )
    ]
    if overlapping_excluded_regions:
        names = ", ".join(sorted(overlapping_excluded_regions))
        raise ValueError(
            "The derived quantum region overlaps excluded patterned regions: "
            f"{names}. Adjust the quantum-region policy or layout metadata."
        )

    return region


def _placed_element_from_dict(data: Dict[str, Any]) -> PlacedElement:
    return PlacedElement(
        name=data["name"],
        gate_type=data["gate_type"],
        layer_name=data["layer_name"],
        gds_layer=data["gds_layer"],
        gds_datatype=data["gds_datatype"],
        voltage_label=data.get("voltage_label"),
        role=data.get("role"),
        notes=data.get("notes"),
        x_min_nm=data["x_min_nm"],
        y_min_nm=data["y_min_nm"],
        x_max_nm=data["x_max_nm"],
        y_max_nm=data["y_max_nm"],
        center_x_nm=data["center_x_nm"],
        center_y_nm=data["center_y_nm"],
        width_nm=data["width_nm"],
        height_nm=data["height_nm"],
        params=data.get("params", {}),
        polygon_xy_nm=data.get("polygon_xy_nm", []),
    )


def _normalize_elements(
    layout_elements: Sequence[Union[PlacedElement, Dict[str, Any]]],
) -> List[PlacedElement]:
    elements: List[PlacedElement] = []
    for element in layout_elements:
        if isinstance(element, PlacedElement):
            elements.append(element)
        else:
            elements.append(_placed_element_from_dict(element))
    return elements


def build_simulation_layout(
    *,
    name: str,
    layout_elements: Sequence[Union[PlacedElement, Dict[str, Any]]],
    process_stack: ProcessStack,
    x_margin_nm: float = 0.0,
    y_margin_nm: float = 0.0,
) -> SimulationLayout:
    """
    Combine a 2D layout spec and a ProcessStack into a 3D simulation-ready layout.

    Parameters
    ----------
    name : str
        Name of the resulting simulation layout.
    layout_elements :
        Usually device.layout_spec() or device.layout_elements().values().
    process_stack : ProcessStack
        Vertical stack and extrusion rules.
    x_margin_nm, y_margin_nm : float
        Optional padding around the lateral layout extents to define the simulation domain.
    """
    if x_margin_nm < 0 or y_margin_nm < 0:
        raise ValueError("Margins must be non-negative.")

    process_stack.validate()
    elements = _normalize_elements(layout_elements)

    if not elements:
        raise ValueError("layout_elements is empty.")

    x_min = min(e.x_min_nm for e in elements) - x_margin_nm
    x_max = max(e.x_max_nm for e in elements) + x_margin_nm
    y_min = min(e.y_min_nm for e in elements) - y_margin_nm
    y_max = max(e.y_max_nm for e in elements) + y_margin_nm

    domain = SimulationDomain(
        x_min_nm=x_min,
        x_max_nm=x_max,
        y_min_nm=y_min,
        y_max_nm=y_max,
        z_min_nm=process_stack.bottom_z_nm(),
        z_max_nm=process_stack.top_z_nm(),
    )

    background_regions = [
        BackgroundRegion(
            name=layer.name,
            material=layer.material,
            x_min_nm=domain.x_min_nm,
            x_max_nm=domain.x_max_nm,
            y_min_nm=domain.y_min_nm,
            y_max_nm=domain.y_max_nm,
            z_min_nm=layer.z_min_nm,
            z_max_nm=layer.z_max_nm,
            alloy_x=layer.alloy_x,
            notes=layer.notes,
            contact_name=layer.contact_name,
        )
        for layer in process_stack.material_layers
    ]

    patterned_regions: List[PatternedRegion] = []
    for element in elements:
        rule = process_stack.gate_rule_for_layer_name(element.layer_name)

        patterned_regions.append(
            PatternedRegion(
                name=element.name,
                gate_type=element.gate_type,
                layer_name=element.layer_name,
                material=rule.material,
                voltage_label=element.voltage_label or rule.default_voltage_label,
                x_min_nm=element.x_min_nm,
                x_max_nm=element.x_max_nm,
                y_min_nm=element.y_min_nm,
                y_max_nm=element.y_max_nm,
                z_min_nm=rule.z_min_nm,
                z_max_nm=rule.z_max_nm,
                gds_layer=element.gds_layer,
                gds_datatype=element.gds_datatype,
                source_rule_name=rule.name,
                params=dict(element.params),
                polygon_xy_nm=element.polygon_xy_nm,
                notes=element.notes or rule.notes,
            )
        )

    return SimulationLayout(
        name=name,
        z_reference_name=process_stack.z_reference_name,
        domain=domain,
        background_regions=background_regions,
        patterned_regions=patterned_regions,
    )
