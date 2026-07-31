from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Union
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
