from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from ..core.component import BaseComponent


@dataclass
class PlacedElement:
    """
    Serializable description of one placed layout element.
    """

    name: str
    gate_type: str
    layer_name: str
    gds_layer: int
    gds_datatype: int
    voltage_label: Optional[str]
    role: Optional[str]
    notes: Optional[str]

    x_min_nm: float
    y_min_nm: float
    x_max_nm: float
    y_max_nm: float

    center_x_nm: float
    center_y_nm: float
    width_nm: float
    height_nm: float

    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def placed_element_from_component(component: BaseComponent, ref) -> PlacedElement:
    """
    Create a PlacedElement from a component and its placed PHIDL reference.
    """
    md = component.metadata

    x_min = float(ref.xmin)
    y_min = float(ref.ymin)
    x_max = float(ref.xmax)
    y_max = float(ref.ymax)

    return PlacedElement(
        name=md.name,
        gate_type=md.gate_type.value,
        layer_name=md.layer.name,
        gds_layer=md.layer.gds_layer,
        gds_datatype=md.layer.gds_datatype,
        voltage_label=md.voltage_label,
        role=md.role,
        notes=md.notes,
        x_min_nm=x_min,
        y_min_nm=y_min,
        x_max_nm=x_max,
        y_max_nm=y_max,
        center_x_nm=0.5 * (x_min + x_max),
        center_y_nm=0.5 * (y_min + y_max),
        width_nm=x_max - x_min,
        height_nm=y_max - y_min,
        params=dict(component.params),
    )