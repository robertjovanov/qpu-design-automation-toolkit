from dataclasses import dataclass
from typing import Optional

from .layers import LayerSpec
from .types import GateType


@dataclass
class ComponentMetadata:
    """
    Semantic and simulation-relevant metadata attached to a component.
    """

    name: str
    gate_type: GateType
    layer: LayerSpec

    voltage_label: Optional[str] = None
    role: Optional[str] = None

    z_bottom_nm: Optional[float] = None
    z_top_nm: Optional[float] = None

    notes: Optional[str] = None