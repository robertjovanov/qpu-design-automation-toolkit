from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from phidl import Device

from .metadata import ComponentMetadata


@dataclass
class BaseComponent(ABC):
    """
    Base semantic wrapper around a PHIDL Device.

    Concrete subclasses:
    - store metadata
    - generate their own geometry
    - expose ports
    - can be added into a larger PHIDL Device
    """

    metadata: ComponentMetadata
    params: Dict[str, Any] = field(default_factory=dict)

    device: Device = field(init=False, repr=False)
    _built: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.device = Device(self.metadata.name)

    @abstractmethod
    def build(self) -> Device:
        """
        Populate self.device with geometry and ports.
        """
        raise NotImplementedError

    def ensure_built(self) -> Device:
        if not self._built:
            self.build()
            self._built = True
        return self.device

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def ports(self):
        return self.ensure_built().ports

    def add_to(self, parent: Device):
        """
        Add this component to a parent PHIDL Device as a reference.
        """
        return parent.add_ref(self.ensure_built())

    def bbox(self) -> Tuple[float, float, float, float]:
        """
        Return bounding box as (xmin, ymin, xmax, ymax).
        """
        d = self.ensure_built()
        return (d.xmin, d.ymin, d.xmax, d.ymax)

    def summary(self) -> Dict[str, Any]:
        """
        Small serializable summary useful for debugging/export.
        """
        return {
            "name": self.metadata.name,
            "gate_type": self.metadata.gate_type.value,
            "layer_name": self.metadata.layer.name,
            "gds_layer": self.metadata.layer.gds_layer,
            "gds_datatype": self.metadata.layer.gds_datatype,
            "voltage_label": self.metadata.voltage_label,
            "role": self.metadata.role,
            "z_bottom_nm": self.metadata.z_bottom_nm,
            "z_top_nm": self.metadata.z_top_nm,
            "notes": self.metadata.notes,
            "params": self.params,
        }