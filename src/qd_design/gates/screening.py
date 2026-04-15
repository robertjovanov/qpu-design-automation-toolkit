from dataclasses import dataclass

from phidl import Device

from ..core.component import BaseComponent


@dataclass
class ScreeningGate(BaseComponent):
    """
    Rectangular screening gate.

    Parameters
    ----------
    width_nm : float
        Width along x.
    length_nm : float
        Length along y.
    """

    width_nm: float = 40.0
    length_nm: float = 200.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params.update(
            {
                "width_nm": self.width_nm,
                "length_nm": self.length_nm,
            }
        )

    def build(self) -> Device:
        if self.width_nm <= 0 or self.length_nm <= 0:
            raise ValueError("ScreeningGate dimensions must be positive.")

        layer = self.metadata.layer.phidl_layer
        w = self.width_nm
        l = self.length_nm

        points = [
            (0.0, -l / 2),
            (w, -l / 2),
            (w, l / 2),
            (0.0, l / 2),
        ]
        self.device.add_polygon(points, layer=layer)

        self.device.add_port(
            name="west",
            midpoint=(0.0, 0.0),
            width=l,
            orientation=180,
        )
        self.device.add_port(
            name="east",
            midpoint=(w, 0.0),
            width=l,
            orientation=0,
        )

        return self.device