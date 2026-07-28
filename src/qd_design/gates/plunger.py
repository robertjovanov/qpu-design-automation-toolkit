from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from phidl import Device

from ..core.component import BaseComponent


@dataclass(frozen=True)
class LollipopPlungerGeometry:
    """Shared dimensional contract for comparable lollipop plunger gates."""

    body_width_nm: float = 40.0
    body_length_nm: float = 50.0
    head_top_width_nm: float = 60.0
    head_max_width_nm: float = 100.0
    head_height_nm: float = 100.0
    upper_taper_height_nm: float = 25.0
    lower_taper_height_nm: float = 25.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


CANONICAL_LOLLIPOP_PLUNGER = LollipopPlungerGeometry()


@dataclass
class PlungerGate(BaseComponent):
    """
    Faceted plunger gate defined as a single polygon.

    Geometry:
    - narrow rectangular body at the bottom
    - widened polygonal head on top
    - one connected, symmetric shape

    Parameters
    ----------
    body_width_nm : float
        Width of the bottom rectangular body along x.
    body_length_nm : float
        Length of the bottom rectangular body along y.
    head_top_width_nm : float
        Width of the flat top edge of the head.
    head_max_width_nm : float
        Maximum width of the head.
    head_height_nm : float
        Total height of the faceted head.
    upper_taper_height_nm : float
        Vertical height of the upper tapered section.
    lower_taper_height_nm : float
        Vertical height of the lower tapered section.
    """

    body_width_nm: float = CANONICAL_LOLLIPOP_PLUNGER.body_width_nm
    body_length_nm: float = CANONICAL_LOLLIPOP_PLUNGER.body_length_nm

    head_top_width_nm: float = CANONICAL_LOLLIPOP_PLUNGER.head_top_width_nm
    head_max_width_nm: float = CANONICAL_LOLLIPOP_PLUNGER.head_max_width_nm
    head_height_nm: float = CANONICAL_LOLLIPOP_PLUNGER.head_height_nm

    upper_taper_height_nm: float = CANONICAL_LOLLIPOP_PLUNGER.upper_taper_height_nm
    lower_taper_height_nm: float = CANONICAL_LOLLIPOP_PLUNGER.lower_taper_height_nm

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params.update(
            {
                "body_width_nm": self.body_width_nm,
                "body_length_nm": self.body_length_nm,
                "head_top_width_nm": self.head_top_width_nm,
                "head_max_width_nm": self.head_max_width_nm,
                "head_height_nm": self.head_height_nm,
                "upper_taper_height_nm": self.upper_taper_height_nm,
                "lower_taper_height_nm": self.lower_taper_height_nm,
            }
        )

    def outline_points(self) -> List[Tuple[float, float]]:
        """Return the canonical body-plus-faceted-head polygon before placement."""
        bw = self.body_width_nm
        bl = self.body_length_nm
        top_w = self.head_top_width_nm
        max_w = self.head_max_width_nm
        hh = self.head_height_nm
        h_upper = self.upper_taper_height_nm
        h_lower = self.lower_taper_height_nm

        if bw <= 0 or bl < 0:
            raise ValueError("body_width_nm must be > 0 and body_length_nm must be >= 0.")
        if top_w <= 0 or max_w <= 0 or hh <= 0:
            raise ValueError("Head dimensions must be positive.")
        if top_w > max_w:
            raise ValueError("head_top_width_nm must be <= head_max_width_nm.")
        if bw > max_w:
            raise ValueError("body_width_nm must be <= head_max_width_nm.")
        if h_upper < 0 or h_lower < 0:
            raise ValueError("Taper heights must be >= 0.")
        if h_upper + h_lower > hh:
            raise ValueError(
                "upper_taper_height_nm + lower_taper_height_nm must be <= head_height_nm."
            )

        y0 = 0.0
        y1 = bl
        y2 = bl + h_lower
        y3 = bl + hh - h_upper
        y4 = bl + hh

        half_bw = bw / 2.0
        half_top = top_w / 2.0
        half_max = max_w / 2.0

        return [
            (-half_bw, y0),
            (half_bw, y0),
            (half_bw, y1),
            (half_max, y2),
            (half_max, y3),
            (half_top, y4),
            (-half_top, y4),
            (-half_max, y3),
            (-half_max, y2),
            (-half_bw, y1),
        ]

    def build(self) -> Device:
        layer = self.metadata.layer.phidl_layer
        polygon_points = self.outline_points()

        self.device.add_polygon(polygon_points, layer=layer)

        bl = self.body_length_nm
        hh = self.head_height_nm
        h_upper = self.upper_taper_height_nm
        h_lower = self.lower_taper_height_nm
        bw = self.body_width_nm
        top_w = self.head_top_width_nm
        max_w = self.head_max_width_nm
        y0 = 0.0
        y2 = bl + h_lower
        y3 = bl + hh - h_upper
        y4 = bl + hh
        half_max = max_w / 2.0
        y_side_mid = 0.5 * (y2 + y3)

        self.device.add_port(
            name="north",
            midpoint=(0.0, y4),
            width=top_w,
            orientation=90,
        )
        self.device.add_port(
            name="south",
            midpoint=(0.0, y0),
            width=bw,
            orientation=270,
        )
        self.device.add_port(
            name="west",
            midpoint=(-half_max, y_side_mid),
            width=max(y3 - y2, 1e-3),
            orientation=180,
        )
        self.device.add_port(
            name="east",
            midpoint=(half_max, y_side_mid),
            width=max(y3 - y2, 1e-3),
            orientation=0,
        )

        return self.device
