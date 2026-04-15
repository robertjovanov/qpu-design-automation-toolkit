from dataclasses import dataclass
import numpy as np

from phidl import Device

from ..core.component import BaseComponent


@dataclass
class SensorGate(BaseComponent):
    """
    Sensor gate with:
    - rectangular body at the bottom
    - elliptical head on top
    - one connected polygonal boundary

    Parameters
    ----------
    body_width_nm : float
        Width of the bottom rectangular body along x.
    body_length_nm : float
        Length of the bottom rectangular body along y.
    head_axis_a_nm : float
        Semi-axis of the ellipse along x.
    head_axis_b_nm : float
        Semi-axis of the ellipse along y.
    n_head_points : int
        Number of points used to approximate the ellipse boundary.
    overlap_nm : float
        Small overlap between body and ellipse.
    """

    body_width_nm: float = 40.0
    body_length_nm: float = 80.0
    head_axis_a_nm: float = 45.0
    head_axis_b_nm: float = 30.0
    n_head_points: int = 128
    overlap_nm: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params.update(
            {
                "body_width_nm": self.body_width_nm,
                "body_length_nm": self.body_length_nm,
                "head_axis_a_nm": self.head_axis_a_nm,
                "head_axis_b_nm": self.head_axis_b_nm,
                "n_head_points": self.n_head_points,
                "overlap_nm": self.overlap_nm,
            }
        )

    def build(self) -> Device:
        bw = self.body_width_nm
        bl = self.body_length_nm
        a = self.head_axis_a_nm
        b = self.head_axis_b_nm
        n = self.n_head_points
        overlap = self.overlap_nm

        if bw <= 0 or bl < 0:
            raise ValueError("body_width_nm must be > 0 and body_length_nm must be >= 0.")
        if a <= 0 or b <= 0:
            raise ValueError("Ellipse semi-axes must be positive.")
        if bw > 2 * a:
            raise ValueError("body_width_nm must be <= 2 * head_axis_a_nm.")
        if n < 32:
            raise ValueError("n_head_points must be at least 32.")
        if overlap < 0:
            raise ValueError("overlap_nm must be >= 0.")

        layer = self.metadata.layer.phidl_layer

        half_bw = bw / 2.0
        y0 = 0.0
        y1 = bl

        # Ellipse center placed above the body so the two shapes connect
        y_center = bl + b - overlap

        # Attachment points between body sides and the ellipse
        x_attach = half_bw
        y_attach_local = -b * np.sqrt(1.0 - (x_attach**2 / a**2))
        y_attach = y_center + y_attach_local

        # Parameter angles for the ellipse arc from right attachment -> top -> left attachment
        t_right = np.arctan2(y_attach_local / b, x_attach / a)
        t_left = np.pi - t_right

        angles = np.linspace(t_right, t_left, n)
        arc_points = [(a * np.cos(t), y_center + b * np.sin(t)) for t in angles]

        polygon_points = [
            (-half_bw, y0),
            (half_bw, y0),
            (half_bw, y1),
            *arc_points,
            (-half_bw, y1),
        ]

        self.device.add_polygon(polygon_points, layer=layer)

        self.device.add_port(
            name="north",
            midpoint=(0.0, y_center + b),
            width=bw,
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
            midpoint=(-a, y_center),
            width=2 * b,
            orientation=180,
        )
        self.device.add_port(
            name="east",
            midpoint=(a, y_center),
            width=2 * b,
            orientation=0,
        )

        return self.device