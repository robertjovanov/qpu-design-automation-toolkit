from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json

from phidl import Device

from ..core import (
    GateType,
    BARRIER,
    OHMIC,
    PLUNGER,
    ComponentMetadata,
)
from ..gates import BarrierGate, OhmicContact, PlungerGate
from ..export import PlacedElement, placed_element_from_component


@dataclass
class SquareDotArrayDevice:
    """
    Reusable builder for a 2x2 quantum-dot array.

    The geometry is composed from two 1x2 linear arrays. The bottom row uses the
    same orientation as ``LinearDotArrayDevice(n_dots=2)``. The top row is the
    180-degree-facing partner placed above it.
    """

    name: str = "square_dot_array"

    device_y_size_nm: float = 200.0
    row_gap_nm: float = 0.0

    ohmic_width_nm: float = 40.0
    ohmic_length_nm: float = 200.0

    barrier_width_nm: float = 40.0
    barrier_length_nm: float = 140.0

    plunger_body_width_nm: float = 40.0
    plunger_body_length_nm: float = 50.0
    plunger_head_top_width_nm: float = 60.0
    plunger_head_max_width_nm: float = 100.0
    plunger_head_height_nm: float = 100.0
    plunger_upper_taper_height_nm: float = 25.0
    plunger_lower_taper_height_nm: float = 25.0

    ohmic_to_barrier_gap_nm: float = 20.0
    barrier_to_plunger_gap_nm: float = 20.0

    device: Device = field(init=False, repr=False)
    refs: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    elements: Dict[str, PlacedElement] = field(init=False, default_factory=dict, repr=False)
    _built: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.device = Device(self.name)

    @property
    def n_columns(self) -> int:
        return 2

    @property
    def n_rows(self) -> int:
        return 2

    @property
    def row_pitch_y_nm(self) -> float:
        return self.device_y_size_nm + self.row_gap_nm

    @property
    def plunger_total_height_nm(self) -> float:
        return self.plunger_body_length_nm + self.plunger_head_height_nm

    @property
    def plunger_pitch_x_nm(self) -> float:
        return (
            self.plunger_head_max_width_nm
            + self.barrier_width_nm
            + 2.0 * self.barrier_to_plunger_gap_nm
        )

    @property
    def facing_plunger_gap_nm(self) -> float:
        return (
            2.0 * (self.device_y_size_nm - self.plunger_total_height_nm)
            + self.row_gap_nm
        )

    def _validate(self) -> None:
        if self.device_y_size_nm <= 0:
            raise ValueError("device_y_size_nm must be positive.")
        if self.row_gap_nm < 0:
            raise ValueError("row_gap_nm must be non-negative.")
        if self.ohmic_width_nm <= 0 or self.ohmic_length_nm <= 0:
            raise ValueError("Ohmic dimensions must be positive.")
        if self.barrier_width_nm <= 0 or self.barrier_length_nm <= 0:
            raise ValueError("Barrier dimensions must be positive.")
        if self.ohmic_to_barrier_gap_nm < 0 or self.barrier_to_plunger_gap_nm < 0:
            raise ValueError("Horizontal gaps must be non-negative.")
        if self.plunger_total_height_nm > self.device_y_size_nm:
            raise ValueError(
                "The plunger body length plus head height must fit inside device_y_size_nm."
            )

    def _make_ohmic(self, *, name: str, voltage_label: str) -> OhmicContact:
        return OhmicContact(
            metadata=ComponentMetadata(
                name=name,
                gate_type=GateType.OHMIC,
                layer=OHMIC,
                voltage_label=voltage_label,
            ),
            width_nm=self.ohmic_width_nm,
            length_nm=self.ohmic_length_nm,
        )

    def _make_barrier(self, *, name: str, voltage_label: str) -> BarrierGate:
        return BarrierGate(
            metadata=ComponentMetadata(
                name=name,
                gate_type=GateType.BARRIER,
                layer=BARRIER,
                voltage_label=voltage_label,
            ),
            width_nm=self.barrier_width_nm,
            length_nm=self.barrier_length_nm,
        )

    def _make_plunger(self, *, name: str, voltage_label: str) -> PlungerGate:
        return PlungerGate(
            metadata=ComponentMetadata(
                name=name,
                gate_type=GateType.PLUNGER,
                layer=PLUNGER,
                voltage_label=voltage_label,
            ),
            body_width_nm=self.plunger_body_width_nm,
            body_length_nm=self.plunger_body_length_nm,
            head_top_width_nm=self.plunger_head_top_width_nm,
            head_max_width_nm=self.plunger_head_max_width_nm,
            head_height_nm=self.plunger_head_height_nm,
            upper_taper_height_nm=self.plunger_upper_taper_height_nm,
            lower_taper_height_nm=self.plunger_lower_taper_height_nm,
        )

    def _plunger_centers_x(self) -> List[float]:
        return [
            (i - 0.5 * (self.n_columns - 1)) * self.plunger_pitch_x_nm
            for i in range(self.n_columns)
        ]

    def _barrier_x_lefts(self, plunger_centers_x: List[float]) -> List[float]:
        half_max = self.plunger_head_max_width_nm / 2.0

        x_lefts = [
            plunger_centers_x[0]
            - half_max
            - self.barrier_to_plunger_gap_nm
            - self.barrier_width_nm
        ]

        for i in range(1, self.n_columns):
            x_lefts.append(
                plunger_centers_x[i - 1]
                + half_max
                + self.barrier_to_plunger_gap_nm
            )

        x_lefts.append(
            plunger_centers_x[-1]
            + half_max
            + self.barrier_to_plunger_gap_nm
        )

        return x_lefts

    def _place_linear_row(
        self,
        *,
        row_index: int,
        y_min_nm: float,
        plunger_faces: str,
    ) -> None:
        if plunger_faces not in {"up", "down"}:
            raise ValueError("plunger_faces must be either 'up' or 'down'.")

        y_max_nm = y_min_nm + self.device_y_size_nm
        plunger_centers_x = self._plunger_centers_x()
        barrier_x_lefts = self._barrier_x_lefts(plunger_centers_x)

        plunger_start = (row_index - 1) * self.n_columns + 1
        barrier_start = (row_index - 1) * (self.n_columns + 1) + 1

        left_ohmic = self._make_ohmic(
            name=f"OC_L_{row_index}",
            voltage_label=f"V_OC_L_{row_index}",
        )
        right_ohmic = self._make_ohmic(
            name=f"OC_R_{row_index}",
            voltage_label=f"V_OC_R_{row_index}",
        )
        barriers = [
            self._make_barrier(
                name=f"B{barrier_start + i}",
                voltage_label=f"V_B{barrier_start + i}",
            )
            for i in range(self.n_columns + 1)
        ]
        plungers = [
            self._make_plunger(
                name=f"P{plunger_start + i}",
                voltage_label=f"V_P{plunger_start + i}",
            )
            for i in range(self.n_columns)
        ]

        ref_oc_l = left_ohmic.add_to(self.device)
        ref_oc_r = right_ohmic.add_to(self.device)
        barrier_refs = [barrier.add_to(self.device) for barrier in barriers]
        plunger_refs = [plunger.add_to(self.device) for plunger in plungers]

        x_ocl_left = (
            barrier_x_lefts[0]
            - self.ohmic_to_barrier_gap_nm
            - self.ohmic_width_nm
        )
        x_ocr_left = (
            barrier_x_lefts[-1]
            + self.barrier_width_nm
            + self.ohmic_to_barrier_gap_nm
        )

        ref_oc_l.move(
            origin=(ref_oc_l.xmin, ref_oc_l.ymin),
            destination=(x_ocl_left, y_min_nm),
        )
        ref_oc_r.move(
            origin=(ref_oc_r.xmin, ref_oc_r.ymin),
            destination=(x_ocr_left, y_min_nm),
        )

        if plunger_faces == "up":
            barrier_y_min = y_min_nm
            for ref_p, xc in zip(plunger_refs, plunger_centers_x):
                ref_p.move(
                    origin=(0.0, ref_p.ymin),
                    destination=(xc, y_min_nm),
                )
        else:
            barrier_y_min = y_max_nm - self.barrier_length_nm
            for ref_p, xc in zip(plunger_refs, plunger_centers_x):
                ref_p.rotate(180, center=(0, 0))
                ref_p.move(
                    origin=(0.0, ref_p.ymax),
                    destination=(xc, y_max_nm),
                )

        for ref_b, x_left in zip(barrier_refs, barrier_x_lefts):
            ref_b.move(
                origin=(ref_b.xmin, ref_b.ymin),
                destination=(x_left, barrier_y_min),
            )

        row_key = f"row_{row_index}"
        self.refs.update(
            {
                f"{row_key}_left_ohmic": ref_oc_l,
                f"{row_key}_right_ohmic": ref_oc_r,
                f"{row_key}_barriers": barrier_refs,
                f"{row_key}_plungers": plunger_refs,
            }
        )

        self.elements[left_ohmic.name] = placed_element_from_component(left_ohmic, ref_oc_l)
        for barrier, ref_b in zip(barriers, barrier_refs):
            self.elements[barrier.name] = placed_element_from_component(barrier, ref_b)
        for plunger, ref_p in zip(plungers, plunger_refs):
            self.elements[plunger.name] = placed_element_from_component(plunger, ref_p)
        self.elements[right_ohmic.name] = placed_element_from_component(right_ohmic, ref_oc_r)

    def build(self) -> Device:
        if self._built:
            return self.device

        self._validate()

        self._place_linear_row(
            row_index=1,
            y_min_nm=0.0,
            plunger_faces="up",
        )
        self._place_linear_row(
            row_index=2,
            y_min_nm=self.row_pitch_y_nm,
            plunger_faces="down",
        )

        self.refs.update(
            {
                "row_pitch_y_nm": self.row_pitch_y_nm,
                "plunger_pitch_x_nm": self.plunger_pitch_x_nm,
                "facing_plunger_gap_nm": self.facing_plunger_gap_nm,
            }
        )

        self._built = True
        return self.device

    def ensure_built(self) -> Device:
        return self.build()

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "device_y_size_nm": self.device_y_size_nm,
            "row_gap_nm": self.row_gap_nm,
            "row_pitch_y_nm": self.row_pitch_y_nm,
            "plunger_pitch_x_nm": self.plunger_pitch_x_nm,
            "facing_plunger_gap_nm": self.facing_plunger_gap_nm,
            "ohmic_width_nm": self.ohmic_width_nm,
            "ohmic_length_nm": self.ohmic_length_nm,
            "barrier_width_nm": self.barrier_width_nm,
            "barrier_length_nm": self.barrier_length_nm,
            "plunger_body_width_nm": self.plunger_body_width_nm,
            "plunger_body_length_nm": self.plunger_body_length_nm,
            "plunger_head_top_width_nm": self.plunger_head_top_width_nm,
            "plunger_head_max_width_nm": self.plunger_head_max_width_nm,
            "plunger_head_height_nm": self.plunger_head_height_nm,
            "plunger_upper_taper_height_nm": self.plunger_upper_taper_height_nm,
            "plunger_lower_taper_height_nm": self.plunger_lower_taper_height_nm,
            "ohmic_to_barrier_gap_nm": self.ohmic_to_barrier_gap_nm,
            "barrier_to_plunger_gap_nm": self.barrier_to_plunger_gap_nm,
        }

    def layout_elements(self) -> Dict[str, PlacedElement]:
        self.ensure_built()
        return self.elements

    def layout_spec(self) -> list[dict[str, Any]]:
        self.ensure_built()
        return [element.to_dict() for element in self.elements.values()]

    def write_layout_spec_json(self, filepath: str) -> None:
        self.ensure_built()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.layout_spec(), f, indent=2)

    def write_gds(self, filepath: str) -> None:
        self.ensure_built().write_gds(filepath)

    def write_svg(self, filepath: str) -> None:
        self.ensure_built().write_svg(filepath)


@dataclass
class SeparatedSquareDotArrayDevice(SquareDotArrayDevice):
    """
    2x2 square array derived from the separated vertical double-dot motif.

    The row geometry is the same as ``SquareDotArrayDevice``: two bottom
    plungers face upward and two top plungers face downward. Compared with the
    base array, ohmics are shortened and aligned to the outer row boundaries,
    leaving the central gap clear. Two horizontal separator barriers sit in
    that gap, one spanning the left column and one spanning the right column.
    """

    name: str = "separated_square_dot_array"
    ohmic_length_nm: Optional[float] = None

    left_separator_barrier_name: str = "B_SEP_L"
    right_separator_barrier_name: str = "B_SEP_R"
    left_separator_barrier_voltage_label: str = "V_B_SEP_L"
    right_separator_barrier_voltage_label: str = "V_B_SEP_R"

    left_separator_start_x_nm: Optional[float] = None
    left_separator_end_x_nm: Optional[float] = None
    right_separator_start_x_nm: Optional[float] = None
    right_separator_end_x_nm: Optional[float] = None

    @property
    def effective_ohmic_length_nm(self) -> float:
        if self.ohmic_length_nm is None:
            return self.plunger_total_height_nm
        return float(self.ohmic_length_nm)

    @property
    def bottom_plunger_top_y_nm(self) -> float:
        return self.plunger_total_height_nm

    @property
    def top_plunger_bottom_y_nm(self) -> float:
        return self.row_pitch_y_nm + self.device_y_size_nm - self.plunger_total_height_nm

    @property
    def separator_center_y_nm(self) -> float:
        return 0.5 * (self.bottom_plunger_top_y_nm + self.top_plunger_bottom_y_nm)

    @property
    def separator_y_min_nm(self) -> float:
        return self.separator_center_y_nm - 0.5 * self.barrier_width_nm

    @property
    def array_left_boundary_x_nm(self) -> float:
        barrier_x_lefts = self._barrier_x_lefts(self._plunger_centers_x())
        return barrier_x_lefts[0] - self.ohmic_to_barrier_gap_nm - self.ohmic_width_nm

    @property
    def array_right_boundary_x_nm(self) -> float:
        barrier_x_lefts = self._barrier_x_lefts(self._plunger_centers_x())
        return (
            barrier_x_lefts[-1]
            + self.barrier_width_nm
            + self.ohmic_to_barrier_gap_nm
            + self.ohmic_width_nm
        )

    @property
    def resolved_left_separator_start_x_nm(self) -> float:
        if self.left_separator_start_x_nm is not None:
            return float(self.left_separator_start_x_nm)
        return self.array_left_boundary_x_nm

    @property
    def resolved_left_separator_end_x_nm(self) -> float:
        if self.left_separator_end_x_nm is not None:
            return float(self.left_separator_end_x_nm)
        return self._plunger_centers_x()[0] + 0.5 * self.plunger_head_max_width_nm

    @property
    def resolved_right_separator_start_x_nm(self) -> float:
        if self.right_separator_start_x_nm is not None:
            return float(self.right_separator_start_x_nm)
        return self._plunger_centers_x()[-1] - 0.5 * self.plunger_head_max_width_nm

    @property
    def resolved_right_separator_end_x_nm(self) -> float:
        if self.right_separator_end_x_nm is not None:
            return float(self.right_separator_end_x_nm)
        return self.array_right_boundary_x_nm

    @property
    def left_separator_length_nm(self) -> float:
        return self.resolved_left_separator_end_x_nm - self.resolved_left_separator_start_x_nm

    @property
    def right_separator_length_nm(self) -> float:
        return self.resolved_right_separator_end_x_nm - self.resolved_right_separator_start_x_nm

    def _validate(self) -> None:
        if self.device_y_size_nm <= 0:
            raise ValueError("device_y_size_nm must be positive.")
        if self.row_gap_nm < 0:
            raise ValueError("row_gap_nm must be non-negative.")
        if self.ohmic_width_nm <= 0 or self.effective_ohmic_length_nm <= 0:
            raise ValueError("Ohmic dimensions must be positive.")
        if self.effective_ohmic_length_nm > self.device_y_size_nm:
            raise ValueError("The effective ohmic length must fit inside device_y_size_nm.")
        if self.barrier_width_nm <= 0 or self.barrier_length_nm <= 0:
            raise ValueError("Barrier dimensions must be positive.")
        if self.ohmic_to_barrier_gap_nm < 0 or self.barrier_to_plunger_gap_nm < 0:
            raise ValueError("Horizontal gaps must be non-negative.")
        if self.plunger_total_height_nm > self.device_y_size_nm:
            raise ValueError(
                "The plunger body length plus head height must fit inside device_y_size_nm."
            )
        if self.facing_plunger_gap_nm <= 0:
            raise ValueError("The facing plunger gap must be positive for separator barriers.")
        if self.barrier_width_nm > self.facing_plunger_gap_nm:
            raise ValueError(
                "The separator barrier width must fit in the gap between facing plungers."
            )
        if self.left_separator_length_nm <= 0:
            raise ValueError("The left separator barrier end x must be greater than its start x.")
        if self.right_separator_length_nm <= 0:
            raise ValueError("The right separator barrier end x must be greater than its start x.")

    def _make_ohmic(self, *, name: str, voltage_label: str) -> OhmicContact:
        return OhmicContact(
            metadata=ComponentMetadata(
                name=name,
                gate_type=GateType.OHMIC,
                layer=OHMIC,
                voltage_label=voltage_label,
            ),
            width_nm=self.ohmic_width_nm,
            length_nm=self.effective_ohmic_length_nm,
        )

    def _make_separator_barrier(
        self,
        *,
        name: str,
        voltage_label: str,
        length_nm: float,
    ) -> BarrierGate:
        return BarrierGate(
            metadata=ComponentMetadata(
                name=name,
                gate_type=GateType.BARRIER,
                layer=BARRIER,
                voltage_label=voltage_label,
            ),
            width_nm=length_nm,
            length_nm=self.barrier_width_nm,
        )

    def _place_linear_row(
        self,
        *,
        row_index: int,
        y_min_nm: float,
        plunger_faces: str,
    ) -> None:
        if plunger_faces not in {"up", "down"}:
            raise ValueError("plunger_faces must be either 'up' or 'down'.")

        y_max_nm = y_min_nm + self.device_y_size_nm
        plunger_centers_x = self._plunger_centers_x()
        barrier_x_lefts = self._barrier_x_lefts(plunger_centers_x)

        plunger_start = (row_index - 1) * self.n_columns + 1
        barrier_start = (row_index - 1) * (self.n_columns + 1) + 1

        left_ohmic = self._make_ohmic(
            name=f"OC_L_{row_index}",
            voltage_label=f"V_OC_L_{row_index}",
        )
        right_ohmic = self._make_ohmic(
            name=f"OC_R_{row_index}",
            voltage_label=f"V_OC_R_{row_index}",
        )
        barriers = [
            self._make_barrier(
                name=f"B{barrier_start + i}",
                voltage_label=f"V_B{barrier_start + i}",
            )
            for i in range(self.n_columns + 1)
        ]
        plungers = [
            self._make_plunger(
                name=f"P{plunger_start + i}",
                voltage_label=f"V_P{plunger_start + i}",
            )
            for i in range(self.n_columns)
        ]

        ref_oc_l = left_ohmic.add_to(self.device)
        ref_oc_r = right_ohmic.add_to(self.device)
        barrier_refs = [barrier.add_to(self.device) for barrier in barriers]
        plunger_refs = [plunger.add_to(self.device) for plunger in plungers]

        x_ocl_left = (
            barrier_x_lefts[0]
            - self.ohmic_to_barrier_gap_nm
            - self.ohmic_width_nm
        )
        x_ocr_left = (
            barrier_x_lefts[-1]
            + self.barrier_width_nm
            + self.ohmic_to_barrier_gap_nm
        )

        if plunger_faces == "up":
            ohmic_y_min = y_min_nm
            barrier_y_min = y_min_nm
            for ref_p, xc in zip(plunger_refs, plunger_centers_x):
                ref_p.move(
                    origin=(0.0, ref_p.ymin),
                    destination=(xc, y_min_nm),
                )
        else:
            ohmic_y_min = y_max_nm - self.effective_ohmic_length_nm
            barrier_y_min = y_max_nm - self.barrier_length_nm
            for ref_p, xc in zip(plunger_refs, plunger_centers_x):
                ref_p.rotate(180, center=(0, 0))
                ref_p.move(
                    origin=(0.0, ref_p.ymax),
                    destination=(xc, y_max_nm),
                )

        ref_oc_l.move(
            origin=(ref_oc_l.xmin, ref_oc_l.ymin),
            destination=(x_ocl_left, ohmic_y_min),
        )
        ref_oc_r.move(
            origin=(ref_oc_r.xmin, ref_oc_r.ymin),
            destination=(x_ocr_left, ohmic_y_min),
        )

        for ref_b, x_left in zip(barrier_refs, barrier_x_lefts):
            ref_b.move(
                origin=(ref_b.xmin, ref_b.ymin),
                destination=(x_left, barrier_y_min),
            )

        row_key = f"row_{row_index}"
        self.refs.update(
            {
                f"{row_key}_left_ohmic": ref_oc_l,
                f"{row_key}_right_ohmic": ref_oc_r,
                f"{row_key}_barriers": barrier_refs,
                f"{row_key}_plungers": plunger_refs,
            }
        )

        self.elements[left_ohmic.name] = placed_element_from_component(left_ohmic, ref_oc_l)
        for barrier, ref_b in zip(barriers, barrier_refs):
            self.elements[barrier.name] = placed_element_from_component(barrier, ref_b)
        for plunger, ref_p in zip(plungers, plunger_refs):
            self.elements[plunger.name] = placed_element_from_component(plunger, ref_p)
        self.elements[right_ohmic.name] = placed_element_from_component(right_ohmic, ref_oc_r)

    def _place_separator_barriers(self) -> None:
        left_separator = self._make_separator_barrier(
            name=self.left_separator_barrier_name,
            voltage_label=self.left_separator_barrier_voltage_label,
            length_nm=self.left_separator_length_nm,
        )
        right_separator = self._make_separator_barrier(
            name=self.right_separator_barrier_name,
            voltage_label=self.right_separator_barrier_voltage_label,
            length_nm=self.right_separator_length_nm,
        )

        ref_left = left_separator.add_to(self.device)
        ref_right = right_separator.add_to(self.device)

        ref_left.move(
            origin=(ref_left.xmin, ref_left.ymin),
            destination=(self.resolved_left_separator_start_x_nm, self.separator_y_min_nm),
        )
        ref_right.move(
            origin=(ref_right.xmin, ref_right.ymin),
            destination=(self.resolved_right_separator_start_x_nm, self.separator_y_min_nm),
        )

        self.refs.update(
            {
                "left_separator_barrier": ref_left,
                "right_separator_barrier": ref_right,
            }
        )
        self.elements[left_separator.name] = placed_element_from_component(left_separator, ref_left)
        self.elements[right_separator.name] = placed_element_from_component(right_separator, ref_right)

    def build(self) -> Device:
        if self._built:
            return self.device

        self._validate()

        self._place_linear_row(
            row_index=1,
            y_min_nm=0.0,
            plunger_faces="up",
        )
        self._place_linear_row(
            row_index=2,
            y_min_nm=self.row_pitch_y_nm,
            plunger_faces="down",
        )
        self._place_separator_barriers()

        self.refs.update(
            {
                "row_pitch_y_nm": self.row_pitch_y_nm,
                "plunger_pitch_x_nm": self.plunger_pitch_x_nm,
                "facing_plunger_gap_nm": self.facing_plunger_gap_nm,
                "effective_ohmic_length_nm": self.effective_ohmic_length_nm,
                "separator_center_y_nm": self.separator_center_y_nm,
                "left_separator_length_nm": self.left_separator_length_nm,
                "right_separator_length_nm": self.right_separator_length_nm,
            }
        )

        self._built = True
        return self.device

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "device_y_size_nm": self.device_y_size_nm,
            "row_gap_nm": self.row_gap_nm,
            "row_pitch_y_nm": self.row_pitch_y_nm,
            "plunger_pitch_x_nm": self.plunger_pitch_x_nm,
            "facing_plunger_gap_nm": self.facing_plunger_gap_nm,
            "ohmic_width_nm": self.ohmic_width_nm,
            "ohmic_length_nm": self.effective_ohmic_length_nm,
            "barrier_width_nm": self.barrier_width_nm,
            "barrier_length_nm": self.barrier_length_nm,
            "left_separator_barrier_name": self.left_separator_barrier_name,
            "right_separator_barrier_name": self.right_separator_barrier_name,
            "left_separator_barrier_voltage_label": self.left_separator_barrier_voltage_label,
            "right_separator_barrier_voltage_label": self.right_separator_barrier_voltage_label,
            "left_separator_start_x_nm": self.resolved_left_separator_start_x_nm,
            "left_separator_end_x_nm": self.resolved_left_separator_end_x_nm,
            "left_separator_length_nm": self.left_separator_length_nm,
            "right_separator_start_x_nm": self.resolved_right_separator_start_x_nm,
            "right_separator_end_x_nm": self.resolved_right_separator_end_x_nm,
            "right_separator_length_nm": self.right_separator_length_nm,
            "separator_center_y_nm": self.separator_center_y_nm,
            "separator_y_min_nm": self.separator_y_min_nm,
            "plunger_body_width_nm": self.plunger_body_width_nm,
            "plunger_body_length_nm": self.plunger_body_length_nm,
            "plunger_head_top_width_nm": self.plunger_head_top_width_nm,
            "plunger_head_max_width_nm": self.plunger_head_max_width_nm,
            "plunger_head_height_nm": self.plunger_head_height_nm,
            "plunger_upper_taper_height_nm": self.plunger_upper_taper_height_nm,
            "plunger_lower_taper_height_nm": self.plunger_lower_taper_height_nm,
            "ohmic_to_barrier_gap_nm": self.ohmic_to_barrier_gap_nm,
            "barrier_to_plunger_gap_nm": self.barrier_to_plunger_gap_nm,
        }
