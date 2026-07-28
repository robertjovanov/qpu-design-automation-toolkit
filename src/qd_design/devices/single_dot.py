from dataclasses import dataclass, field
from typing import Dict, Any
import json

from phidl import Device

from ..core import (
    GateType,
    BARRIER,
    OHMIC,
    PLUNGER,
    ComponentMetadata,
)
from ..gates import CANONICAL_LOLLIPOP_PLUNGER, BarrierGate, OhmicContact, PlungerGate
from ..export import PlacedElement, placed_element_from_component


@dataclass
class SingleDotDevice:
    """
    Reusable builder for the single quantum dot top-view backbone.
    """

    name: str = "single_dot_device"

    device_y_size_nm: float = 200.0

    ohmic_width_nm: float = 40.0
    ohmic_length_nm: float = 200.0

    barrier_width_nm: float = 40.0
    barrier_length_nm: float = 140.0

    plunger_body_width_nm: float = CANONICAL_LOLLIPOP_PLUNGER.body_width_nm
    plunger_body_length_nm: float = CANONICAL_LOLLIPOP_PLUNGER.body_length_nm
    plunger_head_top_width_nm: float = CANONICAL_LOLLIPOP_PLUNGER.head_top_width_nm
    plunger_head_max_width_nm: float = CANONICAL_LOLLIPOP_PLUNGER.head_max_width_nm
    plunger_head_height_nm: float = CANONICAL_LOLLIPOP_PLUNGER.head_height_nm
    plunger_upper_taper_height_nm: float = (
        CANONICAL_LOLLIPOP_PLUNGER.upper_taper_height_nm
    )
    plunger_lower_taper_height_nm: float = (
        CANONICAL_LOLLIPOP_PLUNGER.lower_taper_height_nm
    )

    ohmic_to_barrier_gap_nm: float = 20.0
    barrier_to_plunger_gap_nm: float = 20.0

    device: Device = field(init=False, repr=False)
    refs: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    elements: Dict[str, PlacedElement] = field(init=False, default_factory=dict, repr=False)
    _built: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.device = Device(self.name)

    def build(self) -> Device:
        if self._built:
            return self.device

        if self.device_y_size_nm <= 0:
            raise ValueError("device_y_size_nm must be positive.")
        if self.ohmic_width_nm <= 0 or self.ohmic_length_nm <= 0:
            raise ValueError("Ohmic dimensions must be positive.")
        if self.barrier_width_nm <= 0 or self.barrier_length_nm <= 0:
            raise ValueError("Barrier dimensions must be positive.")
        if self.ohmic_to_barrier_gap_nm < 0 or self.barrier_to_plunger_gap_nm < 0:
            raise ValueError("Horizontal gaps must be non-negative.")

        left_ohmic = OhmicContact(
            metadata=ComponentMetadata(
                name="OC_L",
                gate_type=GateType.OHMIC,
                layer=OHMIC,
                voltage_label="V_OC_L",
            ),
            width_nm=self.ohmic_width_nm,
            length_nm=self.ohmic_length_nm,
        )

        right_ohmic = OhmicContact(
            metadata=ComponentMetadata(
                name="OC_R",
                gate_type=GateType.OHMIC,
                layer=OHMIC,
                voltage_label="V_OC_R",
            ),
            width_nm=self.ohmic_width_nm,
            length_nm=self.ohmic_length_nm,
        )

        left_barrier = BarrierGate(
            metadata=ComponentMetadata(
                name="B_L",
                gate_type=GateType.BARRIER,
                layer=BARRIER,
                voltage_label="V_B_L",
            ),
            width_nm=self.barrier_width_nm,
            length_nm=self.barrier_length_nm,
        )

        right_barrier = BarrierGate(
            metadata=ComponentMetadata(
                name="B_R",
                gate_type=GateType.BARRIER,
                layer=BARRIER,
                voltage_label="V_B_R",
            ),
            width_nm=self.barrier_width_nm,
            length_nm=self.barrier_length_nm,
        )

        plunger = PlungerGate(
            metadata=ComponentMetadata(
                name="P1",
                gate_type=GateType.PLUNGER,
                layer=PLUNGER,
                voltage_label="V_P1",
            ),
            body_width_nm=self.plunger_body_width_nm,
            body_length_nm=self.plunger_body_length_nm,
            head_top_width_nm=self.plunger_head_top_width_nm,
            head_max_width_nm=self.plunger_head_max_width_nm,
            head_height_nm=self.plunger_head_height_nm,
            upper_taper_height_nm=self.plunger_upper_taper_height_nm,
            lower_taper_height_nm=self.plunger_lower_taper_height_nm,
        )

        ref_oc_l = left_ohmic.add_to(self.device)
        ref_oc_r = right_ohmic.add_to(self.device)
        ref_b_l = left_barrier.add_to(self.device)
        ref_b_r = right_barrier.add_to(self.device)
        ref_p = plunger.add_to(self.device)

        plunger_half_max_width = self.plunger_head_max_width_nm / 2.0

        x_bl_right = -plunger_half_max_width - self.barrier_to_plunger_gap_nm
        x_bl_left = x_bl_right - self.barrier_width_nm

        x_br_left = plunger_half_max_width + self.barrier_to_plunger_gap_nm
        x_br_right = x_br_left + self.barrier_width_nm

        x_ocl_left = x_bl_left - self.ohmic_to_barrier_gap_nm - self.ohmic_width_nm
        x_ocr_left = x_br_right + self.ohmic_to_barrier_gap_nm

        ref_oc_l.move(
            origin=(ref_oc_l.xmin, ref_oc_l.ymin),
            destination=(x_ocl_left, 0.0),
        )
        ref_oc_r.move(
            origin=(ref_oc_r.xmin, ref_oc_r.ymin),
            destination=(x_ocr_left, 0.0),
        )

        ref_b_l.move(
            origin=(ref_b_l.xmin, ref_b_l.ymin),
            destination=(x_bl_left, 0.0),
        )
        ref_b_r.move(
            origin=(ref_b_r.xmin, ref_b_r.ymin),
            destination=(x_br_left, 0.0),
        )

        ref_p.rotate(180, center=(0, 0))
        ref_p.move(
            origin=(0.0, ref_p.ymax),
            destination=(0.0, self.device_y_size_nm),
        )

        self.refs = {
            "left_ohmic": ref_oc_l,
            "left_barrier": ref_b_l,
            "plunger": ref_p,
            "right_barrier": ref_b_r,
            "right_ohmic": ref_oc_r,
        }

        self.elements = {
            "OC_L": placed_element_from_component(left_ohmic, ref_oc_l),
            "B_L": placed_element_from_component(left_barrier, ref_b_l),
            "P1": placed_element_from_component(plunger, ref_p),
            "B_R": placed_element_from_component(right_barrier, ref_b_r),
            "OC_R": placed_element_from_component(right_ohmic, ref_oc_r),
        }

        self._built = True
        return self.device

    def ensure_built(self) -> Device:
        return self.build()

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device_y_size_nm": self.device_y_size_nm,
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
