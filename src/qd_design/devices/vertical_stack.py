from dataclasses import dataclass, field
from typing import Any, Dict, Optional
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
class VerticallyStackedDoubleDotDevice:
    """
    Reusable builder for two single-dot gate modules stacked along y.

    The bottom module is built with its plunger head facing upward. The top
    module is the 180-degree-facing partner, with its plunger head facing
    downward. Each module keeps its own left/right ohmic and barrier gates.
    """

    name: str = "vertically_stacked_double_dot"

    device_y_size_nm: float = 200.0
    module_gap_nm: float = 0.0

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

    @property
    def module_pitch_y_nm(self) -> float:
        return self.device_y_size_nm + self.module_gap_nm

    @property
    def plunger_total_height_nm(self) -> float:
        return self.plunger_body_length_nm + self.plunger_head_height_nm

    @property
    def facing_plunger_gap_nm(self) -> float:
        return (
            2.0 * (self.device_y_size_nm - self.plunger_total_height_nm)
            + self.module_gap_nm
        )

    def _validate(self) -> None:
        if self.device_y_size_nm <= 0:
            raise ValueError("device_y_size_nm must be positive.")
        if self.module_gap_nm < 0:
            raise ValueError("module_gap_nm must be non-negative.")
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

    def _module_x_positions(self) -> Dict[str, float]:
        plunger_half_max_width = self.plunger_head_max_width_nm / 2.0

        x_bl_right = -plunger_half_max_width - self.barrier_to_plunger_gap_nm
        x_bl_left = x_bl_right - self.barrier_width_nm

        x_br_left = plunger_half_max_width + self.barrier_to_plunger_gap_nm
        x_br_right = x_br_left + self.barrier_width_nm

        return {
            "left_barrier": x_bl_left,
            "right_barrier": x_br_left,
            "left_ohmic": x_bl_left - self.ohmic_to_barrier_gap_nm - self.ohmic_width_nm,
            "right_ohmic": x_br_right + self.ohmic_to_barrier_gap_nm,
        }

    def _place_single_dot_module(
        self,
        *,
        module_index: int,
        y_min_nm: float,
        plunger_faces: str,
    ) -> None:
        if plunger_faces not in {"up", "down"}:
            raise ValueError("plunger_faces must be either 'up' or 'down'.")

        y_max_nm = y_min_nm + self.device_y_size_nm
        positions = self._module_x_positions()

        left_ohmic = self._make_ohmic(
            name=f"OC_L_{module_index}",
            voltage_label=f"V_OC_L_{module_index}",
        )
        right_ohmic = self._make_ohmic(
            name=f"OC_R_{module_index}",
            voltage_label=f"V_OC_R_{module_index}",
        )
        left_barrier = self._make_barrier(
            name=f"B_L_{module_index}",
            voltage_label=f"V_B_L_{module_index}",
        )
        right_barrier = self._make_barrier(
            name=f"B_R_{module_index}",
            voltage_label=f"V_B_R_{module_index}",
        )
        plunger = self._make_plunger(
            name=f"P{module_index}",
            voltage_label=f"V_P{module_index}",
        )

        ref_oc_l = left_ohmic.add_to(self.device)
        ref_oc_r = right_ohmic.add_to(self.device)
        ref_b_l = left_barrier.add_to(self.device)
        ref_b_r = right_barrier.add_to(self.device)
        ref_p = plunger.add_to(self.device)

        ref_oc_l.move(
            origin=(ref_oc_l.xmin, ref_oc_l.ymin),
            destination=(positions["left_ohmic"], y_min_nm),
        )
        ref_oc_r.move(
            origin=(ref_oc_r.xmin, ref_oc_r.ymin),
            destination=(positions["right_ohmic"], y_min_nm),
        )

        if plunger_faces == "up":
            barrier_y_min = y_min_nm
            ref_p.move(
                origin=(0.0, ref_p.ymin),
                destination=(0.0, y_min_nm),
            )
        else:
            barrier_y_min = y_max_nm - self.barrier_length_nm
            ref_p.rotate(180, center=(0, 0))
            ref_p.move(
                origin=(0.0, ref_p.ymax),
                destination=(0.0, y_max_nm),
            )

        ref_b_l.move(
            origin=(ref_b_l.xmin, ref_b_l.ymin),
            destination=(positions["left_barrier"], barrier_y_min),
        )
        ref_b_r.move(
            origin=(ref_b_r.xmin, ref_b_r.ymin),
            destination=(positions["right_barrier"], barrier_y_min),
        )

        key_prefix = f"dot_{module_index}"
        self.refs.update(
            {
                f"{key_prefix}_left_ohmic": ref_oc_l,
                f"{key_prefix}_left_barrier": ref_b_l,
                f"{key_prefix}_plunger": ref_p,
                f"{key_prefix}_right_barrier": ref_b_r,
                f"{key_prefix}_right_ohmic": ref_oc_r,
            }
        )

        self.elements[left_ohmic.name] = placed_element_from_component(left_ohmic, ref_oc_l)
        self.elements[left_barrier.name] = placed_element_from_component(left_barrier, ref_b_l)
        self.elements[plunger.name] = placed_element_from_component(plunger, ref_p)
        self.elements[right_barrier.name] = placed_element_from_component(right_barrier, ref_b_r)
        self.elements[right_ohmic.name] = placed_element_from_component(right_ohmic, ref_oc_r)

    def build(self) -> Device:
        if self._built:
            return self.device

        self._validate()

        self._place_single_dot_module(
            module_index=1,
            y_min_nm=0.0,
            plunger_faces="up",
        )
        self._place_single_dot_module(
            module_index=2,
            y_min_nm=self.module_pitch_y_nm,
            plunger_faces="down",
        )

        self.refs.update(
            {
                "module_pitch_y_nm": self.module_pitch_y_nm,
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
            "device_y_size_nm": self.device_y_size_nm,
            "module_gap_nm": self.module_gap_nm,
            "module_pitch_y_nm": self.module_pitch_y_nm,
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
class SeparatedVerticallyStackedDoubleDotDevice(VerticallyStackedDoubleDotDevice):
    """
    Vertical face-to-face double dot with separated ohmics and a central barrier.

    Compared with ``VerticallyStackedDoubleDotDevice``, the ohmics can be
    shortened to the plunger height and the central separator barrier is a
    horizontal barrier gate placed in the gap between the two facing plungers.
    """

    name: str = "separated_vertically_stacked_double_dot"
    ohmic_length_nm: Optional[float] = None

    separator_barrier_name: str = "B_SEP"
    separator_barrier_voltage_label: str = "V_B_SEP"
    separator_start_x_nm: Optional[float] = None
    separator_end_x_nm: Optional[float] = None

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
        return self.module_pitch_y_nm + self.device_y_size_nm - self.plunger_total_height_nm

    @property
    def separator_center_y_nm(self) -> float:
        return 0.5 * (self.bottom_plunger_top_y_nm + self.top_plunger_bottom_y_nm)

    @property
    def separator_y_min_nm(self) -> float:
        return self.separator_center_y_nm - 0.5 * self.barrier_width_nm

    @property
    def resolved_separator_start_x_nm(self) -> float:
        if self.separator_start_x_nm is not None:
            return float(self.separator_start_x_nm)
        return self._module_x_positions()["left_ohmic"]

    @property
    def resolved_separator_end_x_nm(self) -> float:
        if self.separator_end_x_nm is not None:
            return float(self.separator_end_x_nm)
        return 0.5 * self.plunger_head_max_width_nm

    @property
    def separator_length_nm(self) -> float:
        return self.resolved_separator_end_x_nm - self.resolved_separator_start_x_nm

    def _validate(self) -> None:
        if self.device_y_size_nm <= 0:
            raise ValueError("device_y_size_nm must be positive.")
        if self.module_gap_nm < 0:
            raise ValueError("module_gap_nm must be non-negative.")
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
            raise ValueError("The facing plunger gap must be positive for a separator barrier.")
        if self.barrier_width_nm > self.facing_plunger_gap_nm:
            raise ValueError(
                "The separator barrier width must fit in the gap between facing plungers."
            )
        if self.separator_length_nm <= 0:
            raise ValueError("The separator barrier end x must be greater than its start x.")

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

    def _make_separator_barrier(self) -> BarrierGate:
        return BarrierGate(
            metadata=ComponentMetadata(
                name=self.separator_barrier_name,
                gate_type=GateType.BARRIER,
                layer=BARRIER,
                voltage_label=self.separator_barrier_voltage_label,
            ),
            width_nm=self.separator_length_nm,
            length_nm=self.barrier_width_nm,
        )

    def _place_single_dot_module(
        self,
        *,
        module_index: int,
        y_min_nm: float,
        plunger_faces: str,
    ) -> None:
        if plunger_faces not in {"up", "down"}:
            raise ValueError("plunger_faces must be either 'up' or 'down'.")

        y_max_nm = y_min_nm + self.device_y_size_nm
        positions = self._module_x_positions()

        left_ohmic = self._make_ohmic(
            name=f"OC_L_{module_index}",
            voltage_label=f"V_OC_L_{module_index}",
        )
        right_ohmic = self._make_ohmic(
            name=f"OC_R_{module_index}",
            voltage_label=f"V_OC_R_{module_index}",
        )
        left_barrier = self._make_barrier(
            name=f"B_L_{module_index}",
            voltage_label=f"V_B_L_{module_index}",
        )
        right_barrier = self._make_barrier(
            name=f"B_R_{module_index}",
            voltage_label=f"V_B_R_{module_index}",
        )
        plunger = self._make_plunger(
            name=f"P{module_index}",
            voltage_label=f"V_P{module_index}",
        )

        ref_oc_l = left_ohmic.add_to(self.device)
        ref_oc_r = right_ohmic.add_to(self.device)
        ref_b_l = left_barrier.add_to(self.device)
        ref_b_r = right_barrier.add_to(self.device)
        ref_p = plunger.add_to(self.device)

        if plunger_faces == "up":
            ohmic_y_min = y_min_nm
            barrier_y_min = y_min_nm
            ref_p.move(
                origin=(0.0, ref_p.ymin),
                destination=(0.0, y_min_nm),
            )
        else:
            ohmic_y_min = y_max_nm - self.effective_ohmic_length_nm
            barrier_y_min = y_max_nm - self.barrier_length_nm
            ref_p.rotate(180, center=(0, 0))
            ref_p.move(
                origin=(0.0, ref_p.ymax),
                destination=(0.0, y_max_nm),
            )

        ref_oc_l.move(
            origin=(ref_oc_l.xmin, ref_oc_l.ymin),
            destination=(positions["left_ohmic"], ohmic_y_min),
        )
        ref_oc_r.move(
            origin=(ref_oc_r.xmin, ref_oc_r.ymin),
            destination=(positions["right_ohmic"], ohmic_y_min),
        )
        ref_b_l.move(
            origin=(ref_b_l.xmin, ref_b_l.ymin),
            destination=(positions["left_barrier"], barrier_y_min),
        )
        ref_b_r.move(
            origin=(ref_b_r.xmin, ref_b_r.ymin),
            destination=(positions["right_barrier"], barrier_y_min),
        )

        key_prefix = f"dot_{module_index}"
        self.refs.update(
            {
                f"{key_prefix}_left_ohmic": ref_oc_l,
                f"{key_prefix}_left_barrier": ref_b_l,
                f"{key_prefix}_plunger": ref_p,
                f"{key_prefix}_right_barrier": ref_b_r,
                f"{key_prefix}_right_ohmic": ref_oc_r,
            }
        )

        self.elements[left_ohmic.name] = placed_element_from_component(left_ohmic, ref_oc_l)
        self.elements[left_barrier.name] = placed_element_from_component(left_barrier, ref_b_l)
        self.elements[plunger.name] = placed_element_from_component(plunger, ref_p)
        self.elements[right_barrier.name] = placed_element_from_component(right_barrier, ref_b_r)
        self.elements[right_ohmic.name] = placed_element_from_component(right_ohmic, ref_oc_r)

    def _place_separator_barrier(self) -> None:
        separator = self._make_separator_barrier()
        ref_sep = separator.add_to(self.device)
        ref_sep.move(
            origin=(ref_sep.xmin, ref_sep.ymin),
            destination=(self.resolved_separator_start_x_nm, self.separator_y_min_nm),
        )

        self.refs["separator_barrier"] = ref_sep
        self.elements[separator.name] = placed_element_from_component(separator, ref_sep)

    def build(self) -> Device:
        if self._built:
            return self.device

        self._validate()

        self._place_single_dot_module(
            module_index=1,
            y_min_nm=0.0,
            plunger_faces="up",
        )
        self._place_single_dot_module(
            module_index=2,
            y_min_nm=self.module_pitch_y_nm,
            plunger_faces="down",
        )
        self._place_separator_barrier()

        self.refs.update(
            {
                "module_pitch_y_nm": self.module_pitch_y_nm,
                "facing_plunger_gap_nm": self.facing_plunger_gap_nm,
                "effective_ohmic_length_nm": self.effective_ohmic_length_nm,
                "separator_length_nm": self.separator_length_nm,
                "separator_center_y_nm": self.separator_center_y_nm,
            }
        )

        self._built = True
        return self.device

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device_y_size_nm": self.device_y_size_nm,
            "module_gap_nm": self.module_gap_nm,
            "module_pitch_y_nm": self.module_pitch_y_nm,
            "facing_plunger_gap_nm": self.facing_plunger_gap_nm,
            "ohmic_width_nm": self.ohmic_width_nm,
            "ohmic_length_nm": self.effective_ohmic_length_nm,
            "barrier_width_nm": self.barrier_width_nm,
            "barrier_length_nm": self.barrier_length_nm,
            "separator_barrier_name": self.separator_barrier_name,
            "separator_barrier_voltage_label": self.separator_barrier_voltage_label,
            "separator_start_x_nm": self.resolved_separator_start_x_nm,
            "separator_end_x_nm": self.resolved_separator_end_x_nm,
            "separator_length_nm": self.separator_length_nm,
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
