from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json

from phidl import Device

from ..core import (
    GateType,
    BARRIER,
    OHMIC,
    PLUNGER,
    SCREENING,
    ComponentMetadata,
)
from ..gates import (
    CANONICAL_LOLLIPOP_PLUNGER,
    BarrierGate,
    OhmicContact,
    PlungerGate,
    ScreeningGate,
)
from ..export import PlacedElement, placed_element_from_component


@dataclass
class LinearDotArrayDevice:
    """
    Reusable builder for a 1xN linear quantum dot array backbone.

    Pattern:
        left ohmic + left barrier + N * (plunger + barrier) + right ohmic

    ``barrier_side`` controls whether the barriers approach the channel from
    the bottom (the historical default) or from the top, alongside the
    plungers.
    """

    name: str = "linear_dot_array"
    n_dots: int = 1

    device_y_size_nm: float = 200.0

    ohmic_width_nm: float = 40.0
    ohmic_length_nm: float = 200.0

    barrier_width_nm: float = 40.0
    barrier_length_nm: float = 140.0
    barrier_side: str = "bottom"

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

    include_screening_gates: bool = False
    screening_gate_width_nm: Optional[float] = None
    screening_gate_length_nm: Optional[float] = None

    device: Device = field(init=False, repr=False)
    refs: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    elements: Dict[str, PlacedElement] = field(init=False, default_factory=dict, repr=False)
    _built: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.device = Device(self.name)

    def _validate(self) -> None:
        if self.n_dots < 1:
            raise ValueError("n_dots must be >= 1.")
        if self.device_y_size_nm <= 0:
            raise ValueError("device_y_size_nm must be positive.")
        if self.ohmic_width_nm <= 0 or self.ohmic_length_nm <= 0:
            raise ValueError("Ohmic dimensions must be positive.")
        if self.barrier_width_nm <= 0 or self.barrier_length_nm <= 0:
            raise ValueError("Barrier dimensions must be positive.")
        if self.barrier_side not in {"bottom", "top"}:
            raise ValueError("barrier_side must be either 'bottom' or 'top'.")
        if self.ohmic_to_barrier_gap_nm < 0 or self.barrier_to_plunger_gap_nm < 0:
            raise ValueError("Horizontal gaps must be non-negative.")
        if self.plunger_body_width_nm <= 0 or self.plunger_body_length_nm < 0:
            raise ValueError(
                "Plunger body width must be positive and body length must be non-negative."
            )
        if self.include_screening_gates:
            if self.effective_screening_gate_width_nm <= 0:
                raise ValueError("Screening gate width must be positive.")
            if self.effective_screening_gate_length_nm <= 0:
                raise ValueError("Screening gate length must be positive.")

    @property
    def effective_screening_gate_width_nm(self) -> float:
        if self.screening_gate_width_nm is None:
            return self.plunger_body_width_nm
        return float(self.screening_gate_width_nm)

    @property
    def effective_screening_gate_length_nm(self) -> float:
        if self.screening_gate_length_nm is None:
            return self.plunger_body_length_nm
        return float(self.screening_gate_length_nm)

    def _make_plunger(self, index: int) -> PlungerGate:
        return PlungerGate(
            metadata=ComponentMetadata(
                name=f"P{index}",
                gate_type=GateType.PLUNGER,
                layer=PLUNGER,
                voltage_label=f"V_P{index}",
            ),
            body_width_nm=self.plunger_body_width_nm,
            body_length_nm=self.plunger_body_length_nm,
            head_top_width_nm=self.plunger_head_top_width_nm,
            head_max_width_nm=self.plunger_head_max_width_nm,
            head_height_nm=self.plunger_head_height_nm,
            upper_taper_height_nm=self.plunger_upper_taper_height_nm,
            lower_taper_height_nm=self.plunger_lower_taper_height_nm,
        )

    def _make_barrier(self, index: int) -> BarrierGate:
        return BarrierGate(
            metadata=ComponentMetadata(
                name=f"B{index}",
                gate_type=GateType.BARRIER,
                layer=BARRIER,
                voltage_label=f"V_B{index}",
            ),
            width_nm=self.barrier_width_nm,
            length_nm=self.barrier_length_nm,
        )

    def _make_screening_gate(self, index: int) -> ScreeningGate:
        return ScreeningGate(
            metadata=ComponentMetadata(
                name=f"SG{index}",
                gate_type=GateType.SCREENING,
                layer=SCREENING,
                voltage_label=f"V_SG{index}",
                role="plunger_body_screen",
                notes=f"Screening gate aligned below the body of P{index}.",
            ),
            width_nm=self.effective_screening_gate_width_nm,
            length_nm=self.effective_screening_gate_length_nm,
        )

    def _make_left_ohmic(self) -> OhmicContact:
        return OhmicContact(
            metadata=ComponentMetadata(
                name="OC_L",
                gate_type=GateType.OHMIC,
                layer=OHMIC,
                voltage_label="V_OC_L",
            ),
            width_nm=self.ohmic_width_nm,
            length_nm=self.ohmic_length_nm,
        )

    def _make_right_ohmic(self) -> OhmicContact:
        return OhmicContact(
            metadata=ComponentMetadata(
                name="OC_R",
                gate_type=GateType.OHMIC,
                layer=OHMIC,
                voltage_label="V_OC_R",
            ),
            width_nm=self.ohmic_width_nm,
            length_nm=self.ohmic_length_nm,
        )

    def build(self) -> Device:
        if self._built:
            return self.device

        self._validate()

        left_ohmic = self._make_left_ohmic()
        right_ohmic = self._make_right_ohmic()

        barriers = [self._make_barrier(i + 1) for i in range(self.n_dots + 1)]
        plungers = [self._make_plunger(i + 1) for i in range(self.n_dots)]
        screening_gates = (
            [self._make_screening_gate(i + 1) for i in range(self.n_dots)]
            if self.include_screening_gates
            else []
        )

        ref_oc_l = left_ohmic.add_to(self.device)
        ref_oc_r = right_ohmic.add_to(self.device)

        barrier_refs = [b.add_to(self.device) for b in barriers]
        plunger_refs = [p.add_to(self.device) for p in plungers]
        screening_refs = [s.add_to(self.device) for s in screening_gates]

        half_max = self.plunger_head_max_width_nm / 2.0
        plunger_pitch = (
            self.plunger_head_max_width_nm
            + self.barrier_width_nm
            + 2.0 * self.barrier_to_plunger_gap_nm
        )

        plunger_centers_x = [
            (i - 0.5 * (self.n_dots - 1)) * plunger_pitch
            for i in range(self.n_dots)
        ]

        barrier_x_lefts: List[float] = []

        x_b0_left = (
            plunger_centers_x[0]
            - half_max
            - self.barrier_to_plunger_gap_nm
            - self.barrier_width_nm
        )
        barrier_x_lefts.append(x_b0_left)

        for i in range(1, self.n_dots):
            x_bi_left = (
                plunger_centers_x[i - 1]
                + half_max
                + self.barrier_to_plunger_gap_nm
            )
            barrier_x_lefts.append(x_bi_left)

        x_blast_left = (
            plunger_centers_x[-1]
            + half_max
            + self.barrier_to_plunger_gap_nm
        )
        barrier_x_lefts.append(x_blast_left)

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
            destination=(x_ocl_left, 0.0),
        )
        ref_oc_r.move(
            origin=(ref_oc_r.xmin, ref_oc_r.ymin),
            destination=(x_ocr_left, 0.0),
        )

        barrier_y_min = (
            0.0
            if self.barrier_side == "bottom"
            else self.device_y_size_nm - self.barrier_length_nm
        )
        for ref_b, x_left in zip(barrier_refs, barrier_x_lefts):
            ref_b.move(
                origin=(ref_b.xmin, ref_b.ymin),
                destination=(x_left, barrier_y_min),
            )

        for ref_p, xc in zip(plunger_refs, plunger_centers_x):
            ref_p.rotate(180, center=(0, 0))
            ref_p.move(
                origin=(0.0, ref_p.ymax),
                destination=(xc, self.device_y_size_nm),
            )

        screening_width = self.effective_screening_gate_width_nm
        screening_length = self.effective_screening_gate_length_nm
        screening_y_min = self.device_y_size_nm - screening_length
        for ref_s, xc in zip(screening_refs, plunger_centers_x):
            ref_s.move(
                origin=(ref_s.xmin, ref_s.ymin),
                destination=(xc - 0.5 * screening_width, screening_y_min),
            )

        self.refs = {
            "left_ohmic": ref_oc_l,
            "right_ohmic": ref_oc_r,
            "barriers": barrier_refs,
            "plungers": plunger_refs,
            "screening_gates": screening_refs,
            "plunger_centers_x": plunger_centers_x,
            "barrier_x_lefts": barrier_x_lefts,
        }

        elements: Dict[str, PlacedElement] = {}
        elements["OC_L"] = placed_element_from_component(left_ohmic, ref_oc_l)

        for i in range(self.n_dots):
            barrier_name = f"B{i + 1}"
            plunger_name = f"P{i + 1}"
            elements[barrier_name] = placed_element_from_component(barriers[i], barrier_refs[i])
            elements[plunger_name] = placed_element_from_component(plungers[i], plunger_refs[i])
            if self.include_screening_gates:
                screening_name = f"SG{i + 1}"
                elements[screening_name] = placed_element_from_component(
                    screening_gates[i],
                    screening_refs[i],
                )

        last_barrier_name = f"B{self.n_dots + 1}"
        elements[last_barrier_name] = placed_element_from_component(
            barriers[-1], barrier_refs[-1]
        )
        elements["OC_R"] = placed_element_from_component(right_ohmic, ref_oc_r)

        self.elements = elements

        self._built = True
        return self.device

    def ensure_built(self) -> Device:
        return self.build()

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_dots": self.n_dots,
            "device_y_size_nm": self.device_y_size_nm,
            "ohmic_width_nm": self.ohmic_width_nm,
            "ohmic_length_nm": self.ohmic_length_nm,
            "barrier_width_nm": self.barrier_width_nm,
            "barrier_length_nm": self.barrier_length_nm,
            "barrier_side": self.barrier_side,
            "plunger_body_width_nm": self.plunger_body_width_nm,
            "plunger_body_length_nm": self.plunger_body_length_nm,
            "plunger_head_top_width_nm": self.plunger_head_top_width_nm,
            "plunger_head_max_width_nm": self.plunger_head_max_width_nm,
            "plunger_head_height_nm": self.plunger_head_height_nm,
            "plunger_upper_taper_height_nm": self.plunger_upper_taper_height_nm,
            "plunger_lower_taper_height_nm": self.plunger_lower_taper_height_nm,
            "ohmic_to_barrier_gap_nm": self.ohmic_to_barrier_gap_nm,
            "barrier_to_plunger_gap_nm": self.barrier_to_plunger_gap_nm,
            "include_screening_gates": self.include_screening_gates,
            "screening_gate_width_nm": self.effective_screening_gate_width_nm,
            "screening_gate_length_nm": self.effective_screening_gate_length_nm,
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
class TopBarrierLinearDotArrayDevice(LinearDotArrayDevice):
    """A 1xN dot array with barriers and plungers entering from the top."""

    barrier_side: str = "top"
