from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from numbers import Integral
from typing import Any, Dict, Iterable, List, Tuple

from phidl import Device

from ..export import PlacedElement
from ..gates import CANONICAL_LOLLIPOP_PLUNGER
from .linear_array import TopBarrierLinearDotArrayDevice


@dataclass
class TwoDDotArrayDevice:
    """Reusable builder for a facing 2xN quantum-dot array.

    Two complete :class:`TopBarrierLinearDotArrayDevice` rows are built.  The
    lower row is rotated by 180 degrees, while the upper row is only
    translated.  Consequently, each row keeps its barriers and plungers on
    the same outer side and the two rows face one another.

    Element names are assigned after the row transforms, in physical
    left-to-right order.  Plungers and barriers use row-major numbering, while
    ohmics use ``OC_L_1``/``OC_R_1`` and ``OC_L_2``/``OC_R_2``.
    """

    name: str = "two_d_dot_array"
    n_columns: int = 2

    device_y_size_nm: float = 200.0
    row_gap_nm: float = 0.0

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

    include_screening_gates: bool = False
    screening_gate_width_nm: float | None = None
    screening_gate_length_nm: float | None = None

    device: Device = field(init=False, repr=False)
    row_builders: Dict[str, TopBarrierLinearDotArrayDevice] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    refs: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    elements: Dict[str, PlacedElement] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    row_element_names: Dict[str, List[str]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _built: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.device = Device(self.name)

    @property
    def n_rows(self) -> int:
        return 2

    @property
    def n_dots(self) -> int:
        return self.n_rows * self.n_columns

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
        if isinstance(self.n_columns, bool) or not isinstance(
            self.n_columns,
            Integral,
        ):
            raise ValueError("n_columns must be an integer >= 1.")
        if self.n_columns < 1:
            raise ValueError("n_columns must be >= 1.")
        if self.device_y_size_nm <= 0:
            raise ValueError("device_y_size_nm must be positive.")
        if self.row_gap_nm < 0:
            raise ValueError("row_gap_nm must be non-negative.")
        if self.plunger_total_height_nm > self.device_y_size_nm:
            raise ValueError(
                "The plunger body length plus head height must fit inside "
                "device_y_size_nm."
            )
        if self.barrier_length_nm > self.device_y_size_nm:
            raise ValueError("barrier_length_nm must fit inside device_y_size_nm.")
        if self.ohmic_length_nm > self.device_y_size_nm:
            raise ValueError("ohmic_length_nm must fit inside device_y_size_nm.")
        if (
            self.include_screening_gates
            and self.screening_gate_length_nm is not None
            and self.screening_gate_length_nm > self.device_y_size_nm
        ):
            raise ValueError(
                "screening_gate_length_nm must fit inside device_y_size_nm."
            )

    def _row_parameters(self) -> Dict[str, Any]:
        return {
            "n_dots": self.n_columns,
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
            "include_screening_gates": self.include_screening_gates,
            "screening_gate_width_nm": self.screening_gate_width_nm,
            "screening_gate_length_nm": self.screening_gate_length_nm,
        }

    @staticmethod
    def _transform_point(
        point: Tuple[float, float],
        *,
        rotation_degrees: int,
        translation_nm: Tuple[float, float],
    ) -> Tuple[float, float]:
        x, y = point
        dx, dy = translation_nm
        if rotation_degrees == 0:
            transformed = (x + dx, y + dy)
        elif rotation_degrees == 180:
            transformed = (-x + dx, -y + dy)
        else:
            raise ValueError("Only 0- and 180-degree row transforms are supported.")

        # PHIDL rotations can leave machine-scale trigonometric residue.  The
        # layout spec is the simulation contract, so keep its coordinates
        # stable while preserving far more precision than the nanometre inputs.
        return tuple(round(float(value), 12) for value in transformed)

    @classmethod
    def _transform_element(
        cls,
        element: PlacedElement,
        *,
        rotation_degrees: int,
        translation_nm: Tuple[float, float],
    ) -> PlacedElement:
        polygons = [
            [
                cls._transform_point(
                    point,
                    rotation_degrees=rotation_degrees,
                    translation_nm=translation_nm,
                )
                for point in polygon
            ]
            for polygon in element.polygon_xy_nm
        ]
        points = [point for polygon in polygons for point in polygon]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        return replace(
            element,
            x_min_nm=x_min,
            y_min_nm=y_min,
            x_max_nm=x_max,
            y_max_nm=y_max,
            center_x_nm=0.5 * (x_min + x_max),
            center_y_nm=0.5 * (y_min + y_max),
            width_nm=x_max - x_min,
            height_nm=y_max - y_min,
            params=dict(element.params),
            polygon_xy_nm=polygons,
        )

    def _name_row_elements(
        self,
        elements: Iterable[PlacedElement],
        *,
        row_index: int,
    ) -> List[PlacedElement]:
        ordered = sorted(elements, key=lambda element: element.center_x_nm)
        type_indices = {
            "barrier": (row_index - 1) * (self.n_columns + 1) + 1,
            "plunger": (row_index - 1) * self.n_columns + 1,
            "screening": (row_index - 1) * self.n_columns + 1,
        }
        ohmic_index = 0
        named: List[PlacedElement] = []

        for element in ordered:
            if element.gate_type == "ohmic":
                ohmic_index += 1
                side = "L" if ohmic_index == 1 else "R"
                name = f"OC_{side}_{row_index}"
            elif element.gate_type == "barrier":
                name = f"B{type_indices['barrier']}"
                type_indices["barrier"] += 1
            elif element.gate_type == "plunger":
                name = f"P{type_indices['plunger']}"
                type_indices["plunger"] += 1
            elif element.gate_type == "screening":
                screening_index = type_indices["screening"]
                name = f"SG{screening_index}"
                type_indices["screening"] += 1
            else:
                raise ValueError(f"Unsupported row element type: {element.gate_type!r}.")

            updates = {"name": name, "voltage_label": f"V_{name}"}
            if element.gate_type == "screening":
                updates["notes"] = (
                    f"Screening gate aligned with the body of P{screening_index} "
                    f"in row {row_index}."
                )
            named.append(replace(element, **updates))

        return named

    def _add_row(
        self,
        *,
        row_name: str,
        row_index: int,
        rotation_degrees: int,
        y_min_nm: float,
    ) -> None:
        row_builder = TopBarrierLinearDotArrayDevice(
            name=f"{self.name}_{row_name}_source",
            **self._row_parameters(),
        )
        row_device = row_builder.ensure_built()
        row_ref = self.device.add_ref(row_device)

        if rotation_degrees == 180:
            row_ref.rotate(180, center=(0.0, 0.0))
            translation_nm = (0.0, y_min_nm + self.device_y_size_nm)
        else:
            translation_nm = (0.0, y_min_nm)
        row_ref.move(origin=(0.0, 0.0), destination=translation_nm)

        transformed = [
            self._transform_element(
                element,
                rotation_degrees=rotation_degrees,
                translation_nm=translation_nm,
            )
            for element in row_builder.layout_elements().values()
        ]
        named = self._name_row_elements(transformed, row_index=row_index)

        self.row_builders[row_name] = row_builder
        self.refs[f"{row_name}_row"] = row_ref
        self.row_element_names[row_name] = [element.name for element in named]
        self.elements.update({element.name: element for element in named})

    def build(self) -> Device:
        if self._built:
            return self.device

        self._validate()
        self._add_row(
            row_name="bottom",
            row_index=1,
            rotation_degrees=180,
            y_min_nm=0.0,
        )
        self._add_row(
            row_name="top",
            row_index=2,
            rotation_degrees=0,
            y_min_nm=self.row_pitch_y_nm,
        )

        self.refs.update(
            {
                "rotated_row": "bottom",
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
            "n_dots": self.n_dots,
            "row_builder": TopBarrierLinearDotArrayDevice.__name__,
            "rotated_row": "bottom",
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
            "include_screening_gates": self.include_screening_gates,
        }

    def layout_elements(self) -> Dict[str, PlacedElement]:
        self.ensure_built()
        return self.elements

    def layout_spec(self) -> list[dict[str, Any]]:
        self.ensure_built()
        return [element.to_dict() for element in self.elements.values()]

    def write_layout_spec_json(self, filepath: str) -> None:
        self.ensure_built()
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.layout_spec(), file, indent=2)

    def write_gds(self, filepath: str) -> None:
        self.ensure_built().write_gds(filepath)

    def write_svg(self, filepath: str) -> None:
        self.ensure_built().write_svg(filepath)
