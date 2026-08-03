from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class MaterialLayer:
    """
    Continuous vertical layer spanning the full simulation domain in x and y.

    Convention:
    - z is in nm
    - z = 0 nm is the top of the Ge QW
    - z < 0 is below the top of the Ge QW
    - z > 0 is above the top of the Ge QW
    """

    name: str
    material: str
    z_min_nm: float
    z_max_nm: float
    alloy_x: Optional[float] = None
    notes: Optional[str] = None
    contact_name: Optional[str] = None

    def thickness_nm(self) -> float:
        return self.z_max_nm - self.z_min_nm

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateExtrusionRule:
    """
    Rule for extruding a 2D gate polygon into a 3D patterned override region.

    This is intended for regions like:
    - barrier gates
    - plunger gates
    - sensor gates
    - ohmic contacts

    The polygon in x-y comes from the 2D layout.
    The z extent comes from this rule.
    """

    name: str
    applies_to_layer_names: List[str]
    material: str
    z_min_nm: float
    z_max_nm: float
    default_voltage_label: Optional[str] = None
    notes: Optional[str] = None

    def applies_to(self, layer_name: str) -> bool:
        return layer_name in self.applies_to_layer_names

    def thickness_nm(self) -> float:
        return self.z_max_nm - self.z_min_nm

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessStack:
    """
    Full vertical process description for a device.

    Contains:
    - continuous material layers spanning the full x-y simulation domain
    - patterned region rules that override only local x-y polygons

    Recommended convention:
    - z_reference_name = 'top_of_Ge_QW'
    - top of Ge QW is fixed at z = 0 nm
    """

    name: str
    z_reference_name: str = "top_of_Ge_QW"

    material_layers: List[MaterialLayer] = field(default_factory=list)
    gate_rules: List[GateExtrusionRule] = field(default_factory=list)

    def validate(self) -> None:
        for layer in self.material_layers:
            if layer.z_max_nm <= layer.z_min_nm:
                raise ValueError(
                    f"Invalid material layer '{layer.name}': z_max_nm must be > z_min_nm."
                )

        for rule in self.gate_rules:
            if rule.z_max_nm <= rule.z_min_nm:
                raise ValueError(
                    f"Invalid gate rule '{rule.name}': z_max_nm must be > z_min_nm."
                )

        seen_layer_names: set[str] = set()
        for rule in self.gate_rules:
            for layer_name in rule.applies_to_layer_names:
                if layer_name in seen_layer_names:
                    raise ValueError(
                        f"Multiple gate rules apply to layer_name='{layer_name}'."
                    )
                seen_layer_names.add(layer_name)

    def gate_rule_for_layer_name(self, layer_name: str) -> GateExtrusionRule:
        matches = [rule for rule in self.gate_rules if rule.applies_to(layer_name)]
        if not matches:
            raise KeyError(f"No gate extrusion rule found for layer_name='{layer_name}'.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple gate extrusion rules match layer_name='{layer_name}'."
            )
        return matches[0]

    def top_z_nm(self) -> float:
        """
        Return the top z extent of the whole stack/rules.
        """
        candidates = [layer.z_max_nm for layer in self.material_layers] + [
            rule.z_max_nm for rule in self.gate_rules
        ]
        if not candidates:
            raise ValueError("ProcessStack is empty.")
        return max(candidates)

    def bottom_z_nm(self) -> float:
        """
        Return the bottom z extent of the whole stack/rules.
        """
        candidates = [layer.z_min_nm for layer in self.material_layers] + [
            rule.z_min_nm for rule in self.gate_rules
        ]
        if not candidates:
            raise ValueError("ProcessStack is empty.")
        return min(candidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "z_reference_name": self.z_reference_name,
            "z_min_nm": self.bottom_z_nm(),
            "z_max_nm": self.top_z_nm(),
            "material_layers": [layer.to_dict() for layer in self.material_layers],
            "gate_rules": [rule.to_dict() for rule in self.gate_rules],
        }


def make_sige_ge_process_stack(
    *,
    name: str = "SiGe_Ge_QW_two_metal_gate_layers",
    z_reference_name: str = "top_of_Ge_QW",
    sige_buffer_thickness_nm: float = 4000.0,
    body_contact_thickness_nm: float = 100.0,
    ge_qw_thickness_nm: float = 15.0,
    sige_cap_thickness_nm: float = 101.0,
    al2o3_total_thickness_nm: float = 72.0,
    sige_alloy_x: float = 0.15,
    barrier_gate_thickness_nm: float = 30.0,
    plunger_gate_thickness_nm: float = 30.0,
    screening_gate_thickness_nm: Optional[float] = None,
    barrier_gate_bottom_z_nm: float = 108.0,
    plunger_gate_bottom_z_nm: float = 143.0,
    screening_gate_bottom_z_nm: Optional[float] = None,
    ohmic_penetration_from_semiconductor_surface_nm: Optional[float] = None,
    ohmic_depth_from_device_top_nm: Optional[float] = None,
    barrier_material: str = "Al",
    plunger_material: str = "Al",
    screening_material: str = "Al",
    ohmic_material: str = "Al",
    dielectric_material: str = "Al2O3",
    cap_material: str = "SiGe",
    qw_material: str = "Ge",
    buffer_material: str = "SiGe",
) -> ProcessStack:
    """
    Build a parameterized process stack with the convention:

    - z = 0 nm is the top of the Ge QW
    - the top of the full device is determined by the top of the plunger gate layer
      unless some other rule/layer extends higher

    Default physical picture:
    - full-domain SiGe body contact below the SiGe buffer
    - SiGe buffer below the Ge QW
    - Ge QW
    - SiGe cap above the QW
    - one continuous Al2O3 dielectric layer above the cap
    - barrier and plunger gates are local patterned overrides inside the dielectric
    - an optional screening gate layer can be inserted as another patterned
      dielectric-embedded metal layer
    - ohmics are local patterned overrides spanning from the top of the device
      down to a penetration depth measured from the semiconductor surface

    Notes
    -----
    The dielectric is modeled as one continuous background layer.
    Gates and ohmics override local polygonal regions inside that background.

    With the default values:
    - SiGe body contact: -4115 .. -4015
    - SiGe buffer:       -4015 ..   -15
    - Ge QW:               -15   ..   0
    - SiGe cap:              0   .. 101
    - Al2O3 dielectric:    101   .. 173
    - barrier gates:       108   .. 138
    - plunger gates:       143   .. 173
    - ohmics:             -149   .. 173

    ``ohmic_penetration_from_semiconductor_surface_nm`` is measured down from
    the top of the SiGe cap and defaults to 250 nm.  The legacy
    ``ohmic_depth_from_device_top_nm`` parameter remains available with its
    original device-top-relative meaning.  Supplying both references is
    ambiguous and therefore rejected.

    If ``screening_gate_bottom_z_nm`` is provided, a screening gate rule is
    added. If its thickness is omitted, the plunger gate thickness is reused.
    """
    if sige_buffer_thickness_nm <= 0:
        raise ValueError("sige_buffer_thickness_nm must be positive.")
    if body_contact_thickness_nm <= 0:
        raise ValueError("body_contact_thickness_nm must be positive.")
    if ge_qw_thickness_nm <= 0:
        raise ValueError("ge_qw_thickness_nm must be positive.")
    if sige_cap_thickness_nm <= 0:
        raise ValueError("sige_cap_thickness_nm must be positive.")
    if al2o3_total_thickness_nm <= 0:
        raise ValueError("al2o3_total_thickness_nm must be positive.")
    if barrier_gate_thickness_nm <= 0:
        raise ValueError("barrier_gate_thickness_nm must be positive.")
    if plunger_gate_thickness_nm <= 0:
        raise ValueError("plunger_gate_thickness_nm must be positive.")
    if screening_gate_thickness_nm is not None and screening_gate_thickness_nm <= 0:
        raise ValueError("screening_gate_thickness_nm must be positive.")
    if screening_gate_thickness_nm is not None and screening_gate_bottom_z_nm is None:
        raise ValueError(
            "screening_gate_bottom_z_nm must be provided when "
            "screening_gate_thickness_nm is provided."
        )
    if screening_gate_bottom_z_nm is not None and screening_gate_thickness_nm is None:
        screening_gate_thickness_nm = plunger_gate_thickness_nm
    if (
        ohmic_penetration_from_semiconductor_surface_nm is not None
        and ohmic_depth_from_device_top_nm is not None
    ):
        raise ValueError(
            "Specify only one of "
            "ohmic_penetration_from_semiconductor_surface_nm and "
            "ohmic_depth_from_device_top_nm."
        )
    if (
        ohmic_penetration_from_semiconductor_surface_nm is not None
        and ohmic_penetration_from_semiconductor_surface_nm <= 0
    ):
        raise ValueError(
            "ohmic_penetration_from_semiconductor_surface_nm must be positive."
        )
    if (
        ohmic_depth_from_device_top_nm is not None
        and ohmic_depth_from_device_top_nm <= 0
    ):
        raise ValueError("ohmic_depth_from_device_top_nm must be positive.")
    if sige_alloy_x < 0 or sige_alloy_x > 1:
        raise ValueError("sige_alloy_x must be between 0 and 1.")

    # -------------------------
    # Continuous layers
    # -------------------------
    ge_qw_z_min = -ge_qw_thickness_nm
    ge_qw_z_max = 0.0

    sige_cap_z_min = ge_qw_z_max
    sige_cap_z_max = sige_cap_z_min + sige_cap_thickness_nm

    dielectric_z_min = sige_cap_z_max
    dielectric_z_max = dielectric_z_min + al2o3_total_thickness_nm

    sige_buffer_z_max = ge_qw_z_min
    sige_buffer_z_min = sige_buffer_z_max - sige_buffer_thickness_nm
    body_contact_z_max = sige_buffer_z_min
    body_contact_z_min = body_contact_z_max - body_contact_thickness_nm

    # -------------------------
    # Patterned gate layers
    # -------------------------
    barrier_z_min = barrier_gate_bottom_z_nm
    barrier_z_max = barrier_z_min + barrier_gate_thickness_nm

    screening_z_min = screening_gate_bottom_z_nm
    screening_z_max = (
        screening_z_min + screening_gate_thickness_nm
        if screening_z_min is not None and screening_gate_thickness_nm is not None
        else None
    )

    plunger_z_min = plunger_gate_bottom_z_nm
    plunger_z_max = plunger_z_min + plunger_gate_thickness_nm

    # The very top of the device is the highest z in the patterned/continuous stack.
    patterned_top_z = [barrier_z_max, plunger_z_max]
    if screening_z_max is not None:
        patterned_top_z.append(screening_z_max)
    device_top_z = max(dielectric_z_max, *patterned_top_z)

    # Ohmics always reach the device top.  New configurations express their
    # physical penetration from the named semiconductor surface (the top of
    # the SiGe cap); the legacy option retains its historical device-top
    # reference for existing callers.
    ohmic_z_max = device_top_z
    if ohmic_depth_from_device_top_nm is not None:
        ohmic_z_min = ohmic_z_max - ohmic_depth_from_device_top_nm
    else:
        penetration_nm = (
            250.0
            if ohmic_penetration_from_semiconductor_surface_nm is None
            else ohmic_penetration_from_semiconductor_surface_nm
        )
        semiconductor_surface_z = sige_cap_z_max
        ohmic_z_min = semiconductor_surface_z - penetration_nm

    if barrier_z_max > dielectric_z_max:
        raise ValueError(
            "Barrier gate extends above the dielectric background layer. "
            "Increase al2o3_total_thickness_nm or move the barrier gate lower."
        )

    if plunger_z_max > dielectric_z_max:
        raise ValueError(
            "Plunger gate extends above the dielectric background layer. "
            "Increase al2o3_total_thickness_nm or move the plunger gate lower."
        )

    if screening_z_min is not None and screening_z_max is not None:
        if screening_z_max > dielectric_z_max:
            raise ValueError(
                "Screening gate extends above the dielectric background layer. "
                "Increase al2o3_total_thickness_nm or move the screening gate lower."
            )
        if screening_z_min < dielectric_z_min:
            raise ValueError(
                "Screening gate extends below the dielectric background layer. "
                "Move the screening gate higher or lower the dielectric bottom."
            )

    gate_rules = [
        GateExtrusionRule(
            name="barrier_gate_rule",
            applies_to_layer_names=["barrier"],
            material=barrier_material,
            z_min_nm=barrier_z_min,
            z_max_nm=barrier_z_max,
        ),
    ]

    if screening_z_min is not None and screening_z_max is not None:
        gate_rules.append(
            GateExtrusionRule(
                name="screening_gate_rule",
                applies_to_layer_names=["screening"],
                material=screening_material,
                z_min_nm=screening_z_min,
                z_max_nm=screening_z_max,
            )
        )

    gate_rules.extend(
        [
            GateExtrusionRule(
                name="plunger_gate_rule",
                applies_to_layer_names=["plunger", "sensor"],
                material=plunger_material,
                z_min_nm=plunger_z_min,
                z_max_nm=plunger_z_max,
            ),
            GateExtrusionRule(
                name="ohmic_gate_rule",
                applies_to_layer_names=["ohmic"],
                material=ohmic_material,
                z_min_nm=ohmic_z_min,
                z_max_nm=ohmic_z_max,
            ),
        ]
    )

    stack = ProcessStack(
        name=name,
        z_reference_name=z_reference_name,
        material_layers=[
            MaterialLayer(
                name="SiGe_body_contact",
                material=buffer_material,
                z_min_nm=body_contact_z_min,
                z_max_nm=body_contact_z_max,
                alloy_x=sige_alloy_x,
                notes="Full-domain body contact below the SiGe buffer.",
                contact_name="Body",
            ),
            MaterialLayer(
                name="SiGe_buffer",
                material=buffer_material,
                z_min_nm=sige_buffer_z_min,
                z_max_nm=sige_buffer_z_max,
                alloy_x=sige_alloy_x,
            ),
            MaterialLayer(
                name="Ge_QW",
                material=qw_material,
                z_min_nm=ge_qw_z_min,
                z_max_nm=ge_qw_z_max,
            ),
            MaterialLayer(
                name="SiGe_cap",
                material=cap_material,
                z_min_nm=sige_cap_z_min,
                z_max_nm=sige_cap_z_max,
                alloy_x=sige_alloy_x,
            ),
            MaterialLayer(
                name="Al2O3_dielectric",
                material=dielectric_material,
                z_min_nm=dielectric_z_min,
                z_max_nm=dielectric_z_max,
            ),
        ],
        gate_rules=gate_rules,
    )

    stack.validate()
    return stack


def make_reference_sige_ge_process_stack() -> ProcessStack:
    """
    Convenience helper using the current agreed reference values.
    """
    return make_sige_ge_process_stack()
