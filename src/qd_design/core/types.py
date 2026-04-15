from enum import Enum


class GateType(str, Enum):
    """Semantic type of a gate/layout component."""

    BARRIER = "barrier"
    OHMIC = "ohmic"
    PLUNGER = "plunger"
    SENSOR = "sensor"
    SCREENING = "screening"
    GENERIC = "generic"