from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    """Definition of a GDS layer/datatype pair."""

    name: str
    gds_layer: int
    gds_datatype: int = 0

    @property
    def phidl_layer(self) -> tuple[int, int]:
        """PHIDL-compatible layer representation."""
        return (self.gds_layer, self.gds_datatype)


PLUNGER = LayerSpec(name="plunger", gds_layer=1, gds_datatype=0)
BARRIER = LayerSpec(name="barrier", gds_layer=2, gds_datatype=0)
OHMIC = LayerSpec(name="ohmic", gds_layer=3, gds_datatype=0)
SENSOR = LayerSpec(name="sensor", gds_layer=4, gds_datatype=0)
SCREENING = LayerSpec(name="screening", gds_layer=5, gds_datatype=0)
ANNOTATION = LayerSpec(name="annotation", gds_layer=10, gds_datatype=0)