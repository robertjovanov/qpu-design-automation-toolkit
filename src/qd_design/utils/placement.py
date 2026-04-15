def _y_center(ref) -> float:
    return 0.5 * (ref.ymin + ref.ymax)


def place_right_of(left_ref, right_ref, gap_nm: float, align_y: str = "center"):
    """
    Place right_ref to the right of left_ref with a given horizontal gap.

    Parameters
    ----------
    left_ref : PHIDL DeviceReference
    right_ref : PHIDL DeviceReference
    gap_nm : float
        Desired horizontal spacing between left_ref.xmax and right_ref.xmin.
    align_y : str
        One of: "center", "ymin", "ymax".
    """
    if gap_nm < 0:
        raise ValueError("gap_nm must be non-negative.")

    if align_y == "center":
        y_target = _y_center(left_ref)
        y_origin = _y_center(right_ref)
    elif align_y == "ymin":
        y_target = left_ref.ymin
        y_origin = right_ref.ymin
    elif align_y == "ymax":
        y_target = left_ref.ymax
        y_origin = right_ref.ymax
    else:
        raise ValueError("align_y must be one of: 'center', 'ymin', 'ymax'.")

    right_ref.move(
        origin=(right_ref.xmin, y_origin),
        destination=(left_ref.xmax + gap_nm, y_target),
    )
    return right_ref