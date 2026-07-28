from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .export import SimulationLayout


DEFAULT_GATE_COLORS = {
    "plunger": "#e41a1c",
    "barrier": "#ff7f00",
    "ohmic": "#984ea3",
    "screening": "#4daf4a",
}

DEFAULT_MATERIAL_COLORS = {
    "Ge": "#4daf4a",
    "SiGe": "#377eb8",
    "Al2O3": "#bdbdbd",
    "Sapphire_zb": "#bdbdbd",
}


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Plotly is required for qd_design visualization helpers."
        ) from error
    return go


def _signed_polygon_area(points: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _point_in_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> bool:
    def cross(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        return float(
            (v[0] - u[0]) * (w[1] - u[1])
            - (v[1] - u[1]) * (w[0] - u[0])
        )

    c1 = cross(a, b, point)
    c2 = cross(b, c, point)
    c3 = cross(c, a, point)
    return c1 >= -tolerance and c2 >= -tolerance and c3 >= -tolerance


def triangulate_polygon(
    polygon_xy: Sequence[Sequence[float]],
) -> list[tuple[int, int, int]]:
    """Ear-clip a simple polygon without replacing it by its convex hull."""
    points = np.asarray(polygon_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("polygon_xy must contain at least three (x, y) vertices.")

    remaining = list(range(len(points)))
    if _signed_polygon_area(points) < 0:
        remaining.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3:
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = points[previous], points[current], points[following]
            edge_ab = b - a
            edge_bc = c - b
            cross = edge_ab[0] * edge_bc[1] - edge_ab[1] * edge_bc[0]
            if cross <= 1e-12:
                continue
            if any(
                _point_in_triangle(points[index], a, b, c)
                for index in remaining
                if index not in {previous, current, following}
            ):
                continue
            triangles.append((previous, current, following))
            remaining.pop(position)
            ear_found = True
            break

        guard += 1
        if not ear_found or guard > len(points) ** 2:
            raise ValueError("Could not triangulate polygon; check that it is simple.")

    triangles.append(tuple(remaining))
    return triangles


def _box_mesh_arrays(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return vertices, faces


def polygonal_prism_mesh_arrays(
    polygon_xy: Sequence[Sequence[float]],
    z_min_nm: float,
    z_max_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact vertices/faces for a polygonal prism."""
    points = np.asarray(polygon_xy, dtype=float)
    if _signed_polygon_area(points) < 0:
        points = points[::-1]
    triangles = triangulate_polygon(points)
    count = len(points)
    vertices = np.vstack(
        [
            np.column_stack([points, np.full(count, z_min_nm)]),
            np.column_stack([points, np.full(count, z_max_nm)]),
        ]
    )

    faces: list[tuple[int, int, int]] = []
    for a, b, c in triangles:
        faces.append((c, b, a))
        faces.append((a + count, b + count, c + count))
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, following + count))
        faces.append((index, following + count, index + count))
    return vertices, np.asarray(faces, dtype=int)


def _element_value(element: Any, key: str) -> Any:
    if isinstance(element, Mapping):
        return element[key]
    return getattr(element, key)


def plot_layout_spec_2d(
    layout_elements: Sequence[Any],
    *,
    title: str = "Gate layout",
    gate_colors: Optional[Mapping[str, str]] = None,
):
    """Plot exact layout polygons from either dictionaries or PlacedElements."""
    go = _plotly_graph_objects()
    colors = {**DEFAULT_GATE_COLORS, **dict(gate_colors or {})}
    fig = go.Figure()
    for element in layout_elements:
        name = str(_element_value(element, "name"))
        gate_type = str(_element_value(element, "gate_type"))
        polygons = _element_value(element, "polygon_xy_nm")
        for polygon_index, polygon in enumerate(polygons, start=1):
            points = np.asarray(polygon, dtype=float)
            closed = np.vstack([points, points[0]])
            trace_name = name if len(polygons) == 1 else f"{name}_{polygon_index}"
            fig.add_trace(
                go.Scatter(
                    x=closed[:, 0],
                    y=closed[:, 1],
                    mode="lines",
                    fill="toself",
                    name=trace_name,
                    line={"color": "#222222", "width": 2},
                    fillcolor=colors.get(gate_type, "#777777"),
                    hovertemplate=(
                        f"{trace_name}<br>type={gate_type}<br>"
                        "x=%{x:.1f} nm<br>y=%{y:.1f} nm<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        height=560,
        xaxis={"title": "x (nm)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "y (nm)"},
        margin={"l": 55, "r": 20, "t": 50, "b": 50},
    )
    return fig


def plot_simulation_layout_3d(
    simulation_layout: SimulationLayout,
    *,
    z_range_nm: Optional[tuple[float, float]] = (-120.0, 180.0),
    gate_colors: Optional[Mapping[str, str]] = None,
    material_colors: Optional[Mapping[str, str]] = None,
    show_polygon_outlines: bool = True,
):
    """Render background boxes and exact patterned polygonal prisms."""
    go = _plotly_graph_objects()
    gates = {**DEFAULT_GATE_COLORS, **dict(gate_colors or {})}
    materials = {**DEFAULT_MATERIAL_COLORS, **dict(material_colors or {})}
    fig = go.Figure()

    for region in simulation_layout.background_regions:
        z0, z1 = region.z_min_nm, region.z_max_nm
        if z_range_nm is not None:
            view_z0, view_z1 = z_range_nm
            if z1 < view_z0 or z0 > view_z1:
                continue
            z0, z1 = max(z0, view_z0), min(z1, view_z1)
        vertices, faces = _box_mesh_arrays(
            (region.x_min_nm, region.x_max_nm),
            (region.y_min_nm, region.y_max_nm),
            (z0, z1),
        )
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=materials.get(region.material, "#cccccc"),
                opacity=0.10,
                name=region.name,
                hovertemplate=region.name + "<extra></extra>",
                showscale=False,
            )
        )

    for region in simulation_layout.patterned_regions:
        polygons = region.polygon_xy_nm or [
            [
                (region.x_min_nm, region.y_min_nm),
                (region.x_max_nm, region.y_min_nm),
                (region.x_max_nm, region.y_max_nm),
                (region.x_min_nm, region.y_max_nm),
            ]
        ]
        for polygon_index, polygon in enumerate(polygons, start=1):
            name = region.name if len(polygons) == 1 else f"{region.name}_{polygon_index}"
            vertices, faces = polygonal_prism_mesh_arrays(
                polygon,
                region.z_min_nm,
                region.z_max_nm,
            )
            color = gates.get(region.gate_type, "#666666")
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.84,
                    name=name,
                    hovertemplate=name + "<extra></extra>",
                    showscale=False,
                    flatshading=True,
                )
            )
            if show_polygon_outlines:
                points = np.asarray(polygon, dtype=float)
                closed = np.vstack([points, points[0]])
                for z_value in (region.z_min_nm, region.z_max_nm):
                    fig.add_trace(
                        go.Scatter3d(
                            x=closed[:, 0],
                            y=closed[:, 1],
                            z=np.full(len(closed), z_value),
                            mode="lines",
                            line={"color": "#222222", "width": 5},
                            name=f"{name} outline",
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    title = "Polygon-preserving 3D simulation layout"
    if z_range_nm is not None:
        title += f" (z = {z_range_nm[0]:g} to {z_range_nm[1]:g} nm)"
    fig.update_layout(
        title=title,
        height=760,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        legend={"itemsizing": "constant"},
        scene={
            "xaxis_title": "x (nm)",
            "yaxis_title": "y (nm)",
            "zaxis": {
                "title": "z (nm)",
                "range": list(z_range_nm) if z_range_nm is not None else None,
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.4, "y": 0.8, "z": 0.8},
            "camera": {
                "eye": {"x": 1.15, "y": -1.35, "z": 2.15},
                "projection": {"type": "orthographic"},
            },
        },
    )
    return fig


__all__ = [
    "DEFAULT_GATE_COLORS",
    "DEFAULT_MATERIAL_COLORS",
    "plot_layout_spec_2d",
    "plot_simulation_layout_3d",
    "polygonal_prism_mesh_arrays",
    "triangulate_polygon",
]
