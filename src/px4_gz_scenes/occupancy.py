"""
3-D occupancy grid exporter for px4_gz_scenes environments.

Converts a :class:`~px4_gz_scenes.scene.Scene` into a dense boolean NumPy
array of shape ``(Nx, Ny, Nz)`` where ``True`` means the voxel centre falls
inside at least one :class:`~px4_gz_scenes.scene_object.SceneObject`.

Grid parameters are derived from the scene:

* **cell size** — ``scene.resolution`` (metres)
* **grid dimensions** — ``Ni = ceil(scene.extent[i] / resolution)``
* **voxel centre** — ``((i + 0.5) * res, (j + 0.5) * res, (k + 0.5) * res)``

The implementation is fully vectorised using NumPy broadcasting; there is no
Python triple-loop over voxels.  For a 15 × 15 × 3 m room at 0.2 m resolution
the working array is ~20 MB — well within memory budgets for interactive use.

Usage::

    import numpy as np
    from px4_gz_scenes import get_scene
    from px4_gz_scenes.occupancy import to_occupancy_grid

    grid = to_occupancy_grid(get_scene("room"))
    print(grid.shape, grid.sum())   # (75, 75, 15)  <occupied count>
    np.save("room_occ.npy", grid)
"""

from __future__ import annotations

import math

import numpy as np

from px4_gz_scenes._math import apply_rotation, quat_to_rotation_matrix
from px4_gz_scenes.scene import Scene
from px4_gz_scenes.scene_object import SceneObject
from px4_gz_scenes.shapes import Box, Composite, Cylinder, Shape, Sphere


# ── Per-shape containment helpers ───────────────────────────────────────────
# All helpers receive *local_pts* of shape (Nx, Ny, Nz, 3) already expressed
# in the shape's local frame (origin at the shape centre).  They return a
# boolean mask of the same leading shape (Nx, Ny, Nz).

def _inside_box(local_pts: np.ndarray, shape: Box) -> np.ndarray:
    hx, hy, hz = shape.size[0] / 2, shape.size[1] / 2, shape.size[2] / 2
    return (
        (np.abs(local_pts[..., 0]) <= hx)
        & (np.abs(local_pts[..., 1]) <= hy)
        & (np.abs(local_pts[..., 2]) <= hz)
    )


def _inside_cylinder(local_pts: np.ndarray, shape: Cylinder) -> np.ndarray:
    r2 = local_pts[..., 0] ** 2 + local_pts[..., 1] ** 2
    return (r2 <= shape.radius ** 2) & (np.abs(local_pts[..., 2]) <= shape.length / 2)


def _inside_sphere(local_pts: np.ndarray, shape: Sphere) -> np.ndarray:
    d2 = np.sum(local_pts ** 2, axis=-1)
    return d2 <= shape.radius ** 2


def _inside_shape(local_pts: np.ndarray, shape: Shape) -> np.ndarray:
    """Dispatch containment test to the appropriate shape helper.

    For :class:`~px4_gz_scenes.shapes.Composite` shapes the function recurses
    into each child, shifting and rotating ``local_pts`` by the child's offset
    and rotation before testing.
    """
    if isinstance(shape, Box):
        return _inside_box(local_pts, shape)

    if isinstance(shape, Cylinder):
        return _inside_cylinder(local_pts, shape)

    if isinstance(shape, Sphere):
        return _inside_sphere(local_pts, shape)

    if isinstance(shape, Composite):
        mask = np.zeros(local_pts.shape[:-1], dtype=bool)
        for child_shape, (ox, oy, oz), child_rot in shape.children:
            child_local = local_pts - np.array([ox, oy, oz], dtype=np.float64)
            R_child = quat_to_rotation_matrix(child_rot)
            # Inverse rotation: apply R_child.T (= R_child^{-1} for unit quat)
            child_local = child_local @ R_child  # pts @ R.T where we pass R.T as R_child
            mask |= _inside_shape(child_local, child_shape)
        return mask

    raise TypeError(f"Unknown shape type: {type(shape)!r}")


def _mark_object(
    grid: np.ndarray,
    obj: SceneObject,
    world_pts: np.ndarray,
) -> None:
    """OR-accumulate occupied voxels for one SceneObject into *grid*."""
    px, py, pz = obj.position
    # Translate voxel centres to the object's local frame origin.
    local_pts = world_pts - np.array([px, py, pz], dtype=np.float64)
    # Apply the inverse rotation (conjugate quaternion ↔ transposed matrix).
    R = quat_to_rotation_matrix(obj.rotation)
    local_pts = apply_rotation(R.T, local_pts)
    grid |= _inside_shape(local_pts, obj.shape)


# ── Public API ───────────────────────────────────────────────────────────────

def to_occupancy_grid(scene: Scene) -> np.ndarray:
    """Return a 3-D occupancy grid for *scene*.

    Args:
        scene: The scene to voxelise.  Uses ``scene.resolution`` for cell size
               and ``scene.extent`` for grid dimensions.

    Returns:
        Boolean NumPy array of shape ``(Nx, Ny, Nz)`` where index ``(i, j, k)``
        is ``True`` if the voxel centre
        ``((i + 0.5) * res, (j + 0.5) * res, (k + 0.5) * res)``
        is inside at least one :class:`~px4_gz_scenes.scene_object.SceneObject`.
    """
    res = scene.resolution
    nx = math.ceil(scene.extent[0] / res)
    ny = math.ceil(scene.extent[1] / res)
    nz = math.ceil(scene.extent[2] / res)

    # Build the (Nx, Ny, Nz, 3) array of voxel centres in world frame.
    ii = (np.arange(nx, dtype=np.float64) + 0.5) * res
    jj = (np.arange(ny, dtype=np.float64) + 0.5) * res
    kk = (np.arange(nz, dtype=np.float64) + 0.5) * res
    Xi, Yj, Zk = np.meshgrid(ii, jj, kk, indexing="ij")
    world_pts = np.stack([Xi, Yj, Zk], axis=-1)  # (Nx, Ny, Nz, 3)

    grid = np.zeros((nx, ny, nz), dtype=bool)
    for obj in scene.objects:
        _mark_object(grid, obj, world_pts)

    return grid
