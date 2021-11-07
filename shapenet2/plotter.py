import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from shapenet2.utils import get_bbox_2d_from_bbox_1d
from typing import List


def draw_bbox_2d(bbox_2d: np.ndarray, ax: plt.Axes, color: str, size: float):
    ax.scatter3D(bbox_2d[:, 0], bbox_2d[:, 1], bbox_2d[:, 2], color=color)

    for j in range(bbox_2d.shape[0]):
        ax.text(bbox_2d[j, 0], bbox_2d[j, 1], bbox_2d[j, 2], f"{j}", color=color)
    for m in range(0, 4):
        ax.plot(
            np.asarray([bbox_2d[m, 0], bbox_2d[(m + 1) % 4, 0]]),
            np.asarray([bbox_2d[m, 1], bbox_2d[(m + 1) % 4, 1]]),
            np.asarray([bbox_2d[m, 2], bbox_2d[(m + 1) % 4, 2]]),
            color=color, linewidth=size
        )
        ax.plot(
            np.asarray([bbox_2d[m + 4, 0], bbox_2d[(m + 5) % 4 + 4, 0]]),
            np.asarray([bbox_2d[m + 4, 1], bbox_2d[(m + 5) % 4 + 4, 1]]),
            np.asarray([bbox_2d[m + 4, 2], bbox_2d[(m + 5) % 4 + 4, 2]]),
            color=color, linewidth=size
        )
        ax.plot(
            np.asarray([bbox_2d[m, 0], bbox_2d[m + 4, 0]]),
            np.asarray([bbox_2d[m, 1], bbox_2d[m + 4, 1]]),
            np.asarray([bbox_2d[m, 2], bbox_2d[m + 4, 2]]),
            color=color, linewidth=size
        )

    return ax


def plot_scene_pc(pcs: List[np.ndarray], bboxes_1d: List[np.ndarray]):

    plt.figure(figsize=(16, 16))
    ax = plt.axes(projection="3d")

    base_colors = mcolors.BASE_COLORS
    pc_color = base_colors['k']
    corners3d_color = base_colors['b']
    pc_size = 1
    corners3d_size = 0.5
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for pc in pcs:
        ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], color=pc_color, s=pc_size)
    for bbox_1d in bboxes_1d:
        bbox_2d = get_bbox_2d_from_bbox_1d(bbox_1d)
        draw_bbox_2d(bbox_2d, ax, corners3d_color, corners3d_size)
    plt.show()
