import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import get_3dcorners_from_bbox


def draw_pc(pc: np.ndarray, ax, color: str, size: int):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], color=color, s=size)
    return ax


def draw_votes(pc: np.ndarray, votes: np.ndarray, votes_mask: np.ndarray, ax, color: str, size: float):

        choices = np.random.randint(0, pc.shape[0], 1000)
        choices_vec = np.zeros_like(votes_mask)
        choices_vec[choices] = 1

        ax.quiver(pc[(votes_mask * choices_vec).astype(bool), 0],
                  pc[(votes_mask * choices_vec).astype(bool), 1],
                  pc[(votes_mask * choices_vec).astype(bool), 2],
                  votes[(votes_mask * choices_vec).astype(bool), 0],
                  votes[(votes_mask * choices_vec).astype(bool), 1],
                  votes[(votes_mask * choices_vec).astype(bool), 2],
                  color=color, linewidth=size)
        return ax


def draw_corners3d(corners3d: np.ndarray, ax, color: str, size: float):
    ax.scatter3D(corners3d[:, 0], corners3d[:, 1], corners3d[:, 2], color=color)

    for j in range(corners3d.shape[0]):
        ax.text(corners3d[j, 0], corners3d[j, 1], corners3d[j, 2], '{0}'.format(j), color=color)
    for m in range(0, 4):
        ax.plot(
            np.asarray([corners3d[m, 0], corners3d[(m + 1) % 4, 0]]),
            np.asarray([corners3d[m, 1], corners3d[(m + 1) % 4, 1]]),
            np.asarray([corners3d[m, 2], corners3d[(m + 1) % 4, 2]]),
            color=color, linewidth=size
        )
        ax.plot(
            np.asarray([corners3d[m + 4, 0], corners3d[(m + 5) % 4 + 4, 0]]),
            np.asarray([corners3d[m + 4, 1], corners3d[(m + 5) % 4 + 4, 1]]),
            np.asarray([corners3d[m + 4, 2], corners3d[(m + 5) % 4 + 4, 2]]),
            color=color, linewidth=size
        )
        ax.plot(
            np.asarray([corners3d[m, 0], corners3d[m + 4, 0]]),
            np.asarray([corners3d[m, 1], corners3d[m + 4, 1]]),
            np.asarray([corners3d[m, 2], corners3d[m + 4, 2]]),
            color=color, linewidth=size
        )

    return ax


def plot_pc(pc: np.ndarray, corners3d: np.ndarray, name):

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")

    base_colors = mcolors.BASE_COLORS
    pc_color = base_colors['k']
    corners3d_color = base_colors['b']
    pc_size = 1
    corners3d_size = 0.5
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_title(name)
    draw_pc(pc, ax, pc_color, pc_size)
    draw_corners3d(corners3d, ax, corners3d_color, corners3d_size)
    plt.show()


def plot_scene_pc(pc: np.ndarray, bboxes: List[np.ndarray]):

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
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    draw_pc(pc, ax, pc_color, pc_size)
    for bbox in bboxes:
        corners3d = get_3dcorners_from_bbox(bbox)
        draw_corners3d(corners3d, ax, corners3d_color, corners3d_size)
    plt.show()
