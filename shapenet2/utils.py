import random
import os
import numpy as np

from loguru import logger
from shapenet2.transforms import ShapenetTransforms
from typing import List, Tuple


def set_seed(seed: int = None):
    """
    Sets the randomization seed.

    :param seed: integer to set as seed.
    """
    if seed:
        logger.info(f"Setting random seed (={seed})")
        random.seed(seed)
        logger.info(f"Setting numpy random seed (={seed})")
        np.random.seed(seed)


def get_axis_aligned_pc_size(pc: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    minx, maxx = np.min(pc[:, 0]), np.max(pc[:, 0])
    miny, maxy = np.min(pc[:, 1]), np.max(pc[:, 1])
    minz, maxz = np.min(pc[:, 2]), np.max(pc[:, 2])

    return minx, maxx, miny, maxy, minz, maxz


def get_bbox_2d_from_pc(pc: np.ndarray) -> np.ndarray:
    minx, maxx, miny, maxy, minz, maxz = get_axis_aligned_pc_size(pc)

    bbox_2d = np.array(
        [
            [minx, miny, minz],
            [minx, miny, maxz],
            [minx, maxy, minz],
            [minx, maxy, maxz],
            [maxx, miny, minz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [maxx, maxy, maxz],
        ],
        dtype=np.float32,
    )

    return bbox_2d


def get_bbox_1d_from_pc(pc: np.ndarray, heading_angle: float = 0.0) -> np.ndarray:
    minx, maxx, miny, maxy, minz, maxz = get_axis_aligned_pc_size(pc)

    centerx = (maxx + minx) / 2
    centery = (maxy + miny) / 2
    centerz = (maxz + minz) / 2
    sizex = maxx - minx
    sizey = maxy - miny
    sizez = maxz - minz

    bbox_1d = np.asarray([centerx, centery, centerz, sizex, sizey, sizez, heading_angle], dtype=np.float32)

    return bbox_1d


def get_bbox_2d_from_bbox_1d(bbox_1d: np.ndarray) -> np.ndarray:
    """
    Calculates the bbox 2d (8 points) by a bbox 1d (vector).

    :param bbox_1d: bbox 1d as np.ndarry of shape (8, ).

    :return: bbox_2d as np.ndarry of shape (8, 3).
    """
    sizex, sizey, sizez = bbox_1d[3:6] / 2
    bbox_2d_at_the_origin = np.array(
        [
            [-sizex, +sizey, +sizez],
            [+sizex, +sizey, +sizez],
            [+sizex, -sizey, +sizez],
            [-sizex, -sizey, +sizez],
            [-sizex, +sizey, -sizez],
            [+sizex, +sizey, -sizez],
            [+sizex, -sizey, -sizez],
            [-sizex, -sizey, -sizez],
        ],
        dtype=np.float32,
    )

    rotation = ShapenetTransforms.rotz(bbox_1d[6])
    bbox_2d = bbox_2d_at_the_origin @ rotation
    bbox_2d = bbox_2d + bbox_1d[:3]

    return bbox_2d


def get_all_files(datapath: str, exclude: List[str]):
    """
    Retrieve all files in dirs given by a path.

    :param datapath: path to classes data.
    :param exclude: list of dirs to exclude.

    :return: list of paths to all files (excluding files in dirs listed by `exclude`).
    """
    assert os.path.exists(datapath), f'{datapath} does not exist!'
    files = []
    for classname in os.listdir(datapath):
        if classname in exclude or classname.startswith("."):
            continue
        classpath = os.path.join(datapath, classname)
        files += [os.path.join(datapath, classname, f) for f in os.listdir(classpath)]
    return files


def l2_distance(ref: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance between a 3d reference point and all points of a pointcloud.

    :param ref: reference 3d point as np.ndarray of shape (3, ).
    :param pc: pointcloud as np.ndarray of shape (n, 3).

    :return: Euclidean distance between ref to each of the n points in pointcloud,
             as np.ndarray of shape (n, ).
    """
    return np.sum((ref - pc) ** 2, axis=1)


def farthest_point_sampling(pc: np.ndarray, k: int, start_ind: int = None, thresh: float = 1.0):
    """
    An approximation of FPS.
    FPS performs over the actual points' distances over the 3d surface,
    while this implementation performs over the euclidean points' distances.
    FPS requires computing the geodesic distances, and is very expensive.
    In this implementation we also consider performance boost, and only sample
    in case the ratio of required points is lower than a given threshold.

    :param pc: pointcloud to sample, as np.ndarray of shape (n, 3).
    :param k: number of points to sample.
    :param start_ind: index of a point to start with. randomized when `None`.
    :param thresh: pointcloud is sampled only if number of ratio of required
                   point is lower than this value.

    :return:
        sampled pointcloud (by approximated FPS), as np.ndarray of shape (k, 3).
    """

    assert 0.0 <= thresh <= 1.0

    if k / pc.shape[0] > thresh:
        indices = np.random.choice(pc.shape[0], k, replace=k > pc.shape[0])
        farthest_pts = pc[indices]
        return farthest_pts

    farthest_pts = np.zeros((k, 3))
    farthest_pts[0] = pc[start_ind if start_ind else np.random.randint(len(pc))]
    distances = l2_distance(farthest_pts[0], pc)
    for i in range(1, k):
        farthest_pts[i] = pc[np.argmax(distances)]
        distances = np.minimum(distances, l2_distance(farthest_pts[i], pc))
    return farthest_pts


def check_intersection(pc: np.ndarray, bboxes_1d: List[np.ndarray], max_intersection_thresh: float = 0.0) -> bool:
    """
    Check wheather a given pointcloud intersects any of given bboxes (1d).

    :param pc: pointcloud as np.ndarray of shape (n, 3).
    :param bboxes_1d: list of 1d bboxed, each is np.ndarray of shape (8, ).
    :param max_intersection_thresh: maximal allowed intersection ratio. whenever pointcloud intersects
            any of the bboxes under this value, it is not counted as an intersection.

    :return: bool indicating True when pointcloud intersects any of bboxes, else False.
    """
    if len(bboxes_1d) == 0:
        return False  # No intersection

    for bbox in bboxes_1d:
        centerx, centery, centerz, sizex, sizey, sizez, heading_angle = bbox
        minx = centerx - sizex / 2
        maxx = centerx + sizex / 2
        miny = centery - sizey / 2
        maxy = centery + sizey / 2
        minz = centerz - sizez / 2
        maxz = centerz + sizez / 2

        in_x = (minx <= pc[:, 0]) * (pc[:, 0] <= maxx)
        in_y = (miny <= pc[:, 1]) * (pc[:, 1] <= maxy)
        in_z = (minz <= pc[:, 2]) * (pc[:, 2] <= maxz)

        in_p = in_x * in_y * in_z

        nb_intersecting_pts = np.sum(in_p)
        if nb_intersecting_pts / len(in_p) > max_intersection_thresh:
            return True

    return False


def random_sampling(pc: np.ndarray, num_sample: int, replace: bool = None, return_choices: bool = False):
    """ Input is (N, C), output is (num_sample, C)."""
    replace = (pc.shape[0] < num_sample) if replace is None else replace
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]
