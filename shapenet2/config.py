import argparse
import os
import numpy as np
import pickle

from itertools import groupby
from loguru import logger
from shapenet2.utils import get_all_files
from typing import Dict, List


class ShapenetConfig:
    class_name_to_original_class_id = {
        "airplane": "02691156",
        "bag": "02773838",
        "cap": "02954340",
        "car": "02958343",
        "chair": "03001627",
        "earphone": "03261776",
        "guitar": "03467517",
        "knife": "03624134",
        "lamp": "03636649",
        "laptop": "03642806",
        "motorbike": "03790512",
        "mug": "03797390",
        "pistol": "03948459",
        "rocket": "04099429",
        "skateboard": "04225987",
        "table": "04379243",
    }
    original_class_id_to_class_name = {
        v: k for k, v in class_name_to_original_class_id.items()
    }
    class_name_to_class_id = {
        c: i for i, c in enumerate(class_name_to_original_class_id.keys())
    }
    class_id_to_class_name = {
        v: k for k, v in class_name_to_class_id.items()
    }


class ShapenetDatasetConfig(object):

    def __init__(self, info_path: str, num_heading_bin: int = 12):

        self.type_to_cls = ShapenetConfig.class_name_to_class_id
        self.cls_to_type = ShapenetConfig.class_id_to_class_name

        self.num_class = len(self.type_to_cls.keys())
        self.num_heading_bin = num_heading_bin

        base_dir = os.path.dirname(os.path.abspath(__file__))
        info_path = os.path.join(base_dir, info_path)
        self.cls_mean_size = get_classes_mean_sizes(info_path)

    def size2class(self, size, cls_name):
        """Convert 3D box size (l, w, h) to size class and size residual."""
        size_class = self.type_to_cls[cls_name]
        size_residual = size - self.cls_mean_size[cls_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        """Inverse function to size2class."""
        mean_size = self.cls_mean_size[self.cls_to_type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        """
        Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.
        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that class*(2pi/N) + number = angle
        """
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class."""
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb


def get_classes_mean_sizes(path: str) -> Dict[str, np.ndarray]:

    assert path.endswith(".pkl")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, path)

    with open(output_path, "rb") as f:
        classes_sizes = pickle.load(f)

    return classes_sizes


def calc_classes_mean_sizes(input_paths: List[str]):
    classes = set(x.split('/')[-1] for p in input_paths for x in os.listdir(p))
    classes.difference_update([x for x in classes if x.startswith(".")])

    classes_means = {k: [[], []] for k in classes}
    for path in input_paths:
        files = get_all_files(path, [])
        class_files = {class_name: list(values_iter) for class_name, values_iter in groupby(files, key=lambda x: x.split('/')[-2])}
        for k, v in class_files.items():
            logger.info(f"Processing {path}/{k}")

            pcs = [np.loadtxt(x) for x in v]
            max_sizes = np.vstack([np.max(pc, axis=0) for pc in pcs])
            min_sizes = np.vstack([np.min(pc, axis=0) for pc in pcs])
            mean_size = np.mean(max_sizes - min_sizes, axis=0)
            classes_means[k][0].append(mean_size)
            classes_means[k][1].append(len(v))

    classes_sizes = {}
    for k, v in classes_means.items():
        means, nums = v
        weights = nums / np.sum(nums)
        weighted_means = np.vstack(means) * weights[..., None]
        classes_sizes[k] = np.sum(weighted_means, axis=0)

    return classes_sizes


if __name__ == '__main__':

    parser = argparse.ArgumentParser("create scenes with shapenet dataset")
    parser.add_argument("--output-name", type=str, default="mean_sizes")
    opts = parser.parse_args()

    logger.info(vars(opts))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_paths = [os.path.join(base_dir, f"data/{mode}") for mode in ("train", "val")]
    opts.output_name = os.path.join(base_dir, f"data/{opts.output_name}") + ".pkl"

    assert all(os.path.exists(x) for x in input_paths)

    if os.path.exists(opts.output_name):
        logger.warning(f"[Errno 17] File exists: '{opts.output_path}'")

    classes_sizes = calc_classes_mean_sizes(input_paths)

    with open(opts.output_name, "wb") as f:
        pickle.dump(classes_sizes, f)
