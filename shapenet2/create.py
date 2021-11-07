import argparse
import os
import numpy as np
import pickle
import random
import time

from loguru import logger
from shapenet2.plotter import plot_scene_pc
from shapenet2.transforms import ShapenetTransforms
from shapenet2.utils import check_intersection, get_all_files, get_bbox_1d_from_pc, farthest_point_sampling, set_seed
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("create scenes with shapenet dataset")

    parser.add_argument("--seed", '-s', type=int, default=None)
    parser.add_argument("--debug", '-d', action="store_true", default=False)

    parser.add_argument("--mode", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--output-name", type=str, required=True)

    parser.add_argument("--min-nb-objects", type=int, default=8)
    parser.add_argument("--max-nb-objects", type=int, default=16)
    parser.add_argument("--max-intersection-thresh", type=float, default=0.0)
    parser.add_argument("--with-repetition", action="store_true", default=False)
    parser.add_argument("--pc-nb-samples", type=int, default=1024)
    parser.add_argument("--nb-scenes", type=int, default=10)
    parser.add_argument("--excludes", type=str, nargs="+", default=[])

    parser.add_argument("--step-size", type=float, default=1.0)

    opts = parser.parse_args()

    return opts


def main():

    opts = parse_args()
    logger.info(vars(opts))

    set_seed(opts.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    opts.input_path = os.path.join(base_dir, f"data/{opts.mode}")
    opts.output_path = os.path.join(base_dir, f"scenes/{opts.output_name}/{opts.mode}")

    assert os.path.exists(opts.input_path), f"{opts.input_path} does not exist!"

    try:
        os.makedirs(opts.output_path)
    except FileExistsError as e:
        logger.warning(e)

    files = get_all_files(opts.input_path, opts.excludes)
    sampler = random.choices if opts.with_repetition else random.sample

    start_time = time.time()
    for i in range(opts.nb_scenes):
        scene_id = str(i).zfill(5)
        k = random.randint(opts.min_nb_objects, opts.max_nb_objects)

        perc = "{:.3f}".format(100. * i / opts.nb_scenes)
        logger.info("[{}%] Rendering scene {} with {} objects ...".format(
            perc.zfill(6), scene_id, str(k).zfill(2))
        )

        # FIXME: classes are uneven, thus samples paths are highly uneven.
        #   consider weighting classes differently to have equal probabilities
        #   of drawing samples from all classes (pass `weights=...` to samples)
        paths = sampler(population=files, k=k)

        scene_data = create_single_scene(opts=opts, files=paths)

        data_path = os.path.join(opts.output_path, scene_id) + '.pkl'
        with open(data_path, 'wb') as f:
            pickle.dump(scene_data, f)
    end_time = time.time()

    logger.info("[100.00%] Rendered {} scenes ({:.3f} [s])".format(opts.nb_scenes, end_time - start_time))


def create_single_scene(opts: argparse.Namespace, files: List[str]) -> Dict[str, List[np.ndarray]]:

    scene_data = {"obj_pc": [], "obj_class": [], "scene_pc": [], "scene_bbox": []}

    for file in files:
        rotation = 2 * np.pi * np.random.rand(1)
        translation = ShapenetTransforms.rand_unit2_vector()

        obj_pc = np.loadtxt(file)
        obj_pc = ShapenetTransforms.to_standard(obj_pc)

        # FIXME: do we need to sample before generating bbox? I think yes to have it tight (nitai 09/10/21).
        obj_pc = farthest_point_sampling(obj_pc, k=opts.pc_nb_samples)
        obj_bbox_1d = get_bbox_1d_from_pc(obj_pc, heading_angle=rotation)

        pc = obj_pc.copy()
        pc = ShapenetTransforms.to_rotate(pc, alpha=rotation)
        bbox = obj_bbox_1d.copy()

        overlapping_bboxes = True
        while overlapping_bboxes:
            pc = pc + translation * opts.step_size
            bbox[:3] = bbox[:3] + translation * opts.step_size
            overlapping_bboxes = check_intersection(
                pc, bboxes_1d=scene_data["scene_bbox"], max_intersection_thresh=opts.max_intersection_thresh
            )

        scene_data["obj_pc"].append(obj_pc)
        scene_data["obj_class"].append(file.split("/")[-2])
        scene_data["scene_pc"].append(pc)
        scene_data["scene_bbox"].append(bbox)

    assert all(len(x) == len(files) for x in scene_data.values())

    # if opts.debug:
    #     plot_scene_pc(scene_data["scene_pc"], scene_data["scene_bbox"])

    return scene_data


if __name__ == "__main__":
    main()
