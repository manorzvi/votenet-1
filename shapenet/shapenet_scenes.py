import sys
import os
import argparse
import random
import json

import numpy as np
from loguru import logger

from utils import set_seed, get_all_files, load_npy_pc, sample_npy_pc, check_intersection, \
    get_pc_measures, get_bbox_from_measures
from shapenet_transforms import ShapenetTransforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util


class Parser(object):

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", '-s', type=int, default=42)
        parser.add_argument("--debug", '-d', action="store_true", default=False)

        parser.add_argument("--min-n-objects", type=int, default=1)
        parser.add_argument("--max-n-objects", type=int, default=16)
        parser.add_argument("--with-repetition", action="store_true", default=False)
        parser.add_argument("--rotate", action="store_true", default=False)
        parser.add_argument("--pc-n-samples", type=int)
        parser.add_argument("--n-scenes", type=int, default=10)
        parser.add_argument("--excludes", type=str, nargs="+", default=[])

        parser.add_argument("--input-path", type=str, required=True)
        parser.add_argument("--output-path", type=str, required=True)

        parser.add_argument("--step-size", type=float, default=1.0)

        opts = parser.parse_args()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        opts.input_path = os.path.join(base_dir, opts.input_path)
        opts.output_path = os.path.join(base_dir, opts.output_path)

        assert os.path.exists(opts.input_path), f'{opts.input_path} does not exist!'

        return opts

    @staticmethod
    def str(opts):
        s = '\n'
        for k, v in opts.__dict__.items():
            s += '\t{0:<20}: {1}\n'.format(k, v)
        return s


if __name__ == '__main__':
    opts = Parser.parse()
    logger.info(Parser.str(opts))
    set_seed(opts.seed)

    try:
        os.mkdir(opts.output_path)
    except FileExistsError as e:
        logger.warning(e)

    files = get_all_files(opts.input_path, opts.excludes)
    # files = [(int(f.split('/')[-1][:-4]), f) for f in files]
    # files.sort()
    # files = [f for _, f in files]

    sampler = random.choices if opts.with_repetition else random.sample
    transforms = ShapenetTransforms()

    for i in range(opts.n_scenes):
        k = random.randint(opts.min_n_objects, opts.max_n_objects)
        logger.info(f"Rendering scene {i} with {k} objects ...")

        paths = sampler(population=files, k=k)

        scene_data = {
            'object_pc' : [],
            'scene_pc': [],
            'scene_bbox': []
        }

        for path in paths:
            obj_pc = load_npy_pc(path)
            obj_pc = transforms.to_standard(obj_pc)

            _, _, _, _, _, _, centerx, centery, centerz, sizex, sizey, sizez = get_pc_measures(obj_pc)

            rotation = 0.0
            pc = obj_pc.copy()
            if opts.rotate:
                rotation = 2 * np.pi * np.random.rand(1)
                pc = transforms.to_rotate(pc, alpha=rotation)

            obj_bbox = get_bbox_from_measures(centerx, centery, centerz, sizex, sizey, sizez, rotation, path.split('/')[-2])

            obj_pc, choices = sample_npy_pc(obj_pc, opts.pc_n_samples)

            t_vec = transforms.rand_unit2_vector()
            
            bbox = obj_bbox.copy()
            overlapping_bboxes = True
            counter = 0
            while overlapping_bboxes:
                pc = transforms.to_translate(pc, t_vec, opts.step_size)
                pc_center = np.mean(pc, axis=0)
                bbox[0] = pc_center[0]
                bbox[1] = pc_center[1]
                bbox[2] = pc_center[2]

                overlapping_bboxes = check_intersection(pc, bboxes=scene_data['scene_bbox'])

                counter += 1

                if counter > 24:
                    logger.warning('Early-stopping add bboxes to scene loop')
                    overlapping_bboxes = False

            scene_data['object_pc'].append(obj_pc)
            scene_data['scene_pc'].append(pc)
            scene_data['scene_bbox'].append(bbox)

        if opts.debug:
            # logger.debug('Plot scene')
            # scene_pc = np.concatenate(scene_data['scene_pc'], axis=0)
            # plot_scene_pc(scene_pc, scene_data['scene_bbox'])
            pc_util.write_ply(np.concatenate(scene_data['scene_pc']), os.path.join('.', 'pc.ply'))
            pc_util.write_oriented_bbox(np.vstack(scene_data['scene_bbox'])[:, :-1], os.path.join('.', 'bbox.ply'))


        # Save scene
        scene_data['object_pc'] = [x.tolist() for x in scene_data['object_pc']]
        scene_data['scene_pc'] = [x.tolist() for x in scene_data['scene_pc']]
        scene_data['scene_bbox'] = [x.tolist() for x in scene_data['scene_bbox']]

        data_path = os.path.join(opts.output_path, str(i).zfill(5)) + '.json'
        with open(data_path, "w") as f:
            json.dump(scene_data, f, indent=4)
        
        





