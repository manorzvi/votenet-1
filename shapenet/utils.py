import os
import random
import statistics
import numpy as np
from loguru import logger
from typing import List, Tuple
from config import ShapenetConfig
from shapenet_transforms import ShapenetTransforms


shapenet_config = ShapenetConfig()
shapenet_transforms = ShapenetTransforms()


def set_seed(seed):
    logger.info(f'Setting random seed (={seed})')
    random.seed(seed)
    logger.info(f'Setting numpy random seed (={seed})')
    np.random.seed(seed)


def rename_dirs(datadir: str):
    assert os.path.exists(datadir), f'{datadir} does not exist!'
    for dir in os.listdir(datadir):
        try:
            logger.info(f'Renaming {dir} to {shapenet_config.originalcode2classname[dir]}')
            os.rename(os.path.join(datadir, dir), os.path.join(datadir, shapenet_config.originalcode2classname[dir]))
        except Exception as e:
            print(e)
            logger.warning(f'{dir} does not exist in originalcode2classname')


def get_all_files(datapath, exclude: List[str]):
    assert os.path.exists(datapath), f'{datapath} does not exist!'
    files = []
    for classname in os.listdir(datapath):
        if classname in exclude:
            continue
        classpath = os.path.join(datapath, classname)
        files += [os.path.join(datapath, classname, f) for f in os.listdir(classpath) if f.endswith('.npy')]
    return files


def load_pts_pc(filepath: str, ndarray_savepath: str = None) -> np.ndarray:
    assert os.path.exists(filepath), f'{filepath} does not exist!'
    assert filepath.endswith('.pts'), f"{filepath} does not end with '.pts'!"
    arr = np.loadtxt(filepath)#, comments=("version:", "n_points:", "{", "}"))
    if ndarray_savepath is not None:
        logger.info(f'Saving {ndarray_savepath} as np.ndarray')
        if os.path.exists(ndarray_savepath):
            logger.warning(f'{ndarray_savepath} is already exist! skip saving.')
        else:
            np.save(ndarray_savepath, arr)
    return arr


def load_npy_pc(filepath: str) -> np.ndarray:
    assert os.path.exists(filepath), f'{filepath} does not exist!'
    assert filepath.endswith('.npy'), f"{filepath} does not end with '.npy'!"
    arr = np.load(filepath)
    return arr


def sample_npy_pc(pc: np.ndarray, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if n_samples is not None:
        choices = np.random.choice(pc.shape[0], n_samples, replace=False)
    else:
        choices = np.asarray(list(range(pc.shape[0])))
    pc = pc[choices, :]
    return pc, choices


def get_bbox_from_pc(pc: np.ndarray, pc_classname: str) -> np.ndarray:
    _, _, _, _, _, _, centerx, centery, centerz, sizex, sizey, sizez = get_pc_measures(pc)
    heading_angle = 0.0  # I observed all ShapeNet point clouds are centered, aligned with x axis & scaled (manorz, Oct 2)
    pc_class = shapenet_config.classname2class[pc_classname]
    bbox = np.asarray([centerx, centery, centerz, sizex, sizey, sizez, heading_angle, pc_class], dtype=np.float)
    return bbox


def get_3dcorners_from_bbox(bbox: np.ndarray, orientation: str = 'xy') -> np.ndarray:
    assert bbox.shape in [(1, 8), (8,)], f'bbox shape ({bbox.shape}) should be (1,8) here'

    centerx = bbox[0]
    centery = bbox[1]
    centerz = bbox[2]
    sizex = bbox[3]
    sizey = bbox[4]
    sizez = bbox[5]
    heading_angle = bbox[6]

    if orientation == 'xy':
        R = shapenet_transforms.rotz(-1 * heading_angle)
    elif orientation == 'xz':
        R = shapenet_transforms.roty(-1 * heading_angle)
    elif orientation == 'yz':
        R = shapenet_transforms.rotx(-1 * heading_angle)
    else:
        raise ValueError(f'orientation unknown: orientation={orientation}. valid values: xy, xz, yz')

    l, w, h = sizex/2, sizey/2, sizez/2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += centerx
    corners_3d[1, :] += centery
    corners_3d[2, :] += centerz

    return np.transpose(corners_3d)


def get_pc_measures(pc: np.ndarray):
    minx, maxx = np.min(pc[:, 0]), np.max(pc[:, 0])
    miny, maxy = np.min(pc[:, 1]), np.max(pc[:, 1])
    minz, maxz = np.min(pc[:, 2]), np.max(pc[:, 2])
    centerx = (maxx + minx) / 2
    centery = (maxy + miny) / 2
    centerz = (maxz + minz) / 2
    sizex = maxx - minx
    sizey = maxy - miny
    sizez = maxz - minz
    return minx, maxx, miny, maxy, minz, maxz, centerx, centery, centerz, sizex, sizey, sizez


def check_intersection(pc: np.ndarray, bboxes: List[np.ndarray]) -> bool:
    if len(bboxes) == 0:
        return False  # No intersection

    for bbox in bboxes:
        minx = bbox[0] - bbox[3]/2
        miny = bbox[1] - bbox[4]/2
        minz = bbox[2] - bbox[5]/2
        maxx = bbox[0] + bbox[3]/2
        maxy = bbox[1] + bbox[4]/2
        maxz = bbox[2] + bbox[5]/2

        in_x = (minx <= pc[:, 0]) * (pc[:, 0] <= maxx)
        in_y = (miny <= pc[:, 1]) * (pc[:, 1] <= maxy)
        in_z = (minz <= pc[:, 2]) * (pc[:, 2] <= maxz)

        in_p = in_x * in_y * in_z

        if np.any(in_p):
            return True  # Intersection between the point cloud and any bounding box

    return False


def collect_dataset_stats(data_dir: str, pts: bool = False):
    assert os.path.exists(data_dir), f'{data_dir} does not exist!'

    stats     = {}
    rev_stats = {}

    for classname in os.listdir(data_dir):
        logger.info(f'Collect {classname} stats')
        stats[classname] = {
            'minx': [], 'maxx': [], 'centerx': [], 'sizex': [],
            'miny': [], 'maxy': [], 'centery': [], 'sizey': [],
            'minz': [], 'maxz': [], 'centerz': [], 'sizez': [],
            'n_samples' : 0
        }
        class_dir = os.path.join(data_dir, classname)
        for filename in os.listdir(class_dir):
            if pts:
                if filename.endswith('.npy'):
                    continue
            else:
                if filename.endswith('.pts'):
                    continue
            filepath = os.path.join(class_dir, filename)
            pc = load_npy_pc(filepath)
            minx, maxx, miny, maxy, minz, maxz, centerx, centery, centerz, sizex, sizey, sizez = get_pc_measures(pc)
            stats[classname]['minx'].append(minx)
            stats[classname]['maxx'].append(maxx)
            stats[classname]['miny'].append(miny)
            stats[classname]['maxy'].append(maxy)
            stats[classname]['minz'].append(minz)
            stats[classname]['maxz'].append(maxz)
            stats[classname]['centerx'].append(centerx)
            stats[classname]['centery'].append(centery)
            stats[classname]['centerz'].append(centerz)
            stats[classname]['sizex'].append(sizex)
            stats[classname]['sizey'].append(sizey)
            stats[classname]['sizez'].append(sizez)
            stats[classname]['n_samples'] += 1

        stats[classname]['mean-minx']       = statistics.mean(stats[classname]['minx'])
        stats[classname]['mean-maxx']       = statistics.mean(stats[classname]['maxx'])
        stats[classname]['mean-centerx']    = statistics.mean(stats[classname]['centerx'])
        stats[classname]['mean-sizex']      = statistics.mean(stats[classname]['sizex'])
        stats[classname]['std-minx']        = statistics.stdev(stats[classname]['minx'])
        stats[classname]['std-maxx']        = statistics.stdev(stats[classname]['maxx'])
        stats[classname]['std-centerx']     = statistics.stdev(stats[classname]['centerx'])
        stats[classname]['std-sizex']       = statistics.stdev(stats[classname]['sizex'])

        stats[classname]['mean-miny']       = statistics.mean(stats[classname]['miny'])
        stats[classname]['mean-maxy']       = statistics.mean(stats[classname]['maxy'])
        stats[classname]['mean-centery']    = statistics.mean(stats[classname]['centery'])
        stats[classname]['mean-sizey']      = statistics.mean(stats[classname]['sizey'])
        stats[classname]['std-miny']        = statistics.stdev(stats[classname]['miny'])
        stats[classname]['std-maxy']        = statistics.stdev(stats[classname]['maxy'])
        stats[classname]['std-centery']     = statistics.stdev(stats[classname]['centery'])
        stats[classname]['std-sizey']       = statistics.stdev(stats[classname]['sizey'])

        stats[classname]['mean-minz']       = statistics.mean(stats[classname]['minz'])
        stats[classname]['mean-maxz']       = statistics.mean(stats[classname]['maxz'])
        stats[classname]['mean-centerz']    = statistics.mean(stats[classname]['centerz'])
        stats[classname]['mean-sizez']      = statistics.mean(stats[classname]['sizez'])
        stats[classname]['std-minz']        = statistics.stdev(stats[classname]['minz'])
        stats[classname]['std-maxz']        = statistics.stdev(stats[classname]['maxz'])
        stats[classname]['std-centerz']     = statistics.stdev(stats[classname]['centerz'])
        stats[classname]['std-sizez']       = statistics.stdev(stats[classname]['sizez'])

    for k, v in stats.items():
        for kk, vv in v.items():
            if kk not in rev_stats:
                rev_stats[kk] = {}
            rev_stats[kk][k] = vv
    return stats, rev_stats


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'original_train_data')
    class_dir = os.path.join(data_dir, '03642806')

    # rename_dirs(data_dir)

    # for classname in os.listdir(data_dir):
    #     class_dir = os.path.join(data_dir, classname)
    #     for filename in os.listdir(class_dir):
    #         filepath = os.path.join(class_dir, filename)
    #         load_pts_pc(filepath, filepath[:-4] + '.npy')

    # for file in os.listdir(class_dir):
    #     if file.endswith('.npy'):
    #         continue
    #     pts_path = os.path.join(class_dir, file)
    #     pts_path_ndarray = pts_path[:-4] + '.npy'
    #     pc = load_npy_pc(pts_path)
    #
    #     base_colors = mcolors.BASE_COLORS
    #     pc_color = base_colors['k']
    #     pc_size = 1
    #
    #     fig = plt.figure(figsize=(16, 16))
    #     ax = plt.axes(projection="3d")
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     ax.set_xlim3d(-0.5, 0.5)
    #     ax.set_ylim3d(-0.5, 0.5)
    #     ax.set_zlim3d(-0.5, 0.5)
    #
    #     draw_pc(pc, ax, pc_color, pc_size)
    #     plt.show()

    stats, rev_stats = collect_dataset_stats(data_dir)
    rev_stats = {k: v for k, v in rev_stats.items() if (k.startswith('mean') or k.startswith('std') or k == 'n_samples')}
    for k, v in rev_stats.items():
        print(k, end='\n'+'-'*len(k)+'\n')
        for kk, vv in v.items():
            if isinstance(vv, float):
                print('{0:}: {1:.4f}'.format(kk, vv), end=' | ')
            else:
                print('{}: {}'.format(kk, vv), end=' | ')
        print()




