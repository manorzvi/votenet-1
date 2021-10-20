import os
import random
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torch.utils.data import Dataset, DataLoader
from shapenet_transforms import ShapenetTransforms
from model_util_shapenet import ShapenetDatasetConfig
from plot_utils import draw_pc, draw_corners3d, draw_votes
from utils import get_3dcorners_from_bbox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

DC = ShapenetDatasetConfig()  # dataset specific config
TRANSFORMS = ShapenetTransforms()


class ShapenetDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000, cond_obj_num_points=2000,
                 use_cond_votes=False,
                 use_neg_votes=False, neg_votes_factor=1.0,
                 use_rand_votes=False, rand_votes_factor=1.0,
                 augment=False):

        assert not (use_rand_votes and not use_cond_votes), "use_rand_votes allowed with use_cond_votes only"
        assert not (use_neg_votes and not use_cond_votes), "use_neg_votes allowed with use_cond_votes only"
        assert not (use_neg_votes and use_rand_votes), "use_rand_votes and use_neg_votes are mutually-exclusive"

        self.data_path = os.path.join(BASE_DIR, 'scenes', split_set)
        assert os.path.exists(self.data_path), f'{self.data_path} does not exist!'

        self.scan_names = sorted(list(set([os.path.basename(x) for x in os.listdir(self.data_path)])))

        self.num_points = num_points
        self.cond_obj_num_points = cond_obj_num_points
        self.augment = augment

        self.use_cond_votes = use_cond_votes
        if use_cond_votes:
            print(f'[I] - shapenet {split_set} dataset with cond_votes!')

        self.use_neg_votes = use_neg_votes
        self.neg_votes_factor = neg_votes_factor
        if use_neg_votes:
            print(f'[I] - shapenet {split_set} dataset with neg_votes with neg_votes_factor={neg_votes_factor}!')

        self.use_rand_votes = use_rand_votes
        self.rand_votes_factor = rand_votes_factor
        if use_rand_votes:
            print(f'[I] - shapenet {split_set} dataset with rand_votes with rand_votes_factor={rand_votes_factor}!')

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scan_name = self.scan_names[idx]
        scan_name = os.path.join(self.data_path, scan_name)

        with open(scan_name, "r") as f:
            data = json.load(f)

        n_obj = len(data['object_pc'])
        cond_obj_ind = random.randint(0, n_obj - 1)

        cond_pc = data['object_pc'][cond_obj_ind]
        cond_pc = np.asarray(cond_pc)

        pc = data['scene_pc']
        pc = [np.asarray(x) for x in pc]
        pc = np.concatenate(pc, axis=0)

        bbox = data['scene_bbox'][cond_obj_ind]
        bbox = np.asarray(bbox)

        n_pts = pc.shape[0]
        pc_votes = np.zeros((n_pts, 10))
        if not self.use_cond_votes or \
                (self.use_cond_votes and self.use_rand_votes) or \
                (self.use_cond_votes and self.use_neg_votes):
            pc_votes[:, 0] = 1

        if self.use_rand_votes:
            minx, maxx = np.min(pc[:, 0]), np.max(pc[:, 0])
            miny, maxy = np.min(pc[:, 1]), np.max(pc[:, 1])
            minz, maxz = np.min(pc[:, 2]), np.max(pc[:, 2])
            rand_votesx = np.random.uniform(low=minx*self.rand_votes_factor, high=maxx*self.rand_votes_factor, size=(n_pts, 1))
            rand_votesy = np.random.uniform(low=miny*self.rand_votes_factor, high=maxy*self.rand_votes_factor, size=(n_pts, 1))
            rand_votesz = np.random.uniform(low=minz, high=maxz*self.rand_votes_factor, size=(n_pts, 1))
            rand_votes = np.zeros((n_pts, 3))
            rand_votes[:, 0] = rand_votesx[:, 0]
            rand_votes[:, 1] = rand_votesy[:, 0]
            rand_votes[:, 2] = rand_votesz[:, 0]

        run_ind = 0
        for i in range(n_obj):
            obj_pc = np.asarray(data['scene_pc'][i])
            obj_center = np.mean(obj_pc, axis=0, keepdims=True)
            obj_n_pts = obj_pc.shape[0]

            if self.use_cond_votes and not (self.use_rand_votes or self.use_neg_votes):
                if i == cond_obj_ind:
                    pc_votes[run_ind:run_ind + obj_n_pts, 0] = 1

            pc_votes[run_ind:run_ind + obj_n_pts, 1:4] = np.repeat(obj_center, repeats=obj_n_pts, axis=0) - obj_pc
            pc_votes[run_ind:run_ind + obj_n_pts, 4:7] = np.repeat(obj_center, repeats=obj_n_pts, axis=0) - obj_pc
            pc_votes[run_ind:run_ind + obj_n_pts, 7:10] = np.repeat(obj_center, repeats=obj_n_pts, axis=0) - obj_pc

            if self.use_rand_votes:
                if i != cond_obj_ind:
                    pc_votes[run_ind:run_ind + obj_n_pts, 1:4] = rand_votes[run_ind:run_ind + obj_n_pts] - obj_pc
                    pc_votes[run_ind:run_ind + obj_n_pts, 4:7] = rand_votes[run_ind:run_ind + obj_n_pts] - obj_pc
                    pc_votes[run_ind:run_ind + obj_n_pts, 7:10] = rand_votes[run_ind:run_ind + obj_n_pts] - obj_pc

            if self.use_neg_votes:
                if i != cond_obj_ind:
                    pc_votes[run_ind:run_ind + obj_n_pts, 1:10] = -pc_votes[run_ind:run_ind + obj_n_pts, 1:10] * self.neg_votes_factor

            run_ind += obj_n_pts

        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                pc[:, 0] = -1 * pc[:, 0]
                bbox[:, 0] = -1 * bbox[:, 0]
                bbox[:, 6] = np.pi - bbox[:, 6]
                pc_votes[:, [1, 4, 7]] = -1 * pc_votes[:, [1, 4, 7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = TRANSFORMS.rotz(rot_angle)

            point_votes_end = np.zeros_like(pc_votes)
            point_votes_end[:, 1:4] = np.dot(pc[:, 0:3] + pc_votes[:, 1:4], np.transpose(rot_mat))
            point_votes_end[:, 4:7] = np.dot(pc[:, 0:3] + pc_votes[:, 4:7], np.transpose(rot_mat))
            point_votes_end[:, 7:10] = np.dot(pc[:, 0:3] + pc_votes[:, 7:10], np.transpose(rot_mat))

            pc[:, 0:3] = np.dot(pc[:, 0:3], np.transpose(rot_mat))
            bbox[:, 0:3] = np.dot(bbox[:, 0:3], np.transpose(rot_mat))
            bbox[:, 6] -= rot_angle
            pc_votes[:, 1:4] = point_votes_end[:, 1:4] - pc[:, 0:3]
            pc_votes[:, 4:7] = point_votes_end[:, 4:7] - pc[:, 0:3]
            pc_votes[:, 7:10] = point_votes_end[:, 7:10] - pc[:, 0:3]

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            pc[:, 0:3] *= scale_ratio
            bbox[:, 0:3] *= scale_ratio
            bbox[:, 3:6] *= scale_ratio
            pc_votes[:, 1:4] *= scale_ratio
            pc_votes[:, 4:7] *= scale_ratio
            pc_votes[:, 7:10] *= scale_ratio

        box3d_center = bbox[0:3]
        box3d_size = bbox[3:6]
        box_sem_cls = bbox[7]
        angle_class, angle_residual = DC.angle2class(bbox[6])
        size_class, size_residual = DC.size2class(box3d_size, DC.class2type[box_sem_cls])

        pc, choices = pc_util.random_sampling(pc, self.num_points, return_choices=True)
        pc_votes_mask = pc_votes[choices, 0]
        pc_votes = pc_votes[choices, 1:]
        cond_pc = pc_util.random_sampling(cond_pc, self.cond_obj_num_points, return_choices=False)

        ret_dict = {
            'point_clouds': pc.astype(np.float32),
            'cond_point_clouds': cond_pc.astype(np.float32),
            'center_label': box3d_center.astype(np.float32),
            'heading_class_label': angle_class,
            'heading_residual_label': angle_residual.astype(np.float32),
            'size_class_label': size_class,
            'size_residual_label': size_residual.astype(np.float32),
            'sem_cls_label': box_sem_cls.astype(np.int64),
            'vote_label': pc_votes.astype(np.float32),
            'vote_label_mask': pc_votes_mask.astype(np.int64),
            'scan_idx': np.array(idx).astype(np.int64),
            'bbox_label': bbox.astype(np.float32)
        }

        return ret_dict


if __name__ == '__main__':

    ds = ShapenetDetectionVotesDataset(
        num_points=5000,
        use_cond_votes=True,
        use_neg_votes=False, neg_votes_factor=1.0,
        use_rand_votes=True, rand_votes_factor=1.0,
        augment=False)
    batch_size = 4
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    for i, minibatch in enumerate(dl):
        for j in range(batch_size):
            scene_pc = minibatch['point_clouds'][j].cpu().numpy()
            cond_pc = minibatch['cond_point_clouds'][j].cpu().numpy()
            pc_votes = minibatch['vote_label'][j].cpu().numpy()
            pc_votes_mask = minibatch['vote_label_mask'][j].cpu().numpy()

            bbox = minibatch['bbox_label'][j].cpu().numpy()
            corners3d = get_3dcorners_from_bbox(bbox)

            plt.figure(figsize=(16, 16))
            ax = plt.axes(projection="3d")

            base_colors = mcolors.BASE_COLORS
            pc_color = base_colors['k']
            cond_pc_color = base_colors['g']
            corners3d_color = base_colors['b']
            votes_color = base_colors['r']
            pc_size = 1
            corners3d_size = 0.5
            votes_size = 0.5
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim3d(-2, 2)
            ax.set_ylim3d(-2, 2)
            ax.set_zlim3d(0, 4)

            ax = draw_pc(scene_pc, ax, pc_color, pc_size)
            ax = draw_pc(cond_pc, ax, cond_pc_color, pc_size)
            ax = draw_corners3d(corners3d, ax, corners3d_color, corners3d_size)
            ax = draw_votes(scene_pc[:, 0:3], pc_votes[:, 0:3], pc_votes_mask, ax, votes_color, votes_size)

            plt.show()


