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
MAX_NUM_OBJ = 64  # maximum number of objects allowed per scene


class ShapenetDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000, cond_obj_num_points=2000,
                 use_cond_votes=False,
                 use_neg_votes=False, neg_votes_factor=1.0,
                 use_rand_votes=False, rand_votes_factor=1.0,
                 use_cond_bboxs=False, augment=False):

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

        self.use_cond_bboxs = use_cond_bboxs
        if use_cond_bboxs:
            print(f'[I] - shapenet {split_set} dataset with cond_bboxs!')

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

        bboxes = data['scene_bbox']
        bboxes = np.asarray(bboxes)

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
                    # pc_votes[run_ind:run_ind + obj_n_pts, 4:7] = -pc_votes[run_ind:run_ind + obj_n_pts, 4:7]
                    # pc_votes[run_ind:run_ind + obj_n_pts, 7:10] = -pc_votes[run_ind:run_ind + obj_n_pts, 7:10]

            run_ind += obj_n_pts

        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                pc[:, 0] = -1 * pc[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]
                pc_votes[:, [1, 4, 7]] = -1 * pc_votes[:, [1, 4, 7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = TRANSFORMS.rotz(rot_angle)

            point_votes_end = np.zeros_like(pc_votes)
            point_votes_end[:, 1:4] = np.dot(pc[:, 0:3] + pc_votes[:, 1:4], np.transpose(rot_mat))
            point_votes_end[:, 4:7] = np.dot(pc[:, 0:3] + pc_votes[:, 4:7], np.transpose(rot_mat))
            point_votes_end[:, 7:10] = np.dot(pc[:, 0:3] + pc_votes[:, 7:10], np.transpose(rot_mat))

            pc[:, 0:3] = np.dot(pc[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle
            pc_votes[:, 1:4] = point_votes_end[:, 1:4] - pc[:, 0:3]
            pc_votes[:, 4:7] = point_votes_end[:, 4:7] - pc[:, 0:3]
            pc_votes[:, 7:10] = point_votes_end[:, 7:10] - pc[:, 0:3]

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            pc[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio
            pc_votes[:, 1:4] *= scale_ratio
            pc_votes[:, 4:7] *= scale_ratio
            pc_votes[:, 7:10] *= scale_ratio

        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros(MAX_NUM_OBJ)
        if self.use_cond_bboxs:
            label_mask[cond_obj_ind] = 1
        else:
            label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0], :] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            box3d_size = bbox[3:6]
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i, :] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i, :] = box3d_size

        pc, choices = pc_util.random_sampling(pc, self.num_points, return_choices=True)
        pc_votes_mask = pc_votes[choices, 0]
        pc_votes = pc_votes[choices, 1:]
        cond_pc = pc_util.random_sampling(cond_pc, self.cond_obj_num_points, return_choices=False)

        ret_dict = {}
        ret_dict['point_clouds'] = pc.astype(np.float32)
        ret_dict['cond_point_clouds'] = cond_pc.astype(np.float32)
        ret_dict['center_label'] = max_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros(MAX_NUM_OBJ)
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = label_mask.astype(np.float32)
        ret_dict['vote_label'] = pc_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = pc_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes
        return ret_dict


if __name__ == '__main__':

    ds = ShapenetDetectionVotesDataset(
        num_points=5000,
        use_cond_votes=True,
        use_neg_votes=True, neg_votes_factor=10.0,
        use_rand_votes=False, rand_votes_factor=2.0,
        use_cond_bboxs=True,
        augment=False)
    batch_size = 4
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    for i, minibatch in enumerate(dl):
        for j in range(batch_size):
            scene_pc = minibatch['point_clouds'][j].cpu().numpy()
            bboxes = minibatch['max_gt_bboxes'][j].cpu().numpy()
            bboxes_mask = minibatch['box_label_mask'][j].cpu().numpy()
            cond_pc = minibatch['cond_point_clouds'][j].cpu().numpy()
            pc_votes = minibatch['vote_label'][j].cpu().numpy()
            pc_votes_mask = minibatch['vote_label_mask'][j].cpu().numpy()

            corners3d = np.zeros((bboxes.shape[0], 8, 3))
            for k in range(bboxes_mask.shape[0]):
                if bboxes_mask[k] == 0:
                    print(f'[I] - skipping object bounding box {k}')
                    continue
                bbox = bboxes[k]
                corner3d = get_3dcorners_from_bbox(bbox)
                corners3d[k, :, :] = corner3d

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
            for k in range(bboxes_mask.shape[0]):
                if bboxes_mask[k] == 0:
                    print(f'[I] - skipping object bounding box {k}')
                    continue
                corner3d = corners3d[k]
                ax = draw_corners3d(corner3d, ax, corners3d_color, corners3d_size)
            ax = draw_votes(scene_pc[:, 0:3], pc_votes[:, 0:3], pc_votes_mask, ax, votes_color, votes_size)

            plt.show()


