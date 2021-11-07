import os
import sys
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from loguru import logger
from torch.utils.data import Dataset, DataLoader
from shapenet2.config import ShapenetConfig, ShapenetDatasetConfig
from shapenet2.transforms import ShapenetTransforms
from shapenet2.utils import get_bbox_2d_from_bbox_1d, random_sampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'shapenet'))
from plot_utils import draw_pc, draw_corners3d, draw_votes

class ShapenetDataset(Dataset):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(
        self, 
        scenes_name: str, 
        mode: str = "train", 
        max_num_points: int = 20000, 
        max_num_objects: int = 64,
        use_height: bool = False, 
        augment: bool = False,
        dataset_info_path: str = f"data/mean_sizes.pkl",
        num_heading_bin: int = 12
    ):

        self.data_path = os.path.join(self.BASE_DIR, f"scenes/{scenes_name}/{mode}")
        assert os.path.exists(self.data_path), f'{self.data_path} does not exist!'

        self.scenes_names = sorted(list(set([os.path.basename(x) for x in os.listdir(self.data_path)])))

        self.max_num_points = max_num_points
        self.max_num_objects = max_num_objects
        self.use_height = use_height
        self.augment = augment

        self.ds_config = ShapenetDatasetConfig(info_path=dataset_info_path, num_heading_bin=num_heading_bin)

    @property
    def num_heading_bin(self):
        return self.ds_config.num_heading_bin

    def __len__(self):
        return len(self.scenes_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (self.max_num_objects,3) for GT box center XYZ
            heading_class_label: (self.max_num_objects,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (self.max_num_objects,)
            size_class_label: (self.max_num_objects,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (self.max_num_objects,3)
            sem_cls_label: (self.max_num_objects,) semantic class index
            box_label_mask: (self.max_num_objects) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scene_name = self.scenes_names[idx]
        scene_name = os.path.join(self.data_path, scene_name)

        with open(scene_name, "rb") as f:
            data = pickle.load(f)

        # slice to max_num_objects if scene has more objects
        n_objects = len(data['obj_pc'])
        max_indices_to_rand = min(self.max_num_objects, n_objects)
        indices = np.random.choice(n_objects, max_indices_to_rand).tolist()
        data = {k: [v[i] for i in indices] for k, v in data.items()}

        cond_pc = random.choice(data['obj_pc'])
        pc = np.concatenate(data['scene_pc'], axis=0)
        bboxes = np.asarray(data['scene_bbox'])

        # pc_votes = []
        # for obj in data['scene_pc']:
        #     obj_center = np.mean(obj, axis=0)
        #     # repeats = np.concatenate([[1], np.repeat(obj_center, 3, axis=0)])
        #     repeats = np.concatenate([[0], np.repeat(obj_center, 3, axis=0)])
        #     center_repeats = np.repeat(repeats[None, ...], obj.shape[0], axis=0)
        #     # obj_repeats = np.concatenate([np.zeros((obj.shape[0], 1)), np.repeat(obj, 3, axis=1)], axis=1)
        #     obj_repeats = np.concatenate([np.ones((obj.shape[0], 1)), np.repeat(obj, 3, axis=1)], axis=1)
        #     # pc_votes.append(center_repeats - obj_repeats)
        #     pc_votes.append(obj_repeats - center_repeats)
        # pc_votes = np.concatenate(pc_votes, axis=0)

        run_ind = 0
        pc_votes = np.zeros((pc.shape[0], 10))
        for i in range(n_objects):
            obj_pc = np.asarray(data['scene_pc'][i])
            obj_center = np.mean(obj_pc, axis=0, keepdims=True)
            obj_n_pts = obj_pc.shape[0]
            pc_votes[run_ind:run_ind + obj_n_pts, 0] = 1
            pc_votes[run_ind:run_ind + obj_n_pts, 1:4] = np.repeat(obj_center, repeats=obj_n_pts, axis=0) - obj_pc
            pc_votes[run_ind:run_ind + obj_n_pts, 4:7] = np.repeat(obj_center, repeats=obj_n_pts, axis=0) - obj_pc
            pc_votes[run_ind:run_ind + obj_n_pts, 7:10] = np.repeat(obj_center, repeats=obj_n_pts, axis=0) - obj_pc
            run_ind += obj_n_pts

        if self.use_height:
            floor_height = np.percentile(pc[:, 2], 0.99)
            heights_wrt_floor = pc[:, 2] - floor_height
            pc = np.concatenate([pc, heights_wrt_floor[..., None]], axis=1)  # (n, 4)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                pc[:, 0] = -1 * pc[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]
                pc_votes[:, [1, 4, 7]] = -1 * pc_votes[:, [1, 4, 7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6  # -30 ~ +30 degree
            rot_mat = ShapenetTransforms.rotz(rot_angle)

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
            scale_ratio = np.random.random()*0.3+0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            pc[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio
            pc_votes[:, 1:4] *= scale_ratio
            pc_votes[:, 4:7] *= scale_ratio
            pc_votes[:, 7:10] *= scale_ratio
            if self.use_height:
                pc[:, -1] *= scale_ratio[0, 0]

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((self.max_num_objects, 3))
        box3d_sizes = np.zeros((self.max_num_objects, 3))

        angle_classes = np.zeros((self.max_num_objects,))
        angle_residuals = np.zeros((self.max_num_objects,))

        size_classes = np.zeros((self.max_num_objects,))
        size_residuals = np.zeros((self.max_num_objects, 3))

        label_mask = np.zeros(self.max_num_objects)
        label_mask[:n_objects] = 1

        semantic_classes = np.array([ShapenetConfig.class_name_to_class_id[x] for x in data["obj_class"]])
        max_bboxes = np.zeros((self.max_num_objects, 8))
        max_bboxes[:n_objects, :7] = bboxes
        max_bboxes[:n_objects, -1] = semantic_classes

        for i in range(n_objects):
            bbox = bboxes[i]
            semantic_class = semantic_classes[i]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = self.ds_config.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6] * 2
            size_class, size_residual = self.ds_config.size2class(box3d_size, self.ds_config.cls_to_type[semantic_class])
            box3d_centers[i, :] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i, :] = box3d_size

        pc, choices = random_sampling(pc, self.max_num_points, return_choices=True)
        point_votes = pc_votes[choices, 1:]
        point_votes_mask = pc_votes[choices, 0]

        ret_dict = {}
        ret_dict['point_clouds'] = pc.astype(np.float32)
        ret_dict['cond_point_clouds'] = cond_pc.astype(np.float32)
        ret_dict['center_label'] = max_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros(self.max_num_objects)
        target_bboxes_semcls[:n_objects] = bboxes[:, -1]  # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = label_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes
        return ret_dict


if __name__ == "__main__":

    ds_train = ShapenetDataset(scenes_name="dev", mode="train", use_height=False, augment=False)
    dl_train = DataLoader(ds_train, batch_size=4, shuffle=False, num_workers=0)

    ds_val = ShapenetDataset(scenes_name="dev", mode="val", use_height=True, augment=True)
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0)

    logger.info(f"train dataloader: {len(dl_train)} batches, total of {len(ds_train)} samples.")
    logger.info(f"val dataloader: {len(dl_val)} batches, total of {len(ds_val)} samples.")

    for i, minibatch in enumerate(dl_train):
        for j in range(4):
            scene_pc = minibatch['point_clouds'][j].cpu().numpy()
            bboxes = minibatch['max_gt_bboxes'][j].cpu().numpy()
            bboxes_mask = minibatch['box_label_mask'][j].cpu().numpy()
            cond_pc = minibatch['cond_point_clouds'][j].cpu().numpy()
            pc_votes = minibatch['vote_label'][j].cpu().numpy()
            pc_votes_mask = minibatch['vote_label_mask'][j].cpu().numpy()

            n_obj = int(np.sum(bboxes_mask).item())
            corners3d = np.zeros((bboxes.shape[0], 8, 3))
            for k in range(n_obj):
                bbox = bboxes[k]
                corner3d = get_bbox_2d_from_bbox_1d(bbox)
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
            for k in range(n_obj):
                corner3d = corners3d[k]
                ax = draw_corners3d(corner3d, ax, corners3d_color, corners3d_size)
            ax = draw_votes(scene_pc[:, 0:3], pc_votes[:, 0:3], pc_votes_mask, ax, votes_color, votes_size)

            plt.show()
