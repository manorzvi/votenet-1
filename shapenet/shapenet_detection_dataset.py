import os
import random
import sys
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from shapenet_transforms import ShapenetTransforms
from model_util_shapenet import

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# DC = ShapenetDatasetConfig() # dataset specific config
shapenet_transforms = ShapenetTransforms()
MAX_NUM_OBJ = 64  # maximum number of objects allowed per scene


class ShapenetDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000, use_height=False, augment=False):

        self.data_path = os.path.join(BASE_DIR, 'scenes', split_set)
        assert os.path.exists(self.data_path), f'{self.data_path} does not exist!'

        self.scan_names = sorted(list(set([os.path.basename(x) for x in os.listdir(self.data_path)])))

        self.num_points = num_points
        self.augment = augment
        self.use_height = use_height
       
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
        cond_obj_ind = random.randint(0, n_obj-1)

        cond_pc = data['object_pc'][cond_obj_ind]
        cond_pc = np.asarray(cond_pc)

        pc = data['scene_pc']
        pc = [np.asarray(x) for x in pc]
        pc = np.concatenate(pc, axis=0)

        bboxes = data['scene_bbox']
        bboxes = np.asarray(bboxes)

        n_pts = pc.shape[0]
        pc_votes = np.zeros((n_pts, 10))
        pc_votes[:, 9] = 1

        run_ind = 0
        for i in range(n_obj):
            obj_center = np.mean(np.asarray(data['scene_pc'][i]), axis=0, keepdims=True)
            obj_n_pts = len(data['scene_pc'][i])
            pc_votes[run_ind:run_ind + obj_n_pts, 0:3] = np.repeat(obj_center, obj_n_pts, axis=0)
            pc_votes[run_ind:run_ind + obj_n_pts, 3:6] = np.repeat(obj_center, obj_n_pts, axis=0)
            pc_votes[run_ind:run_ind + obj_n_pts, 6:9] = np.repeat(obj_center, obj_n_pts, axis=0)
            run_ind += obj_n_pts

        if self.use_height:
            floor_height = np.percentile(pc[:, 2], 0.99)
            height = pc[:, 2] - floor_height
            pc = np.concatenate([pc, np.expand_dims(height, 1)], 1)  # (N,4)

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
            rot_mat = shapenet_transforms.rotz(rot_angle)

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
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros(MAX_NUM_OBJ)
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0], :] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            box3d_size = bbox[3:6]*2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size

        target_bboxes_mask = label_mask 
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes
        return ret_dict


if __name__=='__main__':

    ds = ShapenetDetectionVotesDataset(use_height=True, augment=True)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    for i, minibatch in enumerate(dl):
        print(i, minibatch)
