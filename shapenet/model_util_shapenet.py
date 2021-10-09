import numpy as np
import sys
import os
import json
from config import ShapenetConfig
from utils import get_pc_measures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

shapenet_config = ShapenetConfig()


class ShapenetDatasetConfig(object):
    def __init__(self):

        self.data_path = os.path.join(BASE_DIR, 'scenes', 'train')
        assert os.path.exists(self.data_path), f'{self.data_path} does not exist!'

        self.num_class = 10
        self.num_heading_bin = 12
        self.num_size_cluster = 10

        # self.type2class = {
        #     'airplane': shapenet_config.classname2class['airplane'],
        #     'car': shapenet_config.classname2class['car'],
        #     'chair': shapenet_config.classname2class['chair'],
        #     'guitar': shapenet_config.classname2class['guitar'],
        #     'knife': shapenet_config.classname2class['knife'],
        #     'lamp': shapenet_config.classname2class['lamp'],
        #     'laptop': shapenet_config.classname2class['laptop'],
        #     'motorbike': shapenet_config.classname2class['motorbike'],
        #     'pistol': shapenet_config.classname2class['pistol'],
        #     'table': shapenet_config.classname2class['table']
        # }
        self.type2class = {
            'airplane': 0,
            'car': 1,
            'chair': 2,
            'guitar': 3,
            'knife': 4,
            'lamp': 5,
            'laptop': 6,
            'motorbike': 7,
            'pistol': 8,
            'table': 9
        }

        self.class2type = {self.type2class[t]: t for t in self.type2class}

        # self.type_mean_size = self.get_type_mean_size()
        self.type_mean_size = {
            'table': np.asarray([0.48582524, 0.75855831, 0.37933258]),
            'car': np.asarray([0.89204091, 0.35594545, 0.24248318]),
            'chair': np.asarray([0.45563932, 0.44019648, 0.7551967]),
            'airplane': np.asarray([0.69049206, 0.67386444, 0.19565079]),
            'guitar': np.asarray([0.05837808, 0.32863962, 0.93829038]),
            'laptop': np.asarray([0.56957, 0.67356429, 0.46480143]),
            'pistol': np.asarray([0.84357467, 0.11722067, 0.50951267]),
            'lamp': np.asarray([0.48568316, 0.33136974, 0.74998763]),
            'motorbike': np.asarray([0.86296571, 0.24997286, 0.40955857]),
            'knife': np.asarray([0.192978, 0.078202, 0.963034])
        }

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]

    def get_type_mean_size(self):
        type_mean_size = {}
        for filename in os.listdir(self.data_path):
            print(os.path.join(self.data_path, filename))
            file = os.path.join(self.data_path, filename)
            with open(file, "r") as f:
                data = json.load(f)
            n_obj = len(data['object_pc'])
            for i in range(n_obj):
                _, _, _, _, _, _, _, _, _, sizex, sizey, sizez = get_pc_measures(np.asarray(data['object_pc'][i]))
                obj_cls_ind = data['scene_bbox'][i][7]
                obj_cls = self.class2type[obj_cls_ind]
                if obj_cls in type_mean_size:
                    type_mean_size[obj_cls].append([sizex, sizey, sizez])
                else:
                    type_mean_size[obj_cls] = [[sizex, sizey, sizez]]

        for k, v in type_mean_size.items():
            type_mean_size[k] = np.mean(np.asarray(v), axis=0)

        return type_mean_size

    def size2class(self, size, type_name):
        """ Convert 3D box size (l,w,h) to size class and size residual """
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        """ Inverse function to size2class """
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        """ Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        """
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (0 <= angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """ Inverse function to angle2class """
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
