# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')

parser.add_argument('--use_cond_votes', action='store_true', default=False,
                    help='Use conditional votes label (only points associated with the conditioned object vote)')

parser.add_argument('--use_neg_votes', action='store_true', default=False,
                    help='All points can vote, but only the points associated with the conditioned object vote to the center.'
                         'The rest vote to the negative direction.')
parser.add_argument('--neg_votes_factor', type=float, default=1.0)

parser.add_argument('--use_rand_votes', action='store_true', default=False,
                    help='All points can vote, but only the points associated with the conditioned object vote to the center.'
                         'The rest vote to a random point.')
parser.add_argument('--rand_votes_factor', type=float, default=1.0)

parser.add_argument('--use_two_backbones', action='store_true', default=False,
                    help='Use another pointnet++ backbone net for the conditional point cloud.')

parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
FLAGS = parser.parse_args()

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

assert not (FLAGS.use_rand_votes and not FLAGS.use_cond_votes), "If use_rand_votes=True, use_cond_votes=True is a must!"
assert not (FLAGS.model == 'cond_votenet' and FLAGS.dataset != 'shapenet'), "If model=cond_votenet, dataset=shapenet is a must!"

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert(CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if FLAGS.dataset == 'shapenet':
    sys.path.append(os.path.join(ROOT_DIR, 'shapenet'))
    from shapenet_detection_dataset import ShapenetDetectionVotesDataset
    from model_util_shapenet import ShapenetDatasetConfig

    DATASET_CONFIG = ShapenetDatasetConfig()
    TEST_DATASET = ShapenetDetectionVotesDataset(
        'val', num_points=NUM_POINT,
        augment=False,
        use_cond_votes=FLAGS.use_cond_votes,
        use_rand_votes=FLAGS.use_rand_votes, rand_votes_factor=FLAGS.rand_votes_factor,
        use_neg_votes=FLAGS.use_neg_votes, neg_votes_factor=FLAGS.neg_votes_factor,
    )
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)

print(f'|TEST_DATASET|={len(TEST_DATASET)}')

TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=FLAGS.shuffle_dataset, num_workers=4, worker_init_fn=my_worker_init_fn)

print(f'|TEST_DATALOADER|={len(TEST_DATALOADER)}')

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
elif FLAGS.model == 'cond_votenet':
    Detector = MODEL.CondVoteNet
else:
    Detector = MODEL.VoteNet

if FLAGS.model == 'cond_votenet':
    net = Detector(
        num_class=DATASET_CONFIG.num_class,
        num_heading_bin=DATASET_CONFIG.num_heading_bin,
        num_size_cluster=DATASET_CONFIG.num_size_cluster,
        mean_size_arr=DATASET_CONFIG.mean_size_arr,
        num_proposal=FLAGS.num_target,
        input_feature_dim=0,
        vote_factor=FLAGS.vote_factor,
        sampling=FLAGS.cluster_sampling,
        use_two_backbones=FLAGS.use_two_backbones,
    )
else:
    net = Detector(
        num_class=DATASET_CONFIG.num_class,
        num_heading_bin=DATASET_CONFIG.num_heading_bin,
        num_size_cluster=DATASET_CONFIG.num_size_cluster,
        mean_size_arr=DATASET_CONFIG.mean_size_arr,
        num_proposal=FLAGS.num_target,
        input_feature_dim=num_input_channel,
        vote_factor=FLAGS.vote_factor,
        sampling=FLAGS.cluster_sampling
    )

net.to(device)

criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = {
    'remove_empty_box': (not FLAGS.faster_eval),
    'use_3d_nms': FLAGS.use_3d_nms,
    'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms,
    'cls_nms': FLAGS.use_cls_nms,
    'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh,
    'dataset_config': DATASET_CONFIG
}


def evaluate_one_epoch():
    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        if FLAGS.dataset == 'shapenet':
            inputs = {
                'point_clouds': batch_data_label['point_clouds'],
                'cond_point_clouds': batch_data_label['cond_point_clouds']
            }
        else:
            inputs = {
                'point_clouds': batch_data_label['point_clouds'],
            }

        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    
        # Dump evaluation results for visualization
        if batch_idx == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__=='__main__':
    eval()
