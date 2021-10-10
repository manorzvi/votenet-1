import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss


class CondVoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self,
                 num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps',
                 use_two_backbones: bool = False, use_learnable_cond: bool = False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        self.use_two_backbones = use_two_backbones
        if use_two_backbones:
            print('[I] - build another backbone!')
            self.cond_backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        self.use_learnable_cond = use_learnable_cond
        if use_learnable_cond:
            print('[I] - use learnable cond signal!')
            self.cond_conv1 = torch.nn.Conv1d(256, 256, 1)
            self.cond_conv2 = torch.nn.Conv1d(256, 256, 1)
            self.cond_conv3 = torch.nn.Conv1d(256, 256, 1)
            self.cond_bn1 = torch.nn.BatchNorm1d(256)
            self.cond_bn2 = torch.nn.BatchNorm1d(256)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}

        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        cond_end_points = {}
        if self.use_two_backbones:
            cond_end_points = self.cond_backbone_net(inputs['cond_point_clouds'], cond_end_points)
        else:
            cond_end_points = self.backbone_net(inputs['cond_point_clouds'], cond_end_points)

        if not self.use_learnable_cond:
            end_points['fp2_features'] = end_points['fp2_features'] + cond_end_points['fp2_features']

        if self.use_learnable_cond:
            cond_fp2_features = F.relu(self.cond_bn1(self.cond_conv1(cond_end_points['fp2_features'])))
            cond_fp2_features = F.relu(self.cond_bn2(self.cond_conv2(cond_fp2_features)))
            cond_fp2_features = self.cond_conv3(cond_fp2_features)
            end_points['fp2_features'] = end_points['fp2_features'] + cond_fp2_features
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)

        return end_points