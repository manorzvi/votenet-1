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
from cond_voting_module import CondVotingModule
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
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 sampling='vote_fps', use_two_backbones: bool = False):
        super().__init__()
        print('[I] - use cond_votenet!')
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(0)

        self.use_two_backbones = use_two_backbones
        if use_two_backbones:
            print('[I] - build another backbone!')
            self.cond_backbone_net = Pointnet2Backbone(0)

        # Hough voting
        self.vgen = CondVotingModule(512)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, 1, sampling)

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

        #end_points['fp2_features'] = end_points['fp2_features'] + cond_end_points['fp2_features']
        end_points['fp2_features'] = torch.cat((end_points['fp2_features'], cond_end_points['fp2_features']), dim=1)

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
