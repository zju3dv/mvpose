from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import sys
import os.path as osp

import numpy as np
import torch
from torch.backends import cudnn
from torch.autograd import Variable
from torch import nn

from .reid.utils.serialization import load_checkpoint
from .reid.utils import to_torch
from .reid import models


def extract_cnn_feature(model, inputs, output_feature=None, use_gpu=True):
    model.eval ()
    inputs = to_torch ( inputs )
    inputs = Variable ( inputs )
    # Hack to get quick eval
    if use_gpu:
        if not inputs.is_cuda:
            inputs = inputs.cuda ()
    outputs = model ( inputs.squeeze (), output_feature )
    outputs = outputs.data.cpu ()
    return outputs


def extract_features(model, data_batch, print_freq=1, output_feature=None, use_gpu=True):
    model.eval ()

    features = OrderedDict ()
    imgs, fnames, pids, cam_id = data_batch
    # data_time.update ( time.time () - end )
    outputs = extract_cnn_feature ( model, imgs, output_feature, use_gpu=use_gpu )
    for fname, output, pid in zip ( fnames, outputs, pids ):
        # if isinstance ( fname, list ):
        #     fname = fname[0]
        features[fname] = output
    return features


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    # import pdb; pdb.set_trace()
    x = torch.cat ( [query_features[f].unsqueeze ( 0 ) for f, _, _ in query], 0 )
    y = torch.cat ( [gallery_features[f].unsqueeze ( 0 ) for f, _, _ in gallery], 0 )
    m, n = x.size ( 0 ), y.size ( 0 )
    x = x.view ( m, -1 )
    y = y.view ( n, -1 )
    dist = torch.pow ( x, 2 ).sum ( dim=1, keepdim=True ).expand ( m, n ) + \
           torch.pow ( y, 2 ).sum ( dim=1, keepdim=True ).expand ( n, m ).t ()
    dist.addmm_ ( 1, -2, x, y.t () )
    return dist


def pairwise_affinity(query_features, gallery_features, query=None, gallery=None):
    # import pdb; pdb.set_trace()
    x = torch.cat ( [query_features[f].unsqueeze ( 0 ) for f, _, _ in query], 0 )
    y = torch.cat ( [gallery_features[f].unsqueeze ( 0 ) for f, _, _ in gallery], 0 )
    m, n = x.size ( 0 ), y.size ( 0 )
    x = x.view ( m, -1 )
    y = y.view ( n, -1 )
    dist = torch.pow ( x, 2 ).sum ( dim=1, keepdim=True ).expand ( m, n ) + \
           torch.pow ( y, 2 ).sum ( dim=1, keepdim=True ).expand ( n, m ).t ()
    dist.addmm_ ( 1, -2, x, y.t () )
    normalized_affinity = - (dist - dist.mean ()) / dist.std ()
    affinity = torch.sigmoid ( normalized_affinity * torch.tensor ( 5. ) )  # x5 to match 1->1
    # pro = x @ y.t ()
    # norms = x.norm ( dim=1 ).unsqueeze ( 1 ) @ y.norm ( dim=1 ).unsqueeze ( 0 )
    # affinity = (pro / norms + 1) / 2  # map from (-1, 1) to (0, 1)
    # affinity = torch.sigmoid ( pro / norms ) #  map to (0, 1)
    return affinity


def reranking(query_features, gallery_features, query=None, gallery=None, k1=20, k2=6, lamda_value=0.3):
    x = torch.cat ( [query_features[f].unsqueeze ( 0 ) for f, _, _ in query], 0 )
    y = torch.cat ( [gallery_features[f].unsqueeze ( 0 ) for f, _, _ in gallery], 0 )
    feat = torch.cat ( (x, y) )
    query_num, all_num = x.size ( 0 ), feat.size ( 0 )
    feat = feat.view ( all_num, -1 )

    dist = torch.pow ( feat, 2 ).sum ( dim=1, keepdim=True ).expand ( all_num, all_num )
    dist = dist + dist.t ()
    dist.addmm_ ( 1, -2, feat, feat.t () )

    original_dist = dist.numpy ()
    all_num = original_dist.shape[0]
    original_dist = np.transpose ( original_dist / np.max ( original_dist, axis=0 ) )
    V = np.zeros_like ( original_dist ).astype ( np.float16 )
    initial_rank = np.argsort ( original_dist ).astype ( np.int32 )

    # print ( 'starting re_ranking' )
    for i in range ( all_num ):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where ( backward_k_neigh_index == i )[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range ( len ( k_reciprocal_index ) ):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int ( np.around ( k1 / 2 ) ) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int ( np.around ( k1 / 2 ) ) + 1]
            fi_candidate = np.where ( candidate_backward_k_neigh_index == candidate )[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len ( np.intersect1d ( candidate_k_reciprocal_index, k_reciprocal_index ) ) > 2 / 3 * len (
                    candidate_k_reciprocal_index ):
                k_reciprocal_expansion_index = np.append ( k_reciprocal_expansion_index, candidate_k_reciprocal_index )

        k_reciprocal_expansion_index = np.unique ( k_reciprocal_expansion_index )
        weight = np.exp ( -original_dist[i, k_reciprocal_expansion_index] )
        V[i, k_reciprocal_expansion_index] = weight / np.sum ( weight )
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like ( V, dtype=np.float16 )
        for i in range ( all_num ):
            V_qe[i, :] = np.mean ( V[initial_rank[i, :k2], :], axis=0 )
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range ( all_num ):
        invIndex.append ( np.where ( V[:, i] != 0 )[0] )

    jaccard_dist = np.zeros_like ( original_dist, dtype=np.float16 )

    for i in range ( query_num ):
        temp_min = np.zeros ( shape=[1, all_num], dtype=np.float16 )
        indNonZero = np.where ( V[i, :] != 0 )[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range ( len ( indNonZero ) ):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum ( V[i, indNonZero[j]],
                                                                                 V[indImages[j], indNonZero[j]] )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lamda_value) + original_dist * lamda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


class FeatureExtractor ( object ):
    def __init__(self, model=None):
        super ( FeatureExtractor, self ).__init__ ()
        if model:
            self.model = model
        else:
            # get model
            self.args = dict ( arch='resnet50', batch_size=128, camstyle=46,
                               data_dir='/home/jiangwen/reid/CamStyle/data/',
                               dataset='market', dist_metric='euclidean', dropout=0.5, epochs=50, evaluate=False,
                               features=1024,
                               height=256, logs_dir='logs/market-ide-camstyle-re', lr=0.1, momentum=0.9,
                               output_feature='pool5',
                               print_freq=1, re=0.5, rerank=True, weight_decay=0.0005, width=128, workers=8,
                               resume='logs/market-ide-camstyle-re/checkpoint.pth.tar' )
            self.args['resume'] = osp.join ( osp.dirname ( __file__ ), self.args['resume'] )
            cudnn.benchmark = True
            # Create model
            model = models.create ( self.args['arch'], num_features=self.args['features'],
                                    dropout=self.args['dropout'],
                                    num_classes=751 )  # 751 is where the number of classes at market 1501
            # Load from checkpoint
            checkpoint = load_checkpoint ( self.args['resume'] )
            model.load_state_dict ( checkpoint['state_dict'] )
            start_epoch = checkpoint['epoch']
            print ( "=> Start epoch {} "
                    .format ( start_epoch ) )
            self.model = nn.DataParallel ( model ).cuda ()

    def get_dismat(self, query_batch, output_feature='pool5', rerank=False, use_gpu=True):
        """
        Will compute matrix with itself
        :param query_batch: An iterable object with (this_imgs, fnames, pids, _)
        :param output_feature:'pool5'
        :param rerank: boolean
        :param use_gpu: boolean
        :return:
        """
        query = list ( zip ( *query_batch[1:] ) )
        # Adapt to use dataloader
        # query = list ( map ( lambda x: [i[0] for i in x], query ) )
        query_features = extract_features ( self.model, query_batch, 1, output_feature, use_gpu=use_gpu )
        if rerank:
            distmat = reranking ( query_features, query_features.copy (), query, query )
        else:
            distmat = pairwise_distance ( query_features, query_features.copy (), query, query )
        return distmat

    def get_affinity(self, query_batch, output_feature='pool5', rerank=False, use_gpu=True):
        """
        Will compute matrix with itself
        :param query_batch: An iterable object with (this_imgs, fnames, pids, _)
        :param output_feature:'pool5'
        :param rerank: boolean
        :param use_gpu: boolean
        :return:
        """
        query = list ( zip ( *query_batch[1:] ) )
        # Adapt to use dataloader
        # query = list ( map ( lambda x: [i[0] for i in x], query ) )
        query_features = extract_features ( self.model, query_batch, 1, output_feature, use_gpu=use_gpu )
        if rerank:
            affinity = reranking ( query_features, query_features.copy (), query, query )
        else:
            affinity = pairwise_affinity ( query_features, query_features.copy (), query, query )
        return affinity
