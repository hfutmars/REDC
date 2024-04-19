# -*- coding: utf-8 -*-
"""
修改模块：
1. 预测层，使用逐元素相乘替代拼接；

"""


import argparse
from abc import ABC

import torch
from torch.nn import init
import torch.nn.functional as F

from data_same_node import Data
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from util import *


def config():
    parser = argparse.ArgumentParser(description='RGC')
    parser.add_argument('--device', default='0', type=int, help='gpu.')
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--review_feat_size', type=int, default=64)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--train_max_iter', type=int, default=1000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--train_classification', type=bool, default=True)

    args = parser.parse_args()
    args.model_short_name = 'RGC'

    args.dataset_name = 'Digital_Music_5'
    args.dataset_path = '../data/Digital_Music_5/Digital_Music_5.json'
    args.review_feat_size = 64
    # dropout确实还挺有用的
    args.gcn_dropout = 0.7
    args.device = torch.device(args.device)
    args.train_max_iter = 1000

    args.gcn_agg_units = args.review_feat_size
    args.gcn_out_units = args.review_feat_size

    return args


class GCMCGraphConv(nn.Module, ABC):

    def __init__(self,
                 user_num,
                 item_num,
                 out_feats, # embedding size
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dropout = nn.Dropout(dropout_rate)

        # 设置需要更新的变量
        self.prob_score = nn.Linear(out_feats, 1, bias=False)
        # self.review_score = nn.Linear(out_feats, 1, bias=False)
        self.review_w = nn.Linear(out_feats, out_feats, bias=False)

        self.weight = nn.Parameter(torch.Tensor(self.user_num + self.item_num, out_feats))

        self.reset_parameters()

    def reset_parameters(self): # 初始化该模块的变量的参数
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.prob_score.weight)
        # init.xavier_uniform_(self.review_score.weight)
        init.xavier_uniform_(self.review_w.weight)
        # init.xavier_uniform_(self.review_w_.weight)

    # def edge_attention(self, edges):
    #     h = torch.cat([edges.src['h'], edges.dst['h'], self.review_w(edges.data['review_feat'])], 1)
    #     # print(h.shape)
    #     a = self.atten_fc(h)
    #
    #     # r = self.review_w(edges['review_feat'])
    #     # h = torch.sum(edges.src['h'] * edges.dst['h'], dim=1).reshape(-1, 1)
    #
    #     return {'e': F.leaky_relu(a)}
    #
    # def message_func(self, edges):
    #     return {'h': (edges.src['h'] + edges.data['r']) * self.dropout(edges.src['ci'])}
    #
    # def reduce_fuc(self, nodes):
    #     # alpha = F.softmax(nodes.mailbox['e'], dim=1)
    #     h = torch.sum(nodes.mailbox['h'], dim=1)
    #     return {'h': h}
    #
    # def forward(self, graph, x):
    #     feat = self.weight
    #     graph.ndata['h'] = feat
    #     graph.edata['r'] = self.review_w_(graph.edata['review_feat'])
    #     # graph.apply_edges(self.edge_attention)
    #     graph.update_all(self.message_func, self.reduce_fuc)
    #     graph.ndata['h'] = graph.ndata['h'] * graph.dstdata['ci']
    #     return graph.ndata.pop('h')

    def forward(self, graph, x):

        # if x[0] is not None:
        #     feat = x[0]
        # else:
        #     feat = self.weight
        graph.srcdata['h'] = x[0]
        graph.srcdata['h_r'] = self.weight
        review_feat = graph.edata['review_feat']
        graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))
        # graph.edata['ra'] = torch.sigmoid(self.review_score(review_feat))
        graph.edata['rf'] = self.review_w(review_feat)

        # graph.update_all(lambda edges: {'m': (edges.src['h'] * edges.data['pa']
        #                                       + edges.data['rf'] * edges.data['ra'])
        #                                      * self.dropout(edges.src['ci'])},
        #                  fn.sum(msg='m', out='h'))

        graph.update_all(lambda edges: {'m': (edges.src['h']
                                              + edges.src['h_r']
                                              + edges.data['rf'])
                                             * self.dropout(edges.src['ci'])},
                         fn.sum(msg='m', out='h'))

        rst = graph.dstdata['h']
        rst = rst * graph.dstdata['ci']

        return rst

class GCMCLayer(nn.Module, ABC):

    def __init__(self,
                 rating_vals,
                 user_in_units,
                 item_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.num_user = user_in_units
        self.num_item = item_in_units
        self.rating_vals = rating_vals
        self.fc_user = nn.Linear(msg_units, msg_units)
        self.fc_item = nn.Linear(msg_units, out_units)
        self.dropout = nn.Dropout(dropout_rate)
        sub_conv = {}
        for rating in rating_vals:
            rating = str(rating)
            self.W_r = None
            sub_conv[rating] = GCMCGraphConv(user_in_units,
                                             item_in_units,
                                             msg_units,
                                             device=device,
                                             dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate='sum')
        self.agg_act = nn.LeakyReLU(0.1)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, in_feats):

        # in_feats = {'node': None}

        out_feats = self.conv(graph, in_feats)

        feat = out_feats['node']
        u_feat, i_feat = torch.split(feat, [self.num_user, self.num_item], dim=0)

        # fc and non-linear
        # u_feat = self.agg_act(u_feat)
        u_feat = self.dropout(u_feat)
        u_feat = self.fc_user(u_feat)

        # i_feat = self.agg_act(i_feat)
        i_feat = self.dropout(i_feat)
        i_feat = self.fc_item(i_feat)

        feat = torch.cat([u_feat, i_feat], dim=0)

        return feat


class MLPPredictor(nn.Module, ABC):

    def __init__(self,
                 in_units,
                 src_in_units,
                 dst_in_units,
                 num_classes,
                 dropout_rate=0.0):
        super(MLPPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.predictor = nn.Sequential(
            nn.Linear(in_units, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, num_classes, bias=False),
        )
        self.b = nn.Parameter(torch.zeros((src_in_units + dst_in_units, 1)))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def apply_edges(self, edges):

        h_u = edges.src['h']
        h_v = edges.dst['h']
        # b_u = edges.src['b']
        # b_i = edges.dst['b']

        score = self.predictor(h_u * h_v)

        return {'score': score}


    def forward(self, graph, feat):
        graph.nodes['node'].data['h'] = feat

        graph.nodes['node'].data['b'] = self.b

        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Net(nn.Module, ABC):
    def __init__(self, params):
        super(Net, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(params.src_in_units + params.dst_in_units, params.gcn_out_units))

        self.encoder = GCMCLayer(params.rating_vals,
                                 params.src_in_units,
                                 params.dst_in_units,
                                 params.gcn_agg_units,
                                 params.gcn_out_units,
                                 dropout_rate=params.gcn_dropout,
                                 device=params.device)
        # self.encoder_2 = GCMCLayer(params.rating_vals,
        #                          params.src_in_units,
        #                          params.dst_in_units,
        #                          params.gcn_agg_units,
        #                          params.gcn_out_units,
        #                          dropout_rate=params.gcn_dropout,
        #                          device=params.device)

        self.decoder = MLPPredictor(in_units=params.gcn_out_units,
                                    src_in_units=params.src_in_units,
                                    dst_in_units=params.dst_in_units,
                                    num_classes=len(params.rating_vals))

        init.xavier_uniform_(self.weight)


    def forward(self, enc_graph, dec_graph):
        in_feats = {'node': self.weight}

        out_1 = self.encoder(enc_graph, in_feats)
        # out_2 = self.encoder(enc_graph, {'node': out_1})
        # out_3 = self.encoder(enc_graph, {'node': out_2})

        pred_ratings = self.decoder(dec_graph, out_1).squeeze()

        return pred_ratings


def evaluate(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = torch.FloatTensor(possible_rating_values).to(args.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with torch.no_grad():
        pred_ratings = net(enc_graph, dec_graph)
        if args.train_classification:
            real_pred_ratings = (torch.softmax(pred_ratings, dim=1) *
                                 nd_possible_rating_values.view(1, -1)).sum(dim=1)
            rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
        else:
            rmse = ((pred_ratings - rating_values) ** 2.).mean().item()
        # rmse = np.sqrt(rmse)
    return rmse


def train(params):
    print(params)

    dataset = Data(params.dataset_name,
                        params.dataset_path,
                        params.device,
                        params.review_feat_size)
    print("Loading data finished ...\n")

    params.src_in_units = dataset._num_user
    params.dst_in_units = dataset._num_item
    params.rating_vals = dataset.possible_rating_values

    net = Net(params)
    net = net.to(params.device)

    nd_possible_rating_values = torch.FloatTensor(dataset.possible_rating_values).to(params.device)
    rating_loss_net = nn.CrossEntropyLoss() if params.train_classification else nn.MSELoss()
    learning_rate = params.train_lr
    # l2正则也很重要
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    print("Loading network finished ...\n")

    # prepare training data
    if params.train_classification:
        train_gt_labels = dataset.train_labels
        train_gt_ratings = dataset.train_truths
    else:
        train_gt_labels = dataset.train_truths.float()
        train_gt_ratings = dataset.train_truths.float()

    # declare the loss information
    best_valid_rmse = np.inf
    best_test_rmse = np.inf
    no_better_valid = 0
    best_iter = -1

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(params.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
    dataset.valid_enc_graph = dataset.valid_enc_graph.int().to(params.device)
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(params.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)

    print("Start training ...")
    for iter_idx in range(1, params.train_max_iter):
        net.train()
        pred_ratings = net(dataset.train_enc_graph, dataset.train_dec_graph)
        loss = rating_loss_net(pred_ratings, train_gt_labels).mean()
        # print(pred_ratings)
        # print(train_gt_labels)
        # exit()

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
        optimizer.step()

        if params.train_classification:
            real_pred_ratings = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        else:
            real_pred_ratings = pred_ratings

        train_rmse = ((real_pred_ratings - train_gt_ratings) ** 2).mean()

        valid_rmse = evaluate(args=params, net=net, dataset=dataset, segment='valid')

        test_rmse = evaluate(args=params, net=net, dataset=dataset, segment='test')

        logging_str = f"Iter={iter_idx:>3d}, " \
                      f"Train_MSE={train_rmse:.4f}, Valid_MSE={valid_rmse:.4f}, Test_MSE={test_rmse:.4f}"

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_iter = iter_idx
            no_better_valid = 0
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience:
                print("Early stopping threshold reached. Stop training.")
                break
            # if no_better_valid > params.train_decay_patience:
            #     new_lr = max(learning_rate * params.train_lr_decay_factor, params.train_min_lr)
            #     if new_lr < learning_rate:
            #         learning_rate = new_lr
            #         print("\tChange the LR to %g" % new_lr)
            #         for p in optimizer.param_groups:
            #             p['lr'] = learning_rate
            #         no_better_valid = 0

        print(logging_str)
    print(f'Best Iter Idx={best_iter}, Best Test MSE={best_test_rmse:.4f}')


if __name__ == '__main__':
    config_args = config()
    train(config_args)