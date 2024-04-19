import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch as th
import dgl
import sys

sys.path.append("..")

from load_data import *
from util import *


class Data(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size):
        self._device = device
        self._review_fea_size = review_fea_size

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = load_sentiment_data(dataset_path)

        self._num_user = dataset_info['user_size']
        self._num_item = dataset_info['item_size']

        review_feat_path = f'../checkpoint/{dataset_name}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        self.review_feat_updated = {}
        
        for key, value in self.train_review_feat.items():
            self.review_feat_updated[(key[0], key[1] + self._num_user)] = value
            self.review_feat_updated[(key[1] + self._num_user, key[0])] = value

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            # 让物品的id从max user id开始，相当于将用户和物品节点视为一类节点；
            item_id = [int(i) + self._num_user for i in info['item_id'].to_list()]
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        print(f'user number: {self._num_user}, item number: {self._num_item}')

        def _generate_train_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))
            rating_pairs_rev = (np.array(item_id, dtype=np.int64),
                            np.array(user_id, dtype=np.int64))
            rating_pairs = np.concatenate([rating_pairs, rating_pairs_rev], axis=1)

            rating_values = np.concatenate([np.array(rating, dtype=np.float32), np.array(rating, dtype=np.float32)], axis=0)

            return rating_pairs, rating_values

        def _generate_test_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            return rating_pairs, rating_values

        self.train_rating_pairs, self.train_rating_values = _generate_train_pair_value(self.train_datas)
        self.valid_rating_pairs, self.valid_rating_values = _generate_test_pair_value(self.valid_datas)
        self.test_rating_pairs, self.test_rating_values = _generate_test_pair_value(self.test_datas)

        # generate train_review_pairs
        self.train_review_pairs = []
        for idx in range(len(self.train_rating_values)):
            u, i = self.train_rating_pairs[0][idx], self.train_rating_pairs[1][idx]
            review = self.review_feat_updated[(u, i)].numpy()
            self.train_review_pairs.append(review)

        self.train_review_pairs = np.array(self.train_review_pairs)

        self.train_enc_graph = self._generate_enc_graph(self.train_rating_pairs, self.train_rating_values)
        # self.train_dec_graph = self._generate_dec_graph(self.train_rating_pairs, review_feat=self.review_feat_updated)
        # self.train_labels = th.LongTensor(self.train_rating_values - 1).to(device)
        # self.train_truths = th.FloatTensor(self.train_rating_values).to(device)

        # self.valid_enc_graph = self.train_enc_graph
        # self.valid_dec_graph = self._generate_dec_graph(self.valid_rating_pairs, review_feat=self.review_feat_updated)
        # self.valid_labels = th.LongTensor(self.valid_rating_values - 1).to(device)
        # self.valid_truths = th.FloatTensor(self.valid_rating_values).to(device)

        # self.test_enc_graph = self.train_enc_graph
        # self.test_dec_graph = self._generate_dec_graph(self.test_rating_pairs, review_feat=self.review_feat_updated)
        # self.test_labels = th.LongTensor(self.valid_rating_values - 1).to(device)
        # self.test_truths = th.FloatTensor(self.test_rating_values).to(device)

    def _generate_enc_graph(self, rating_pairs, rating_values):
        user_item_r = np.zeros((self._num_user + self._num_item, self._num_item + self._num_user), dtype=np.float32)
        for i in range(len(rating_values)):
            user_item_r[[rating_pairs[0][i], rating_pairs[1][i]]] = rating_values[i]
        record_size = rating_pairs[0].shape[0]
        review_feat_list = [self.review_feat_updated[(rating_pairs[0][x], rating_pairs[1][x])] for x in range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        rating_row, rating_col = rating_pairs

        graph_dict = {}
        for rating in self.possible_rating_values:

            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]

            graph_dict[str(rating)] = dgl.graph((rrow, rcol), num_nodes=self._num_user + self._num_item)
            graph_dict[str(rating)].edata['review_feat'] = review_feat_list[ridx]

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = th.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)

        c = []
        for r_1 in self.possible_rating_values:
            c.append(graph_dict[str(r_1)].in_degrees())
            graph_dict[str(r_1)].ndata['ci_r'] = _calc_norm(graph_dict[str(r_1)].in_degrees(), 0.5)

        c_sum = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 0.5)

        for r_1 in self.possible_rating_values:
            # graph_dict[str(r_1)].ndata['ci'] = _calc_norm(graph_dict[str(r_1)].in_degrees())
            graph_dict[str(r_1)].ndata['ci'] = c_sum


        return graph_dict

    
    def _train_data(self, batch_size=1024):

        rating_pairs, rating_values = self.train_rating_pairs, self.train_rating_values
        idx = np.arange(0, len(rating_values))
        np.random.shuffle(idx)
        rating_pairs = (rating_pairs[0][idx], rating_pairs[1][idx])
        rating_values = rating_values[idx]

        data_len = len(rating_values)

        users, items = rating_pairs[0], rating_pairs[1]
        u_list, i_list, r_list = [], [], []
        review_list = []
        n_batch = data_len // batch_size + 1

        for i in range(n_batch):
            begin_idx = i * batch_size
            end_idx = begin_idx + batch_size
            batch_users, batch_items, batch_ratings = users[begin_idx: end_idx], items[begin_idx: end_idx], rating_values[begin_idx: end_idx]
            batch_reviews = self.train_review_pairs[begin_idx: end_idx]

            u_list.append(torch.LongTensor(batch_users).to('cuda:0'))
            i_list.append(torch.LongTensor(batch_items).to('cuda:0'))
            r_list.append(torch.FloatTensor(batch_ratings).to('cuda:0'))

            review_list.append(torch.FloatTensor(batch_reviews).to('cuda:0'))

        return u_list, i_list, r_list

    def _test_data(self, flag='valid'):
        if flag == 'valid':
            rating_pairs, rating_values = self.valid_rating_pairs, self.valid_rating_values
        else:
            rating_pairs, rating_values = self.test_rating_pairs, self.test_rating_values
        u_list, i_list, r_list = [], [], []
        for i in range(len(rating_values)):
            u_list.append(rating_pairs[0][i])
            i_list.append(rating_pairs[1][i])
            r_list.append(rating_values[i])
        u_list = torch.LongTensor(u_list).to('cuda:0')
        i_list = torch.LongTensor(i_list).to('cuda:0')
        r_list = torch.FloatTensor(r_list).to('cuda:0')
        return u_list, i_list, r_list