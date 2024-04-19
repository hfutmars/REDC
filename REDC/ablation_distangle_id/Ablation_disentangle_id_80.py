# -*- coding: utf-8 -*-
import argparse
import dgl.function as fn
from util import *
import torch
import dgl
from load_data_graph_augmutation import *
from util import *
import random
import heapq

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.manual_seed(2022)
np.random.seed(2022)

global_emb_size = 16
dataset_name = os.listdir("../data/")[0]
class Data(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size):
        self._device = device
        self._review_fea_size = review_fea_size

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info, strange_users, strange_users_max, user_items_train = load_sentiment_data(dataset_path)

        # self.strange_users = strange_users
        #
        # self.strange_users_high = strange_users_max
        #
        # self.user_items_train = user_items_train
        # self.remove_users = []

        self._num_user = dataset_info['user_size']
        self._num_item = dataset_info['item_size']

        review_feat_path = f'../checkpoint/{dataset_name}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        # strange_users_reviews = {}
        # for key, value in self.train_review_feat.items():
        #     u, i = key[0], key[1]
        #     if u in self.strange_users:
        #         if u not in strange_users_reviews:
        #             strange_users_reviews[u] = []
        #         strange_users_reviews[u].append(value)
        #
        # for u, values in strange_users_reviews.items():
        #     cos_list = []
        #     for i in range(len(values)):
        #         for j in np.arange(i + 1, len(values)):
        #             cos = torch.cosine_similarity(values[i], values[j], dim=0)
        #             cos_list.append(cos.numpy())
        #     if np.mean(cos_list) > 0.95:
        #         self.remove_users.append(u)

        self.review_feat_updated = {}


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

        # print(f'user number: {self._num_user}, item number: {self._num_item}')

        self.user_item_rating = {}
        self.user_rating_count = {}
        self.user_ratings_test ={}
        self.user_item_ratings = {}

        self.user_items = {}

        def _generate_train_pair_value(data: tuple):
            user_id, item_id, rating = np.array(data[0], dtype=np.int64), np.array(data[1], dtype=np.int64), \
                                       np.array(data[2], dtype=np.int64)

            rating_pairs = (user_id, item_id)
            rating_pairs_rev = (item_id, user_id)
            rating_pairs = np.concatenate([rating_pairs, rating_pairs_rev], axis=1)

            rating_values = np.concatenate([rating, rating],
                                           axis=0)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_item_rating:
                    self.user_item_rating[uid] = []
                    self.user_item_ratings[uid] = {}
                    self.user_items[uid] = []
                self.user_item_rating[uid].append((iid, rating[i]))
                self.user_item_ratings[uid][iid] = rating[i]
                self.user_items[uid].append(iid)

                if uid not in self.user_rating_count:
                    self.user_rating_count[uid] = [0,0,0,0,0]

                self.user_rating_count[uid][rating[i] - 1] += 1

            return rating_pairs, rating_values

        def _generate_valid_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            return rating_pairs, rating_values

        def _generate_test_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_ratings_test:
                    self.user_ratings_test[uid] = []

                self.user_ratings_test[uid].append(rating[i])

            return rating_pairs, rating_values

        print('Generating train/valid/test data.\n')
        self.train_rating_pairs, self.train_rating_values = _generate_train_pair_value(self.train_datas)
        self.valid_rating_pairs, self.valid_rating_values = _generate_valid_pair_value(self.valid_datas)
        self.test_rating_pairs, self.test_rating_values = _generate_test_pair_value(self.test_datas)

        count_mis = 0
        count_same = 0
        count_all = 0
        for uid, items in self.user_ratings_test.items():
            count_all += len(items)
            max_rate_train = np.where(self.user_rating_count[uid] == np.max(self.user_rating_count[uid]))[0]
            for i in items:
                if i - 1 not in max_rate_train:
                    count_mis += 1
                else:
                    count_same += 1

        print(count_mis, count_same, count_all, len(self.test_rating_values))

        ## find and collect extremely distributed samples
        self.extra_dist_pairs = {}
        self.extra_uid, self.extra_iid, self.extra_r_idx = [], [], []
        for uid, l in self.user_rating_count.items():

            max_count = np.max(l)
            max_idx = np.where(l == max_count)[0]

            for i, c in enumerate(l):
                # if c == 0 or abs(max_idx.max() - i) <= 1 or abs(max_idx.min() - i) <= 1:
                if i in max_idx or c == 0:
                    continue

                if c / max_count <= 0.2:
                    if uid not in self.extra_dist_pairs:
                        self.extra_dist_pairs[uid] = []
                    self.extra_dist_pairs[uid].append((i+1, c))
                    for item in self.user_item_rating[uid]:

                        self.extra_uid.append(uid)
                        self.extra_iid.append(item[0])
                        self.extra_r_idx.append(i)

        # for uid, l in self.extra_dist_pairs.items():
        #     print(uid, l)
        #     print(self.user_rating_count[uid])

        # self.extra_uid = torch.LongTensor(self.extra_uid).to('cuda:0')
        # self.extra_r_idx = torch.LongTensor(self.extra_r_idx).to('cuda:0')
        # print(self.extra_uid)
        # print(self.extra_r_idx)
        # exit()

        self.item_rate_review = {}

        # for k, v in self.train_review_feat.items():
        #     uid, iid = k[0], k[1] + self._num_user
        #
        #     if iid not in self.user_item_ratings[uid]:
        #         continue
        #     r = self.user_item_ratings[uid][iid]
        #
        #     if iid not in self.item_rate_review:
        #         self.item_rate_review[iid] = {}
        #
        #     if r not in self.item_rate_review[iid]:
        #         self.item_rate_review[iid][r] = []
        #     self.item_rate_review[iid][r].append(v)

        for u, d in self.user_item_ratings.items():
            for i, r in d.items():
                review = self.train_review_feat[(u, i - self._num_user)]
                if i not in self.item_rate_review:
                    self.item_rate_review[i] = {}
                if r not in self.item_rate_review[i]:
                    self.item_rate_review[i][r] = []
                self.item_rate_review[i][r].append(review)

        self.mean_review_feat_list_1 = []
        self.mean_review_feat_list_2 = []
        self.mean_review_feat_list_3 = []
        self.mean_review_feat_list_4 = []
        self.mean_review_feat_list_5 = []
        for key, value in self.train_review_feat.items():
            self.review_feat_updated[(key[0], key[1] + self._num_user)] = value
            self.review_feat_updated[(key[1] + self._num_user, key[0])] = value
            if key[1] + self._num_user not in self.user_item_ratings[key[0]]:
                continue

            if self.user_item_ratings[key[0]][key[1] + self._num_user] == 1:
                self.mean_review_feat_list_1.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 2:
                self.mean_review_feat_list_2.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 3:
                self.mean_review_feat_list_3.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 4:
                self.mean_review_feat_list_4.append(value)

            else:
                self.mean_review_feat_list_5.append(value)

        self.mean_review_feat_1 = torch.mean(torch.stack(self.mean_review_feat_list_1, dim=0), dim=0)
        self.mean_review_feat_2 = torch.mean(torch.stack(self.mean_review_feat_list_2, dim=0), dim=0)
        self.mean_review_feat_3 = torch.mean(torch.stack(self.mean_review_feat_list_3, dim=0), dim=0)
        self.mean_review_feat_4 = torch.mean(torch.stack(self.mean_review_feat_list_4, dim=0), dim=0)
        self.mean_review_feat_5 = torch.mean(torch.stack(self.mean_review_feat_list_5, dim=0), dim=0)

        print('Generating train graph.\n')
        self.train_enc_graph = self._generate_enc_graph()

    def update_graph(self, uid_list, iid_list, r_list):
        uid_list, iid_list, r_list = np.array(uid_list), np.array(iid_list), np.array(r_list)
        rating_pairs = (uid_list, iid_list)
        rating_pairs_rev = (iid_list, uid_list)
        self.train_rating_pairs = np.concatenate([self.train_rating_pairs, rating_pairs, rating_pairs_rev], axis=1)

        self.train_rating_values = np.concatenate([self.train_rating_values, r_list, r_list], axis=0)
        c0, c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0, 0

        for i, u in enumerate(uid_list):

            r = r_list[i]
            iid = iid_list[i]
            if r in self.item_rate_review[iid]:
                review = torch.mean(torch.stack(self.item_rate_review[iid][r], dim=0), dim=0)
                c0 += 1
            elif r == 1:
                review = self.mean_review_feat_1
                c1 += 1
            elif r == 2:
                review = self.mean_review_feat_2
                c2 += 1
            elif r == 3:
                review = self.mean_review_feat_3
                c3 += 1
            elif r == 4:
                review = self.mean_review_feat_4
                c4 += 1
            else:
                review = self.mean_review_feat_5
                c5 += 1

            self.review_feat_updated[(u, iid_list[i])] = review
            self.review_feat_updated[(iid_list[i], u)] = review
        print(c0, c1, c2, c3, c4, c5)

        self.train_enc_graph_updated = self._generate_enc_graph()


    def _generate_enc_graph(self):
        user_item_r = np.zeros((self._num_user + self._num_item, self._num_item + self._num_user), dtype=np.float32)
        for i in range(len(self.train_rating_values)):
            user_item_r[[self.train_rating_pairs[0][i], self.train_rating_pairs[1][i]]] = self.train_rating_values[i]
        record_size = self.train_rating_pairs[0].shape[0]
        review_feat_list = [self.review_feat_updated[(self.train_rating_pairs[0][x], self.train_rating_pairs[1][x])] for x in
                            range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        rating_row, rating_col = self.train_rating_pairs

        graph_dict = {}
        for rating in self.possible_rating_values:
            ridx = np.where(self.train_rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]

            graph_dict[str(rating)] = dgl.graph((rrow, rcol), num_nodes=self._num_user + self._num_item)
            graph_dict[str(rating)].edata['review_feat'] = review_feat_list[ridx]

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)

        c = []
        for r_1 in self.possible_rating_values:
            c.append(graph_dict[str(r_1)].in_degrees())
            graph_dict[str(r_1)].ndata['ci_r'] = _calc_norm(graph_dict[str(r_1)].in_degrees(), 0.5)

        c_sum = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 0.5)
        c_sum_mean = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 1)

        for r_1 in self.possible_rating_values:
            graph_dict[str(r_1)].ndata['ci'] = c_sum
            graph_dict[str(r_1)].ndata['ci_mean'] = c_sum_mean

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
            end_idx = begin_idx + batch_size if i != n_batch - 1 else len(self.train_rating_values)
            batch_users, batch_items, batch_ratings = users[begin_idx: end_idx], items[
                                                                                 begin_idx: end_idx], rating_values[
                                                                                                      begin_idx: end_idx]

            u_list.append(torch.LongTensor(batch_users).to('cuda:0'))
            i_list.append(torch.LongTensor(batch_items).to('cuda:0'))
            r_list.append(torch.LongTensor(batch_ratings - 1).to('cuda:0'))

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


def config():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--device', default='0', type=int, help='gpu.')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--train_max_iter', type=int, default=1000)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)

    args = parser.parse_args()
    args.model_short_name = 'RGC'
    args.dataset_name = dataset_name
    args.dataset_path = f'../data/{dataset_name}/{dataset_name}.json'
    args.emb_size = 64
    args.emb_dim = 12
    args.origin_emb_dim = 60

    args.gcn_dropout = 0.7
    args.device = torch.device(args.device)
    args.train_max_iter = 1000
    # args.batch_size = 271466
    args.batch_size = 100000

    return args

gloabl_dropout = 0.5



global_review_size = 128

class GCN_1(nn.Module):
    def __init__(self):
        super(GCN_1, self).__init__()
        self.dropout = nn.Dropout(0.7)
        self.review_w = nn.Linear(64, global_review_size, bias=False, device='cuda:0')
        # self.emb_w = nn.Linear(12, 64, bias=False, device='cuda:0')

    def forward(self, g, feature):

        g.srcdata['h_r'] = feature
        g.edata['r'] = self.review_w(g.edata['review_feat'])

        g.update_all(lambda edges: {
            'm': (torch.cat([edges.data['r'], edges.src['h_r']], dim=-1)) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst


class GCN_2(nn.Module):
    def __init__(self):
        super(GCN_2, self).__init__()
        self.dropout = nn.Dropout(0.7)
        self.review_w = nn.Linear(64, global_review_size, bias=False, device='cuda:0')

    def forward(self, g, feature):

        g.srcdata['h_r'] = feature
        g.edata['r'] = self.review_w(g.edata['review_feat'])

        g.update_all(lambda edges: {
            'm': (torch.cat([edges.data['r'], edges.src['h_r']], dim=-1)) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst

class GCN_3(nn.Module):
    def __init__(self):
        super(GCN_3, self).__init__()
        self.dropout = nn.Dropout(gloabl_dropout)
        self.review_w = nn.Linear(64, global_review_size, bias=False, device='cuda:0')

    def forward(self, g, feature):

        g.srcdata['h_r'] = feature
        g.edata['r'] = self.review_w(g.edata['review_feat'])

        g.update_all(lambda edges: {
            'm': (torch.cat([edges.data['r'], edges.src['h_r']], dim=-1)) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        self.weight = nn.ParameterDict({
            str(r): nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
            for r in [1, 2, 3, 4, 5]
        })

        self.weight_origin = nn.Parameter(torch.Tensor(params.num_users + params.num_items, global_emb_size * 5))

        self.encoder_1 = nn.ModuleDict({
            str(i + 1): GCN_1() for i in range(5)
        })

        self.encoder_2 = nn.ModuleDict({
            str(i + 1): GCN_2() for i in range(5)
        })

        self.encoder_3 = nn.ModuleDict({
            str(i + 1): GCN_3() for i in range(5)
        })

        self.num_user = params.num_users
        self.num_item = params.num_items

        self.fc_user = nn.Linear(global_emb_size * 5 + global_review_size * 5 * 1, global_emb_size * 5 + global_review_size * 5 * 1)
        self.fc_item = nn.Linear(global_emb_size * 5 + global_review_size * 5 * 1, global_emb_size * 5 + global_review_size * 5 * 1)
        self.dropout = nn.Dropout(0.7)

        self.predictor = nn.Sequential(
            nn.Linear(global_emb_size * 5 + global_review_size * 5 * 1, global_emb_size * 5 + global_review_size * 5 * 1, bias=False),
            nn.ReLU(),
            nn.Linear(global_emb_size * 5 + global_review_size * 5 * 1, 5, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_graph_dict, users, items):
        feat_all = []
        rate_list = [1, 2, 3, 4, 5]
        weight_origin_all = torch.split(self.weight_origin, [global_emb_size, global_emb_size, global_emb_size, global_emb_size, global_emb_size], dim=-1)

        for r in rate_list:
            # first layer
            feat_ = self.encoder_1[str(r)](enc_graph_dict[str(r)], weight_origin_all[r - 1])

            # second layer
            # feat_ = self.encoder_2[str(r)](enc_graph_dict[str(r)], feat_)

            # third layer
            # feat_ = self.encoder_3[str(r)](enc_graph_dict[str(r)], feat_)

            feat_all.append(feat_)

        feat = torch.cat(feat_all, dim=-1)

        user_feat, item_feat = torch.split(feat, [self.num_user, self.num_item], dim=0)

        user_feat = self.dropout(user_feat)
        u_feat = self.fc_user(user_feat)

        item_feat = self.dropout(item_feat)
        i_feat = self.fc_item(item_feat)

        feat = torch.cat([u_feat, i_feat], dim=0)

        user_embeddings, item_embeddings = feat[users], feat[items]

        pred_ratings = self.predictor(user_embeddings * item_embeddings).squeeze()

        return pred_ratings
    
    
def evaluate(args, net, dataset, flag='valid', add=False, epoch=256):
    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(args.device)

    u_list, i_list, r_list = dataset._test_data(flag=flag)

    net.eval()
    with torch.no_grad():

        if epoch <= g_epoch:

            pred_ratings = net(dataset.train_enc_graph, u_list, i_list)
        else:
            pred_ratings = net(dataset.train_enc_graph_updated, u_list, i_list)

        real_pred_ratings = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

        u_list = u_list.cpu().numpy()
        r_list = r_list.cpu().numpy()
        real_pred_ratings = real_pred_ratings.cpu().numpy()

        mse = ((real_pred_ratings - r_list) ** 2.).mean()
        mae = (np.abs(real_pred_ratings - r_list)).mean()

    return mse, mae

g_epoch = 1000
def train(params):

    dataset = Data(params.dataset_name,
                        params.dataset_path,
                        params.device,
                        params.emb_size,
                        )
    print("Loading data finished.\n")

    params.num_users = dataset._num_user
    params.num_items = dataset._num_item

    params.rating_vals = dataset.possible_rating_values

    print(f'Dataset information:\n \tuser num: {params.num_users}\n\titem num: {params.num_items}\n\ttrain interaction num: {len(dataset.train_rating_values)}\n')

    net = Net(params)
    net = net.to(params.device)

    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = params.train_lr

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    print("Loading network finished.\n")

    best_test_mse = np.inf
    final_test_mae = np.inf
    no_better_valid = 0
    best_iter = -1

    for r in [1, 2, 3, 4, 5]:
        dataset.train_enc_graph[str(r)] = dataset.train_enc_graph[str(r)].int().to(params.device)
    print(dataset.train_enc_graph)

    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(params.device)

    print("Training and evaluation.")
    u_list, i_list, r_list = dataset._train_data(batch_size=params.batch_size)
    for b in u_list:
        print(b.shape)
    for iter_idx in range(1, params.train_max_iter):
        net.train()

        train_mse = 0.

        for idx in range(len(r_list)):
            batch_user = u_list[idx]
            batch_item = i_list[idx]
            batch_rating = r_list[idx]
            if iter_idx <= g_epoch:
                pred_ratings = net(dataset.train_enc_graph, batch_user, batch_item)
            else:
                pred_ratings = net(dataset.train_enc_graph_updated, batch_user,
                                                                         batch_item)

            real_pred_ratings = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

            loss = rating_loss_net(pred_ratings, batch_rating).mean()

            loss_f = loss

            optimizer.zero_grad()

            loss_f.backward()

            optimizer.step()

            train_mse += ((real_pred_ratings - batch_rating - 1) ** 2).sum()

        train_mse = train_mse / len(dataset.train_rating_values)

        # valid_mse = evaluate(args=params, net=net, dataset=dataset, flag='valid')

        test_mse, test_mae = evaluate(args=params, net=net, dataset=dataset, flag='test', add=False, epoch=iter_idx)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            final_test_mae = test_mae
            best_iter = iter_idx
            no_better_valid = 0
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience:
                print("Early stopping threshold reached. Stop training.")
                break

        # print(f'Epoch {iter_idx}, Train_MSE={train_mse:.4f}, Test_MSE={test_mse:.4f}, Test_MAE={test_mae:.4f}')
    print(f'Best Iter Idx={best_iter}, Best Test MSE={best_test_mse:.4f}, corresponding MAE={final_test_mae:.4f}')


if __name__ == '__main__':
    config_args = config()
    for i in range(5):
        train(config_args)
