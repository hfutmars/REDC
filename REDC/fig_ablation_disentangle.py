import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('ablation_distangle_id_origin.pickle', 'rb') as file:
    origin = pickle.load(file)

with open('ablation_distangle_id.pickle', 'rb') as file:
    corr = pickle.load(file)

with open('ablation_distangle_id_user.pickle', 'rb') as file:
    corr_user = pickle.load(file)

with open('ablation_distangle_id_item.pickle', 'rb') as file:
    corr_item = pickle.load(file)

fig = plt.figure(figsize=(8, 3), dpi=200)

X = np.arange(0, 79) * 5

print(X)

origin_list, corr_list, corr_user_list, corr_item_list = [], [], [], []
print(len(origin))
for i in X:
    origin_list.append(origin[i])
    corr_list.append(corr[i])
    corr_user_list.append(corr_user[i])
    corr_item_list.append(corr_item[i])


# exit()

sub_fig_1 = plt.subplot(111)
title_name = '1) Digital Music'
y_label = r'MSE'

sub_fig_1.plot(X, origin_list, 'yo-', label='origin', markersize='3', alpha=0.8)
# sub_fig_1.plot(X, corr_user_list, 'bo-', label='corr_user', markersize='3', alpha=0.8)
# sub_fig_1.plot(X, corr_user_list, 'co-', label='corr_item', markersize='3', alpha=0.8)
sub_fig_1.plot(X, corr_list, 'mo-', label='corr', markersize='3', alpha=0.8)
sub_fig_1.set_ylim(0.2, 0.5)
sub_fig_1.legend(fontsize=8)
# sub_fig_1.set_xlabel(r'Top K', fontsize=15)
sub_fig_1.set_ylabel(y_label, fontsize=12)
sub_fig_1.set_title(title_name, fontsize=12)
sub_fig_1.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
sub_fig_1.set_xticks(np.arange(0, 20)*21)
# sub_fig_1.grid()

plt.tight_layout()
plt.savefig('corr.pdf')
plt.savefig('corr.jpg')