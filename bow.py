import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.cluster.vq import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VLAD Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='tokyo', type=str, choices=['tokyo', 'cityscapes', 'synthia'],
                        help='Dataset name')
    parser.add_argument('--ranks', nargs='+', default=[1, 2, 4, 8], help='Selected recall')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, ranks, save_root = args.data_root, args.data_name, args.ranks, args.save_root

    # data prepare
    original_images = glob.glob(os.path.join(data_root, data_name, 'original', '*', 'val', '*.jpg'))
    original_images.sort()

    results = {'val_precise': []}
    for rank in ranks:
        results['val_ab_recall@{}'.format(rank)] = []
        results['val_ba_recall@{}'.format(rank)] = []
        results['val_cross_recall@{}'.format(rank)] = []
    save_name_pre = '{}_BOW'.format(data_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # for loop
    vectors = []
    for data_path in tqdm(original_images, desc='Feature extracting', dynamic_ncols=True):
        data = cv2.imread(data_path)
        scale_percent = data.shape[0] / 256
        width = int(data.shape[1] / scale_percent)
        height = 256
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        vectors.append(descriptors)

    descriptors = vectors[0]
    for descriptor in vectors[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    clusters = KMeans(n_clusters=64).fit(descriptors).cluster_centers_

    im_features = np.zeros((len(vectors), 64), np.float32)
    for i in range(0, len(vectors)):
        words, distance = vq(vectors[i], clusters)
        for w in words:
            im_features[i][w] += 1

    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(vectors) + 1) / (1.0 * nbr_occurences + 1)), np.float32)
    im_features = im_features * idf
    vectors = preprocessing.normalize(im_features, norm='l2')
    vectors = torch.tensor(vectors)

    # matching
    sim_matrix = torch.mm(vectors, vectors.t())

    with torch.no_grad():
        labels = torch.arange(len(vectors) // 2)
        labels = torch.cat((labels, labels), dim=0)
        a_labels = labels[:len(vectors) // 2]
        b_labels = labels[len(vectors) // 2:]
        # domain a ---> domain b
        sim_a = sim_matrix[:len(vectors) // 2, len(vectors) // 2:]
        idx_a = sim_a.topk(k=ranks[-1], dim=-1, largest=True)[1]
        # domain b ---> domain a
        sim_b = sim_matrix[len(vectors) // 2:, :len(vectors) // 2]
        idx_b = sim_b.topk(k=ranks[-1], dim=-1, largest=True)[1]
        # cross domain
        sim_matrix.fill_diagonal_(-np.inf)
        idx = sim_matrix.topk(k=ranks[-1], dim=-1, largest=True)[1]

        acc_a, acc_b, acc = [], [], []
        for r in ranks:
            correct_a = (torch.eq(b_labels[idx_a[:, 0:r]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
            acc_a.append((torch.sum(correct_a) / correct_a.size(0)).item())
            correct_b = (torch.eq(a_labels[idx_b[:, 0:r]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
            acc_b.append((torch.sum(correct_b) / correct_b.size(0)).item())
            correct = (torch.eq(labels[idx[:, 0:r]], labels.unsqueeze(dim=-1))).any(dim=-1)
            acc.append((torch.sum(correct) / correct.size(0)).item())

        precise = (acc_a[0] + acc_b[0] + acc[0]) / 3
        desc = 'Val: '
        for i, r in enumerate(ranks):
            results['val_ab_recall@{}'.format(r)].append(acc_a[i] * 100)
            results['val_ba_recall@{}'.format(r)].append(acc_b[i] * 100)
            results['val_cross_recall@{}'.format(r)].append(acc[i] * 100)
        desc += '| (A->B) R@{}:{:.2f}% | '.format(ranks[0], acc_a[0] * 100)
        desc += '(B->A) R@{}:{:.2f}% | '.format(ranks[0], acc_b[0] * 100)
        desc += '(A<->B) R@{}:{:.2f}% | '.format(ranks[0], acc[0] * 100)
        print(desc)

    results['val_precise'].append(precise * 100)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, 2))
    data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')