import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils import DomainDataset

data_name = 'tokyo'

val_data = DomainDataset('data', data_name, split='val')
vectors = torch.load('result/{}_daco_vectors.pth'.format(data_name))
labels = torch.arange(len(vectors) // 2, device=vectors.device)
labels = torch.cat((labels, labels), dim=0)
a_vectors = vectors[:len(vectors) // 2]
b_vectors = vectors[len(vectors) // 2:]
a_labels = labels[:len(vectors) // 2]
b_labels = labels[len(vectors) // 2:]
# domain b ---> domain a
sim_b = b_vectors.mm(a_vectors.t())
idx_b = sim_b.topk(k=1, dim=-1, largest=True)[1]
correct_b = (torch.eq(a_labels[idx_b[:, 0:1]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
for i in tqdm(range(len(correct_b))):
    if not correct_b[i]:
        query_path = val_data.original_images[i + len(a_vectors)]
        save_path = 'result/b_a/{}/{}'.format(data_name, query_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.open(query_path).save('{}/query.png'.format(save_path))
        rank_path = val_data.original_images[idx_b[i][0]]
        Image.open(rank_path).save('{}/rank.png'.format(save_path))

# domain a ---> domain b
sim_a = a_vectors.mm(b_vectors.t())
idx_a = sim_a.topk(k=1, dim=-1, largest=True)[1]
correct_a = (torch.eq(b_labels[idx_a[:, 0:1]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
for i in tqdm(range(len(correct_a))):
    if not correct_a[i]:
        query_path = val_data.original_images[i]
        save_path = 'result/a_b/{}/{}'.format(data_name, query_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.open(query_path).save('{}/query.png'.format(save_path))
        rank_path = val_data.original_images[idx_a[i][0] + len(a_vectors)]
        Image.open(rank_path).save('{}/rank.png'.format(save_path))

# domain a <---> domain b
sim = vectors.mm(vectors.t())
sim.fill_diagonal_(-np.inf)
idx = sim.topk(k=1, dim=-1, largest=True)[1]
correct = (torch.eq(labels[idx[:, 0:1]], labels.unsqueeze(dim=-1))).any(dim=-1)
for i in tqdm(range(len(correct))):
    if not correct[i]:
        query_path = val_data.original_images[i]
        save_path = 'result/ab/{}/{}'.format(data_name, query_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.open(query_path).save('{}/query.png'.format(save_path))
        rank_path = val_data.original_images[idx[i][0]]
        Image.open(rank_path).save('{}/rank.png'.format(save_path))
