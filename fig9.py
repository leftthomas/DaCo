import os

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
        save_path = 'result/{}'.format(query_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.open(query_path).save('{}/query.png'.format(save_path))
        rank_path = val_data.original_images[idx_b[i][0]]
        Image.open(rank_path).save('{}/rank.png'.format(save_path))
