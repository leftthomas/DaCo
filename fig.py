import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils import DomainDataset

data_name = 'tokyo'

val_data = DomainDataset('data', data_name, split='val')
npid_vectors = torch.load('result/{}_npid_vectors.pth'.format(data_name))
simclr_vectors = torch.load('result/{}_simclr_vectors.pth'.format(data_name))
simsiam_vectors = torch.load('result/{}_simsiam_vectors.pth'.format(data_name))
moco_vectors = torch.load('result/{}_moco_vectors.pth'.format(data_name))
daco_vectors = torch.load('result/{}_daco_vectors.pth'.format(data_name))

labels = torch.arange(len(simclr_vectors) // 2, device=simclr_vectors.device)
labels = torch.cat((labels, labels), dim=0)
a_labels = labels[:len(simclr_vectors) // 2]
b_labels = labels[len(simclr_vectors) // 2:]

npid_a_vectors = npid_vectors[:len(npid_vectors) // 2]
npid_b_vectors = npid_vectors[len(npid_vectors) // 2:]
simclr_a_vectors = simclr_vectors[:len(simclr_vectors) // 2]
simclr_b_vectors = simclr_vectors[len(simclr_vectors) // 2:]
simsiam_a_vectors = simsiam_vectors[:len(simsiam_vectors) // 2]
simsiam_b_vectors = simsiam_vectors[len(simsiam_vectors) // 2:]
moco_a_vectors = moco_vectors[:len(moco_vectors) // 2]
moco_b_vectors = moco_vectors[len(moco_vectors) // 2:]
daco_a_vectors = daco_vectors[:len(daco_vectors) // 2]
daco_b_vectors = daco_vectors[len(daco_vectors) // 2:]

# domain a ---> domain b
npid_sim_a = npid_a_vectors.mm(npid_b_vectors.t())
simclr_sim_a = simclr_a_vectors.mm(simclr_b_vectors.t())
simsiam_sim_a = simsiam_a_vectors.mm(simsiam_b_vectors.t())
moco_sim_a = moco_a_vectors.mm(moco_b_vectors.t())
daco_sim_a = daco_a_vectors.mm(daco_b_vectors.t())
npid_idx_a = npid_sim_a.topk(k=1, dim=-1, largest=True)[1]
npid_correct_a = (torch.eq(b_labels[npid_idx_a[:, 0:1]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
simclr_idx_a = simclr_sim_a.topk(k=1, dim=-1, largest=True)[1]
simclr_correct_a = (torch.eq(b_labels[simclr_idx_a[:, 0:1]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
simsiam_idx_a = simsiam_sim_a.topk(k=1, dim=-1, largest=True)[1]
simsiam_correct_a = (torch.eq(b_labels[simsiam_idx_a[:, 0:1]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
moco_idx_a = moco_sim_a.topk(k=1, dim=-1, largest=True)[1]
moco_correct_a = (torch.eq(b_labels[moco_idx_a[:, 0:1]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
daco_idx_a = daco_sim_a.topk(k=1, dim=-1, largest=True)[1]
daco_correct_a = (torch.eq(b_labels[daco_idx_a[:, 0:1]], a_labels.unsqueeze(dim=-1))).any(dim=-1)

# domain b ---> domain a
npid_sim_b = npid_b_vectors.mm(npid_a_vectors.t())
simclr_sim_b = simclr_b_vectors.mm(simclr_a_vectors.t())
simsiam_sim_b = simsiam_b_vectors.mm(simsiam_a_vectors.t())
moco_sim_b = moco_b_vectors.mm(moco_a_vectors.t())
daco_sim_b = daco_b_vectors.mm(daco_a_vectors.t())
npid_idx_b = npid_sim_b.topk(k=1, dim=-1, largest=True)[1]
npid_correct_b = (torch.eq(a_labels[npid_idx_b[:, 0:1]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
simclr_idx_b = simclr_sim_b.topk(k=1, dim=-1, largest=True)[1]
simclr_correct_b = (torch.eq(a_labels[simclr_idx_b[:, 0:1]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
simsiam_idx_b = simsiam_sim_b.topk(k=1, dim=-1, largest=True)[1]
simsiam_correct_b = (torch.eq(a_labels[simsiam_idx_b[:, 0:1]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
moco_idx_b = moco_sim_b.topk(k=1, dim=-1, largest=True)[1]
moco_correct_b = (torch.eq(a_labels[moco_idx_b[:, 0:1]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
daco_idx_b = daco_sim_b.topk(k=1, dim=-1, largest=True)[1]
daco_correct_b = (torch.eq(a_labels[daco_idx_b[:, 0:1]], b_labels.unsqueeze(dim=-1))).any(dim=-1)

# domain a <---> domain b
npid_sim = npid_vectors.mm(npid_vectors.t())
simclr_sim = simclr_vectors.mm(simclr_vectors.t())
simsiam_sim = simsiam_vectors.mm(simsiam_vectors.t())
moco_sim = moco_vectors.mm(moco_vectors.t())
daco_sim = daco_vectors.mm(daco_vectors.t())
npid_sim.fill_diagonal_(-np.inf)
simclr_sim.fill_diagonal_(-np.inf)
simsiam_sim.fill_diagonal_(-np.inf)
moco_sim.fill_diagonal_(-np.inf)
daco_sim.fill_diagonal_(-np.inf)
npid_idx = npid_sim.topk(k=1, dim=-1, largest=True)[1]
npid_correct = (torch.eq(labels[npid_idx[:, 0:1]], labels.unsqueeze(dim=-1))).any(dim=-1)
simclr_idx = simclr_sim.topk(k=1, dim=-1, largest=True)[1]
simclr_correct = (torch.eq(labels[simclr_idx[:, 0:1]], labels.unsqueeze(dim=-1))).any(dim=-1)
simsiam_idx = simsiam_sim.topk(k=1, dim=-1, largest=True)[1]
simsiam_correct = (torch.eq(labels[simsiam_idx[:, 0:1]], labels.unsqueeze(dim=-1))).any(dim=-1)
moco_idx = moco_sim.topk(k=1, dim=-1, largest=True)[1]
moco_correct = (torch.eq(labels[moco_idx[:, 0:1]], labels.unsqueeze(dim=-1))).any(dim=-1)
daco_idx = daco_sim.topk(k=1, dim=-1, largest=True)[1]
daco_correct = (torch.eq(labels[daco_idx[:, 0:1]], labels.unsqueeze(dim=-1))).any(dim=-1)

for i in tqdm(range(len(simclr_correct_a))):
    if daco_correct_a[i] and not npid_correct_a[i] and not simclr_correct_a[i] and not simsiam_correct_a[i] and not \
    moco_correct_a[i]:
        query_path = val_data.original_images[i]
        save_path = 'result/domain_a/{}'.format(query_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.open(query_path).save('{}/query.jpg'.format(save_path))

        npid_rank_path = val_data.original_images[npid_idx_a[i][0] + len(simclr_a_vectors)]
        Image.open(npid_rank_path).save(
            '{}/npid_{:.2f}.jpg'.format(save_path, npid_sim_a[i][npid_idx_a[i][0]].item()))
        simclr_rank_path = val_data.original_images[simclr_idx_a[i][0] + len(simclr_a_vectors)]
        Image.open(simclr_rank_path).save(
            '{}/simclr_{:.2f}.jpg'.format(save_path, simclr_sim_a[i][simclr_idx_a[i][0]].item()))
        simsiam_rank_path = val_data.original_images[simsiam_idx_a[i][0] + len(simclr_a_vectors)]
        Image.open(simsiam_rank_path).save(
            '{}/simsiam_{:.2f}.jpg'.format(save_path, simsiam_sim_a[i][simsiam_idx_a[i][0]].item()))
        moco_rank_path = val_data.original_images[moco_idx_a[i][0] + len(simclr_a_vectors)]
        Image.open(moco_rank_path).save(
            '{}/moco_{:.2f}.jpg'.format(save_path, moco_sim_a[i][moco_idx_a[i][0]].item()))
        daco_rank_path = val_data.original_images[daco_idx_a[i][0] + len(simclr_a_vectors)]
        Image.open(daco_rank_path).save(
            '{}/daco_{:.2f}.jpg'.format(save_path, daco_sim_a[i][daco_idx_a[i][0]].item()))

for i in tqdm(range(len(simclr_correct_b))):
    if daco_correct_b[i] and not npid_correct_b[i] and not simclr_correct_b[i] and not simsiam_correct_b[i] and not \
    moco_correct_b[i]:
        query_path = val_data.original_images[i + len(simclr_a_vectors)]
        save_path = 'result/domain_b/{}'.format(query_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.open(query_path).save('{}/query.jpg'.format(save_path))

        npid_rank_path = val_data.original_images[npid_idx_b[i][0]]
        Image.open(npid_rank_path).save(
            '{}/npid_{:.2f}.jpg'.format(save_path, npid_sim_b[i][npid_idx_b[i][0]].item()))
        simclr_rank_path = val_data.original_images[simclr_idx_b[i][0]]
        Image.open(simclr_rank_path).save(
            '{}/simclr_{:.2f}.jpg'.format(save_path, simclr_sim_b[i][simclr_idx_b[i][0]].item()))
        simsiam_rank_path = val_data.original_images[simsiam_idx_b[i][0]]
        Image.open(simsiam_rank_path).save(
            '{}/simsiam_{:.2f}.jpg'.format(save_path, simsiam_sim_b[i][simsiam_idx_b[i][0]].item()))
        moco_rank_path = val_data.original_images[moco_idx_b[i][0]]
        Image.open(moco_rank_path).save(
            '{}/moco_{:.2f}.jpg'.format(save_path, moco_sim_b[i][moco_idx_b[i][0]].item()))
        daco_rank_path = val_data.original_images[daco_idx_b[i][0]]
        Image.open(daco_rank_path).save(
            '{}/daco_{:.2f}.jpg'.format(save_path, daco_sim_b[i][daco_idx_b[i][0]].item()))

for i in tqdm(range(len(simclr_correct))):
    if daco_correct[i] and not npid_correct[i] and not simclr_correct[i] and not simsiam_correct[i] and not \
    moco_correct[i]:
        query_path = val_data.original_images[i]
        save_path = 'result/domain_ab/{}'.format(query_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.open(query_path).save('{}/query.jpg'.format(save_path))

        npid_rank_path = val_data.original_images[npid_idx[i][0]]
        Image.open(npid_rank_path).save(
            '{}/npid_{:.2f}.jpg'.format(save_path, npid_sim[i][npid_idx[i][0]].item()))
        simclr_rank_path = val_data.original_images[simclr_idx[i][0]]
        Image.open(simclr_rank_path).save(
            '{}/simclr_{:.2f}.jpg'.format(save_path, simclr_sim[i][simclr_idx[i][0]].item()))
        simsiam_rank_path = val_data.original_images[simsiam_idx[i][0]]
        Image.open(simsiam_rank_path).save(
            '{}/simsiam_{:.2f}.jpg'.format(save_path, simsiam_sim[i][simsiam_idx[i][0]].item()))
        moco_rank_path = val_data.original_images[moco_idx[i][0]]
        Image.open(moco_rank_path).save(
            '{}/moco_{:.2f}.jpg'.format(save_path, moco_sim[i][moco_idx[i][0]].item()))
        daco_rank_path = val_data.original_images[daco_idx[i][0]]
        Image.open(daco_rank_path).save(
            '{}/daco_{:.2f}.jpg'.format(save_path, daco_sim[i][daco_idx[i][0]].item()))
