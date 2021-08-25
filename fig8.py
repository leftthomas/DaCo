import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from utils import DomainDataset


def plot_embedding(data, domain, label, idx, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    correct = 0

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if domain[i]:
            color = '#00CED1'
        else:
            color = '#DC143C'
        if label[i] == label[idx[i]]:
            correct += 1
            shape = '*'
        else:
            shape = 'o'
        plt.scatter(data[i, 0], data[i, 1], s=10, c=color, marker=shape)
    plt.savefig('result/{}_{}.pdf'.format(title, correct), dpi=30, bbox_inches='tight', pad_inches=0)


data_name = 'tokyo'

val_data = DomainDataset('data', data_name, split='val')
npid_vectors = torch.load('result/{}_npid_vectors.pth'.format(data_name)).cpu()
simclr_vectors = torch.load('result/{}_simclr_vectors.pth'.format(data_name)).cpu()
moco_vectors = torch.load('result/{}_moco_vectors.pth'.format(data_name)).cpu()
simsiam_vectors = torch.load('result/{}_simsiam_vectors.pth'.format(data_name)).cpu()
daco_vectors = torch.load('result/{}_daco_vectors.pth'.format(data_name)).cpu()

domains = torch.cat((torch.ones(len(simclr_vectors) // 2, dtype=torch.long),
                     torch.zeros(len(simclr_vectors) // 2, dtype=torch.long)), dim=0).cpu()
labels = torch.arange(len(simclr_vectors) // 2).cpu()
labels = torch.cat((labels, labels), dim=0)

npid_sim = npid_vectors.mm(npid_vectors.t())
npid_sim.fill_diagonal_(-np.inf)
npid_idx = npid_sim.topk(k=1, dim=-1, largest=True)[1].squeeze()

simclr_sim = simclr_vectors.mm(simclr_vectors.t())
simclr_sim.fill_diagonal_(-np.inf)
simclr_idx = simclr_sim.topk(k=1, dim=-1, largest=True)[1].squeeze()

moco_sim = moco_vectors.mm(moco_vectors.t())
moco_sim.fill_diagonal_(-np.inf)
moco_idx = moco_sim.topk(k=1, dim=-1, largest=True)[1].squeeze()

simsiam_sim = simsiam_vectors.mm(simsiam_vectors.t())
simsiam_sim.fill_diagonal_(-np.inf)
simsiam_idx = simsiam_sim.topk(k=1, dim=-1, largest=True)[1].squeeze()

daco_sim = daco_vectors.mm(daco_vectors.t())
daco_sim.fill_diagonal_(-np.inf)
daco_idx = daco_sim.topk(k=1, dim=-1, largest=True)[1].squeeze()

tsne = TSNE(n_components=2, init='pca', random_state=0)

npid_results = tsne.fit_transform(npid_vectors.numpy())
plot_embedding(npid_results, domains.numpy(), labels.numpy(), npid_idx.numpy(), 'npid_{}'.format(data_name))
simclr_results = tsne.fit_transform(simclr_vectors.numpy())
plot_embedding(simclr_results, domains.numpy(), labels.numpy(), simclr_idx.numpy(), 'simclr_{}'.format(data_name))
moco_results = tsne.fit_transform(moco_vectors.numpy())
plot_embedding(moco_results, domains.numpy(), labels.numpy(), moco_idx.numpy(), 'moco_{}'.format(data_name))
simsiam_results = tsne.fit_transform(simsiam_vectors.numpy())
plot_embedding(simsiam_results, domains.numpy(), labels.numpy(), simsiam_idx.numpy(), 'simsiam_{}'.format(data_name))
daco_results = tsne.fit_transform(daco_vectors.numpy())
plot_embedding(daco_results, domains.numpy(), labels.numpy(), daco_idx.numpy(), 'daco_{}'.format(data_name))
