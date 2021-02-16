import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from utils import DomainDataset


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    colors1 = '#00CED1'
    colors2 = '#DC143C'
    for i in range(data.shape[0]):
        if label[i]:
            colors = colors1
        else:
            colors = colors2
        plt.scatter(data[i, 0], data[i, 1], s=10, c=colors)
    plt.savefig('result/{}.pdf'.format(title), dpi=30, bbox_inches='tight', pad_inches=0)


data_name = 'tokyo'

val_data = DomainDataset('data', data_name, split='val')
npid_vectors = torch.load('result/{}_npid_vectors.pth'.format(data_name)).cpu().numpy()
simclr_vectors = torch.load('result/{}_simclr_vectors.pth'.format(data_name)).cpu().numpy()
moco_vectors = torch.load('result/{}_moco_vectors.pth'.format(data_name)).cpu().numpy()
daco_vectors = torch.load('result/{}_daco_vectors.pth'.format(data_name)).cpu().numpy()

labels = torch.cat((torch.ones(len(simclr_vectors) // 2, dtype=torch.long),
                    torch.zeros(len(simclr_vectors) // 2, dtype=torch.long)), dim=0).cpu().numpy()

tsne = TSNE(n_components=2, init='pca', random_state=0)

npid_results = tsne.fit_transform(npid_vectors)
plot_embedding(npid_results, labels, 'npid_{}'.format(data_name))
simclr_results = tsne.fit_transform(simclr_vectors)
plot_embedding(simclr_results, labels, 'simclr_{}'.format(data_name))
moco_results = tsne.fit_transform(moco_vectors)
plot_embedding(moco_results, labels, 'moco_{}'.format(data_name))
daco_results = tsne.fit_transform(daco_vectors)
plot_embedding(daco_results, labels, 'daco_{}'.format(data_name))
