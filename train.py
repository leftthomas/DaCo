import argparse

import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model
from utils import DomainDataset

# for reproducibility
np.random.seed(0)
torch.manual_seed(0)


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for pos_1, pos_2, _, __ in train_bar:
        pos_1, pos_2 = pos_1.to(device_ids[0]), pos_2.to(device_ids[0])
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.size(0), device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.size(0), -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += pos_1.size(0)
        total_loss += loss.item() * pos_1.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch to obtain the test vectors
def test(net, test_data_loader, device_id):
    net.eval()
    image_names, feature_bank, feature_vectors = [], [], {}
    with torch.no_grad():
        # generate feature bank
        for data, _, image_name in tqdm(test_data_loader, desc='Feature extracting', dynamic_ncols=True):
            image_names += image_name
            feature_bank.append(net(data.to(device_id))[0])
        feature_bank = torch.cat(feature_bank, dim=0)
    for index in range(len(image_names)):
        feature_vectors[image_names[index].split('/')[-1]] = feature_bank[index]
    return feature_vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='dnim', type=str, choices=['dnim', 'cityscapes', 'alderley'],
                        help='Dataset name')
    parser.add_argument('--method_name', default='daco', type=str, choices=['daco', 'simclr', 'moco', 'npid'],
                        help='Dataset name')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu_ids', nargs='+', type=int, required=True, help='Selected gpus to train')
    # args for DaCo
    parser.add_argument('--lamda', default=0.9, type=float, help='Lambda used for the weight of soft constrain')
    # args for NPID
    parser.add_argument('--negs', default=4096, type=int, help='Negative sample number')
    # args for MoCo
    parser.add_argument('--momentum', default=0.5, type=float, help='Momentum used for the update of memory bank')

    # args parse
    args = parser.parse_args()
    data_root, data_name, method_name, gpu_ids = args.data_root, args.data_name, args.method_name, args.gpu_ids
    feature_dim, temperature, batch_size, epochs = args.feature_dim, args.temperature, args.batch_size, args.epochs
    lamda, negs, momentum = args.lamda, args.negs, args.momentum

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    # model setup and optimizer config
    model = Model(feature_dim).cuda(gpu_ids[0])
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    if len(gpu_ids) > 1:
        model = DataParallel(model, device_ids=gpu_ids)
