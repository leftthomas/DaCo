import os

import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from tqdm import tqdm

import utils

# for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


if __name__ == '__main__':

    # args parse
    args = utils.get_opts()
    feature_dim, temperature, batch_size, epochs = args.feature_dim, args.temperature, args.batch_size, args.epochs
    data_path, data_name = args.data_path, args.data_name
    device_ids = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpu_ids.split(',')]

    # data prepare
    train_loader, test_loader = utils.get_dataset(data_path, data_name, batch_size, True)

    # model setup and optimizer config
    model, optimizer = utils.get_model_optimizer(feature_dim, device_ids[0])
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids)

    # training loop
    results = {'train_loss': []}
    save_name_pre = 'simclr_{}_{}_{}_{}_{}'.format(data_name, feature_dim, temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        if epoch % 10 == 0:
            test_vectors = utils.test(model, test_loader, device_ids[0])
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('results/{}_results.csv'.format(save_name_pre), index_label='epoch')
            if len(device_ids) > 1:
                torch.save(model.module.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
            else:
                torch.save(model.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
            torch.save(test_vectors, 'results/{}_{}_vectors.pth'.format(save_name_pre, epoch))
