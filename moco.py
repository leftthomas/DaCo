import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from tqdm import tqdm

import utils
from model import Model

# for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for x_q, x_k, _, __ in train_bar:
        x_q, x_k = x_q.to(device_ids[0]), x_k.to(device_ids[0])
        _, query = encoder_q(x_q)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _, key = encoder_k(x_k[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':

    # args parse
    args = utils.get_opts()
    feature_dim, temperature, batch_size, epochs = args.feature_dim, args.temperature, args.batch_size, args.epochs
    data_path, data_name, m, momentum = args.data_path, args.data_name, args.m, args.momentum
    device_ids = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpu_ids.split(',')]

    # data prepare
    train_loader, test_loader = utils.get_dataset(data_path, data_name, batch_size, True)

    # model setup and optimizer config
    model_q, optimizer = utils.get_model_optimizer(feature_dim, device_ids[0])
    model_k = Model(feature_dim)
    model_k.to(device_ids[0])
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    if len(device_ids) > 1:
        model_q = DataParallel(model_q, device_ids=device_ids)
        model_k = DataParallel(model_k, device_ids=device_ids)

    # init memory queue as unit random vector ---> [M, D]
    memory_queue = F.normalize(torch.randn(m, feature_dim).to(device_ids[0]), dim=-1)

    # training loop
    results = {'train_loss': []}
    save_name_pre = 'moco_{}_{}_{}_{}_{}_{}_{}'.format(data_name, feature_dim, temperature, batch_size, epochs, m,
                                                       momentum)
    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(1, epochs + 1):
        train_loss = train(model_q, model_k, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        if epoch % 10 == 0:
            test_vectors = utils.test(model_q, test_loader, device_ids[0])
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('results/{}_results.csv'.format(save_name_pre), index_label='epoch')
            if len(device_ids) > 1:
                torch.save(model_q.module.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
            else:
                torch.save(model_q.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
            torch.save(test_vectors, 'results/{}_{}_vectors.pth'.format(save_name_pre, epoch))
