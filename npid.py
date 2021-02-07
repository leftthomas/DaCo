import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    global z
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for data, pos_index, _ in train_bar:
        data = data.to(device_ids[0])
        _, features = net(data)

        # randomly generate M+1 sample indexes for each batch ---> [B, M+1]
        idx = torch.randint(high=n, size=(data.size(0), m + 1))
        # make the first sample as positive
        idx[:, 0] = pos_index
        # select memory vectors from memory bank ---> [B, 1+M, D]
        samples = torch.index_select(memory_bank, dim=0, index=idx.view(-1)).view(data.size(0), -1, feature_dim)
        # compute cos similarity between each feature vector and memory bank ---> [B, 1+M]
        sim_matrix = torch.bmm(samples.to(device=features.device), features.unsqueeze(dim=-1)).view(data.size(0), -1)
        out = torch.exp(sim_matrix / temperature)
        # Monte Carlo approximation, use the approximation derived from initial batches as z
        if z is None:
            z = out.detach().mean() * n
        # compute P(i|v) ---> [B, 1+M]
        output = out / z

        # compute loss
        # compute log(h(i|v))=log(P(i|v)/(P(i|v)+M*P_n(i))) ---> [B]
        p_d = (output.select(dim=-1, index=0) / (output.select(dim=-1, index=0) + m / n)).log()
        # compute log(1-h(i|v'))=log(1-P(i|v')/(P(i|v')+M*P_n(i))) ---> [B, M]
        p_n = ((m / n) / (output.narrow(dim=-1, start=1, length=m) + m / n)).log()
        # compute J_NCE(θ)=-E(P_d)-M*E(P_n)
        loss = - (p_d.sum() + p_n.sum()) / data.size(0)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # update memory bank ---> [B, D]
        pos_samples = samples.select(dim=1, index=0)
        pos_samples = features.detach().cpu() * momentum + pos_samples * (1.0 - momentum)
        pos_samples = F.normalize(pos_samples, dim=-1)
        memory_bank.index_copy_(0, pos_index, pos_samples)

        total_num += data.size(0)
        total_loss += loss.item() * data.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':
    # args parse
    args = utils.get_opts()
    feature_dim, temperature, batch_size, epochs = args.feature_dim, args.temperature, args.batch_size, args.epochs
    data_path, data_name, m, momentum = args.data_root, args.data_name, args.m, args.momentum
    device_ids = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpu_ids.split(',')]

    # data prepare
    train_loader, test_loader = utils.get_dataset(data_path, data_name, batch_size)

    # model setup and optimizer config
    model, optimizer = utils.get_model_optimizer(feature_dim, device_ids[0])
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids)

    # z as normalizer, init with None, n as num of train data
    z, n = None, len(train_loader.dataset)
    # init memory bank as unit random vector ---> [N, D]
    memory_bank = F.normalize(torch.randn(n, feature_dim), dim=-1)

    # training loop
    results = {'train_loss': []}
    save_name_pre = 'npid_{}_{}_{}_{}_{}_{}_{}'.format(data_name, feature_dim, temperature, batch_size, epochs, m,
                                                       momentum)
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
