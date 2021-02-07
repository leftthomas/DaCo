import torch.nn.functional as F
from tqdm import tqdm


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    global z
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for data, pos_index, _ in train_bar:
        data = data.to(device_ids[0])
        _, features = net(data)

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


