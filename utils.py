import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# TODO
normalizer = {'dnim': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
              'cityscapes': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
              'alderley': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]}


def get_transform(data_name, split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        original_path = os.path.join(data_root, data_name, 'original', '*', split, '*.png')
        self.original_images = glob.glob(original_path)
        self.original_images.sort()

        generated_path = os.path.join(data_root, data_name, 'generated', '*', split, '*.png')
        self.generated_images = glob.glob(generated_path)
        self.generated_images.sort()

        self.transform = get_transform(data_name, split)

    def __getitem__(self, index):
        original_img_name = self.original_images[index]
        original_img = Image.open(original_img_name)
        original_img_1 = self.transform(original_img)
        original_img_2 = self.transform(original_img)
        generated_img_name = self.generated_images[index]
        generated_img = Image.open(generated_img_name)
        generated_img_1 = self.transform(generated_img)
        generated_img_2 = self.transform(generated_img)
        assert original_img_name.split('/')[-3] == generated_img_name.split('/')[-3]
        domain = original_img_name.split('/')[-3]
        assert os.path.basename(original_img_name) == os.path.basename(generated_img_name)
        img_name = os.path.basename(original_img_name)
        return original_img_1, original_img_2, generated_img_1, generated_img_2, domain, img_name, index

    def __len__(self):
        return len(self.original_images)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj_1, proj_2):
        batch_size = proj_1.size(0)
        # [2*B, Dim]
        out = torch.cat([proj_1, proj_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(proj_1 * proj_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


class MoCoLoss(nn.Module):
    def __init__(self, negs, proj_dim, temperature):
        super(MoCoLoss, self).__init__()
        # init memory queue as unit random vector ---> [Negs, Dim]
        self.queue = F.normalize(torch.randn(negs, proj_dim), dim=-1)
        self.temperature = temperature

    def forward(self, query, key):
        batch_size = query.size(0)
        # [B, 1]
        score_pos = torch.sum(query * key, dim=-1, keepdim=True)
        # [B, Negs]
        score_neg = torch.mm(query, self.queue.t().contiguous())
        # [B, 1+Negs]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / self.temperature, torch.zeros(batch_size, dtype=torch.long, device=query.device))
        return loss

    def enqueue(self, key):
        # update queue
        self.queue = torch.cat((self.queue, key), dim=0)[key.size(0):]


class NPIDLoss(nn.Module):
    def __init__(self, n, negs, proj_dim, momentum, temperature):
        super(NPIDLoss, self).__init__()
        self.n = n
        self.negs = negs
        self.proj_dim = proj_dim
        self.momentum = momentum
        self.temperature = temperature
        # init memory bank as unit random vector ---> [N, Dim]
        self.bank = F.normalize(torch.randn(n, proj_dim), dim=-1)
        # z as normalizer, init with None
        self.z = None

    def forward(self, proj, pos_index):
        batch_size = proj.size(0)
        # randomly generate Negs+1 sample indexes for each batch ---> [B, Negs+1]
        idx = torch.randint(high=self.n, size=(batch_size, self.negs + 1))
        # make the first sample as positive
        idx[:, 0] = pos_index
        # select memory vectors from memory bank ---> [B, 1+Negs, Dim]
        samples = torch.index_select(self.bank, dim=0, index=idx.view(-1)).view(batch_size, -1, self.proj_dim)
        # compute cos similarity between each feature vector and memory bank ---> [B, 1+Negs]
        sim_matrix = torch.bmm(samples.to(device=proj.device), proj.unsqueeze(dim=-1)).view(batch_size, -1)
        out = torch.exp(sim_matrix / self.temperature)
        # Monte Carlo approximation, use the approximation derived from initial batches as z
        if self.z is None:
            self.z = out.detach().mean() * self.n
        # compute P(i|v) ---> [B, 1+Negs]
        output = out / self.z

        # compute loss
        # compute log(h(i|v))=log(P(i|v)/(P(i|v)+Negs*P_n(i))) ---> [B]
        p_d = (output.select(dim=-1, index=0) / (output.select(dim=-1, index=0) + self.negs / self.n)).log()
        # compute log(1-h(i|v'))=log(1-P(i|v')/(P(i|v')+Negs*P_n(i))) ---> [B, Negs]
        p_n = ((self.negs / self.n) / (output.narrow(dim=-1, start=1, length=self.negs) + self.negs / self.n)).log()
        # compute J_NCE(θ)=-E(P_d)-Negs*E(P_n)
        loss = - (p_d.sum() + p_n.sum()) / batch_size

        pos_samples = samples.select(dim=1, index=0)
        return loss, pos_samples

    def enqueue(self, proj, pos_index, pos_samples):
        # update memory bank ---> [B, Dim]
        pos_samples = proj.detach().cpu() * self.momentum + pos_samples * (1.0 - self.momentum)
        pos_samples = F.normalize(pos_samples, dim=-1)
        self.bank.index_copy_(0, pos_index, pos_samples)
