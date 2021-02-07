import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
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
        return original_img_1, original_img_2, generated_img_1, generated_img_2, domain, img_name

    def __len__(self):
        return len(self.original_images)


def simclr_loss(proj_1, proj_2, temperature):
    batch_size = proj_1.size(0)
    # [2*B, Dim]
    out = torch.cat([proj_1, proj_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(proj_1 * proj_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def moco_loss(query, key, queue, temperature):
    batch_size = query.size(0)
    # [B, 1]
    score_pos = torch.sum(query * key, dim=-1, keepdim=True)
    # [B, N]
    score_neg = torch.mm(query, queue.t().contiguous())
    # [B, 1+M]
    out = torch.cat([score_pos, score_neg], dim=-1)
    # compute loss
    loss = F.cross_entropy(out / temperature, torch.zeros(batch_size, dtype=torch.long, device=query.device))
    return loss
