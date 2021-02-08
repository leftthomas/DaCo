import glob
import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

normalizer = {'dnim': [(0.361, 0.337, 0.315), (0.191, 0.186, 0.177)],
              'cityscapes': [(0.223, 0.241, 0.222), (0.061, 0.062, 0.062)],
              'alderley': [(0.361, 0.374, 0.330), (0.206, 0.196, 0.189)]}


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
        domain = original_img_name.split('/')[-3] == 'domain_a'
        img_name = os.path.basename(original_img_name)
        return original_img_1, original_img_2, generated_img_1, generated_img_2, domain, img_name, index

    def __len__(self):
        return len(self.original_images)


def metrics_dnim(names, domains, vectors):
    labels = torch.zeros(len(names), dtype=torch.long, device=vectors.device)
    for i, name in enumerate(names):
        labels[i] = int(name.split('_')[0])
    domain_a_vectors = vectors[domains, :]
    domain_b_vectors = vectors[~domains, :]
    domain_a_labels = labels[domains]
    domain_b_labels = labels[~domains]
    # domain a ---> domain b
    sim = torch.mm(domain_a_vectors, domain_b_vectors.t().contiguous())
    idx = torch.argmax(sim, dim=-1)
    precise_ab = torch.eq(domain_b_labels[idx], domain_a_labels).float().mean()
    # domain b ---> domain a
    sim = torch.mm(domain_b_vectors, domain_a_vectors.t().contiguous())
    idx = torch.argmax(sim, dim=-1)
    precise_ba = torch.eq(domain_a_labels[idx], domain_b_labels).float().mean()
    precise = (precise_ab + precise_ba) / 2
    return precise_ab.item(), precise_ba.item(), precise.item()
