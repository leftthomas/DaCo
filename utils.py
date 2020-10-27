import argparse
import os

import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from model import Model


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Alderley(Dataset):
    def __init__(self, root, train=True, pair=False):
        super(Alderley, self).__init__()

        image_file_names = [os.path.join(root, 'images', x) for x in os.listdir(root + '/images') if is_image_file(x)]
        self.image_file_names = []
        if train:
            for image_name in image_file_names:
                image_id = int(image_name.split('/')[-1].split('.')[0].split('_')[0][5:])
                if 'day' in image_name:
                    if image_id <= 11686:
                        self.image_file_names.append(image_name)
                if 'night' in image_name:
                    if image_id <= 13800:
                        self.image_file_names.append(image_name)
            self.transform = train_transform
        else:
            for image_name in image_file_names:
                image_id = int(image_name.split('/')[-1].split('.')[0].split('_')[0][5:])
                if 'day' in image_name:
                    if image_id > 11686:
                        self.image_file_names.append(image_name)
                if 'night' in image_name:
                    if image_id > 13800:
                        self.image_file_names.append(image_name)
            self.transform = test_transform
        # decide whether using paired images
        self.pair = pair

    def __getitem__(self, index):
        img_name = self.image_file_names[index]
        img = Image.open(img_name)
        image = self.transform(img)
        if self.pair:
            pair_image = self.transform(img)
            return image, pair_image, index, img_name
        else:
            return image, index, img_name

    def __len__(self):
        return len(self.image_file_names)


# TODO
class Seasons(Dataset):
    def __init__(self, root, train=True, pair=False):
        super(Seasons, self).__init__()
        # decide whether using paired images
        self.pair = pair

    def __getitem__(self, index):
        return None

    def __len__(self):
        return None


def get_opts():
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_path', default='/home/data', type=str, help='Datasets path')
    parser.add_argument('--data_name', default='alderley', type=str, choices=['alderley', 'seasons'],
                        help='Dataset name')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    # args for NPID
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.5, type=float, help='Momentum used for the update of memory bank')

    # args parse
    args = parser.parse_args()
    return args


# dataset prepare
def get_dataset(data_path, data_name, batch_size, pair=False):
    if data_name == 'alderley':
        train_data = Alderley(root='{}/{}'.format(data_path, data_name), train=True, pair=pair)
        test_data = Alderley(root='{}/{}'.format(data_path, data_name), train=False)
    else:
        train_data = Seasons(root='{}/{}'.format(data_path, data_name), train=True, pair=pair)
        test_data = Seasons(root='{}/{}'.format(data_path, data_name), train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16)
    return train_loader, test_loader


# model setup and optimizer config
def get_model_optimizer(feature_dim):
    model = Model(feature_dim).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    return model, optimizer
