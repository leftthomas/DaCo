from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os


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
    def __init__(self, root, train=True):
        super(Alderley, self).__init__()
        if train:
            self.image_file_names = [os.path.join(root, 'train', x) for x in os.listdir(root + '/train') if
                                     is_image_file(x)]
            self.transform = train_transform
        else:
            self.image_file_names = [os.path.join(root, 'val', x) for x in os.listdir(root + '/val') if
                                     is_image_file(x)]
            self.transform = test_transform

    def __getitem__(self, index):
        img_name = self.image_file_names[index]
        img = Image.open(img_name)
        p, n = os.path.split(img_name)
        img_g = Image.open(os.path.join(p, "gen", n.split('.')[0] + '_gen' + n.split('.')[-1]))

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        pos_3 = self.transform(img_g)
        pos_4 = self.transform(img_g)
        return pos_1, pos_2, pos_3, pos_4, img_name

    def __len__(self):
        return len(self.image_file_names)

