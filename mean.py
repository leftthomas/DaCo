import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

search_path = os.path.join('data/tokyo/original/*/train/*.png')
files = glob.glob(search_path)
files.sort()
# BGR
means = [0.0, 0.0, 0.0]
stds = [0.0, 0.0, 0.0]

for image_path in tqdm(files):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = np.asarray(image, np.float32) / 255.0
    for i in range(3):
        means[i] += image[:, :, i].mean()
        stds[i] += image[:, :, i].std()
means = np.asarray(means) / len(files)
stds = np.asarray(stds) / len(files)
print(means, stds)
