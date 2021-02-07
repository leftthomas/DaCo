import glob
import os
import shutil
import sys
from multiprocessing.dummy import Pool as ThreadPool

progress = 0
search_path = os.path.join('data/dnim/*/*.png')
files = glob.glob(search_path)
files.sort()


def processing_data(image_path):
    label, name = image_path.split('/')[-2], image_path.split('/')[-1].split('.')[0]
    label_path = os.path.join('data/DNIM/time_stamp/{}.txt'.format(label))
    data_type
    domain_type
    target_path = 'data/dnim/{}_{}.png'.format(label, image_path.split('/')[-1].split('.')[0])
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path))
    shutil.move(image_path, target_path)
    global progress
    progress += 1
    print("\rProgress: {:>3} %".format(progress * 100 / len(files)), end=' ')
    sys.stdout.flush()


print('\nProcessing {} images'.format(len(files)))
print("Progress: {:>3} %".format(progress * 100 / len(files)), end=' ')
pool = ThreadPool()
pool.map(processing_data, files)
pool.close()
pool.join()
