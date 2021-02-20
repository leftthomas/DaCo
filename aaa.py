import glob
import os
import shutil

# search_path = os.path.join('../../Downloads/original/domain_a/*/*.png')
# files = glob.glob(search_path)
# files.sort()
#
# tt = glob.glob(os.path.join('../../Downloads/original/domain_b/*/*.png'))
# tt.sort()
# for i, f in enumerate(tt):
#     shutil.move(f, '{}/{}'.format(os.path.dirname(f), os.path.basename(files[i])))

data_seq = '05'

sun_path = os.path.join('../../Downloads/{}-SUNSET/*/*.png'.format(data_seq))
sun_files = glob.glob(sun_path)
sun_files.sort()

rain_path = os.path.join('../../Downloads/{}-RAINNIGHT/*/*.png'.format(data_seq))
rain_files = glob.glob(rain_path)
rain_files.sort()

for sun_file, rain_file in zip(sun_files, rain_files):
    assert sun_file.split('/')[-2] == rain_file.split('/')[-2]
    direct = sun_file.split('/')[-2][-1]
    name = os.path.basename(sun_file)
    save_sun = '../../Downloads/synthia/original/domain_a/train/sun_{}_{}_{}'.format(data_seq, direct, name)
    save_rain = '../../Downloads/synthia/original/domain_b/train/rain_{}_{}_{}'.format(data_seq, direct, name)
    if not os.path.exists(os.path.dirname(save_sun)):
        os.makedirs(os.path.dirname(save_sun))
    if not os.path.exists(os.path.dirname(save_rain)):
        os.makedirs(os.path.dirname(save_rain))
    shutil.copy(sun_file, save_sun)
    shutil.copy(rain_file, save_rain)
