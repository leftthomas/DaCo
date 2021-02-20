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

data_type = 'SUNSET'
data_seq = '02'

search_path = os.path.join('../../Downloads/original/{}/*/{}_*.png'.format(data_type, data_seq))
files = glob.glob(search_path)
files.sort()
for f in files:
    name = os.path.basename(f)
    l_path = '../../Downloads/SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left/Omni_L/{}'.format(data_seq, data_type, name[3:])
    r_path = '../../Downloads/SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left/Omni_R/{}'.format(data_seq, data_type, name[3:])
    f_path = '../../Downloads/SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left/Omni_F/{}'.format(data_seq, data_type, name[3:])
    b_path = '../../Downloads/SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left/Omni_B/{}'.format(data_seq, data_type, name[3:])
    save_l_path = l_path.replace('SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left'.format(data_seq, data_type),
                                 '{}-{}'.format(data_seq, data_type))
    save_r_path = r_path.replace('SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left'.format(data_seq, data_type),
                                 '{}-{}'.format(data_seq, data_type))
    save_f_path = f_path.replace('SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left'.format(data_seq, data_type),
                                 '{}-{}'.format(data_seq, data_type))
    save_b_path = b_path.replace('SYNTHIA-SEQS-{}-{}/RGB/Stereo_Left'.format(data_seq, data_type),
                                 '{}-{}'.format(data_seq, data_type))
    if not os.path.exists(os.path.dirname(save_l_path)):
        os.makedirs(os.path.dirname(save_l_path))
    if not os.path.exists(os.path.dirname(save_r_path)):
        os.makedirs(os.path.dirname(save_r_path))
    if not os.path.exists(os.path.dirname(save_f_path)):
        os.makedirs(os.path.dirname(save_f_path))
    if not os.path.exists(os.path.dirname(save_b_path)):
        os.makedirs(os.path.dirname(save_b_path))
    shutil.copy(l_path, save_l_path)
    shutil.copy(r_path, save_r_path)
    shutil.copy(f_path, save_f_path)
    shutil.copy(b_path, save_b_path)
