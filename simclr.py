import os

import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel

import utils

# for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    # args parse
    args = utils.get_opts()
    feature_dim, temperature, batch_size, epochs = args.feature_dim, args.temperature, args.batch_size, args.epochs
    data_path, data_name = args.data_root, args.data_name
    device_ids = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpu_ids.split(',')]

    # data prepare
    train_loader, test_loader = utils.get_dataset(data_path, data_name, batch_size, True)

    # model setup and optimizer config
    model, optimizer = utils.get_model_optimizer(feature_dim, device_ids[0])
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids)

    # training loop
    results = {'train_loss': []}
    save_name_pre = 'simclr_{}_{}_{}_{}_{}'.format(data_name, feature_dim, temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        if epoch % 10 == 0:
            test_vectors = utils.test(model, test_loader, device_ids[0])
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('results/{}_results.csv'.format(save_name_pre), index_label='epoch')
            if len(device_ids) > 1:
                torch.save(model.module.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
            else:
                torch.save(model.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre, epoch))
            torch.save(test_vectors, 'results/{}_{}_vectors.pth'.format(save_name_pre, epoch))
