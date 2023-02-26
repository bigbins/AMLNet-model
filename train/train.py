from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys


import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.function import train_mixup,mixup_data,mixup_criterion,LabelSmoothingCrossEntropy, EarlyStopping
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

from dataset.transforms import build_transforms
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    args = parser.parse_args()
    print(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    update_config(config, args)
    print(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # Model
    model = eval('models.'+'efficient'+'.get_net')(
        config)



    in_channels = 1 if config.DATASET.GRAY else 3
    dump_input = torch.rand(
        (1, in_channels, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))
    logger.info(print(model))


    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer

    criterion = LabelSmoothingCrossEntropy().cuda()

    patience = 5  
    early_stopping = EarlyStopping(patience, verbose=True)

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False

    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)


    # Data loading code
    print(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)

    train_transforms = build_transforms(config, True)
    val_transforms = build_transforms(config, False)

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform=val_transforms),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    es = 0
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        # train for one epoch
        # cur_lr = optimizer.param_groups[-1]['lr']
        train_mixup(config, train_loader, model, criterion, optimizer, epoch,
                    final_output_dir, tb_log_dir, writer_dict)
        lr_scheduler.step()

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
            es = 0

        else:
            es += 1
            print("Counter {} of 5".format(es))
            best_model = False
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

        print('现在的patience是：',es)
        print('---------------------')
        print('---------------------')
        print('---------------------')
        print('---------------------')
        if es > patience:
            print("Early stopping with best_acc: ", best_perf, "and val_acc for this epoch: ", perf_indicator, "...")
            break

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
