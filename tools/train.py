from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.models import get_net
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils.autoanchor import check_anchors

import lib.dataset
import lib.models


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # philly
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
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    args = parser.parse_args()

    return args


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    model = get_net(cfg)
    # DP mode
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # DDP mode
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    device = select_device(logger, batch_size=cfg.BATCH_SIZE)
    # if args.local_rank != -1:
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device('cuda', opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    #     opt.batch_size = opt.total_batch_size // opt.world_size

    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, True,  transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, False, transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device).cuda()
    optimizer = get_optimizer(cfg, model)

    # load checkpoint model
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    else:
        if cfg.NEED_AUTOANCHOR:
            check_anchors(train_dataset, model=model, imgsz=min(cfg.MODEL.IMAGE_SIZE))

    # training
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch, writer_dict, rank)
        
        lr_scheduler.step()

        # evaluate on validation set
        if epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH+1 and rank in [-1, 0]:
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict
            )

            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            # save checkpoint model and best model
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    # save final model
    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()