import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_vision', default=1e-5, type=float)
    parser.add_argument('--lr_language', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--clip_max_norm', default=0., type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--lr_drop', nargs="+", default=[45, 58])
    
    # Augmentation options
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # Architecture options
    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--bert_model', default='roberta-base', type=str, help='bert model')
    parser.add_argument('--vit_model', default='small', type=str, help='vit model')
    parser.add_argument('--separate_qkv', action='store_true')
    parser.add_argument('--reg_out_type', default='reg_input', type=str, 
                        help='option for output regression source feature')
    parser.add_argument('--language_modulation', type=str, default='cross_attn',
                        help='language_modulation should be one of ["cross_attn", "concat_linear", "cls_token"]')
    parser.add_argument('--num_modulation', type=int, default=4, help='number of v-l blocks')
    parser.add_argument('--modulate_in_last_blocks', action='store_true')
    parser.add_argument('--reg_token_in_last_blocks', action='store_true')
    parser.add_argument('--without_visual_mask', action='store_true')
    parser.add_argument('--num_vpt', type=int, default=0)

    # Language Branch
    # parser.add_argument('--language_prompt_tuning', action='store_true')
    parser.add_argument('--language_frozen_embedding', action='store_true')
    parser.add_argument('--langauge_frozen_encoder', action='store_true')
    parser.add_argument('--max_query_len', default=20, type=int,
                            help='maximum time steps (lang length) per batch')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/unc/unc+/gref/gref_umd')

    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--visual_pretrained', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--pretrained_lm_path', default='../Language-Models', type=str, 
                        help='root path at which we save language models')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    visu_param = [p for n, p in model_without_ddp.named_parameters() if (("visual_branch" in n) and p.requires_grad)]
    text_param = [p for n, p in model_without_ddp.named_parameters() if (("linguistic_branch" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visual_branch" not in n) and ("linguistic_branch" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param},
                  {"params": visu_param, "lr": args.lr_vision},
                  {"params": text_param, "lr": args.lr_language},
                 ]

    # visu_param = [p for n, p in model_without_ddp.named_parameters() if "visumodel" in n and p.requires_grad]
    # text_param = [p for n, p in model_without_ddp.named_parameters() if "textmodel" in n and p.requires_grad]
    # rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]
    
    frozen_params = [n for n, p in model_without_ddp.named_parameters() if not p.requires_grad]
    print('Frozen parameters', frozen_params)
    
    # using AdamW optimizer and MultiStepLr scheduler
    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    # using polynomial lr scheduler or half decay every 10 epochs or step


    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val   = build_dataset('val', args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.visual_pretrained is not None:
        checkpoint = torch.load(args.visual_pretrained, map_location='cpu')
        # if use sparse vit, split qkv.weight/bias into q.weight/bias and kv.weight/bias
        if args.separate_qkv:
            pretrained_param_dict = {}
            for key in checkpoint['model'].keys():
                if 'qkv' in key:
                    num_out = checkpoint['model'][key].shape[0]
                    num_q = int(num_out // 3)
                    if 'weight' in key:
                        pretrained_param_dict[key.replace('qkv', 'q_in')] = checkpoint['model'][key][:num_q, :]
                        pretrained_param_dict[key.replace('qkv', 'k_in')] = checkpoint['model'][key][num_q:2*num_q, :]
                        pretrained_param_dict[key.replace('qkv', 'v_in')] = checkpoint['model'][key][2*num_q:, :]
                    elif 'bias' in key:
                        pretrained_param_dict[key.replace('qkv', 'q_in')] = checkpoint['model'][key][:num_q]
                        pretrained_param_dict[key.replace('qkv', 'k_in')] = checkpoint['model'][key][num_q:2*num_q]
                        pretrained_param_dict[key.replace('qkv', 'v_in')] = checkpoint['model'][key][2*num_q:]

                    # pretrained_param_dict[key.replace('qkv', 'q')] = checkpoint['model'][key]
                    # pretrained_param_dict[key.replace('qkv', 'kv')] = checkpoint['model'][key]
                else:
                    pretrained_param_dict[key] = checkpoint['model'][key]
        else:
            pretrained_param_dict = checkpoint['model']

        missing_keys, unexpected_keys = model_without_ddp.visual_branch.load_state_dict(pretrained_param_dict, strict=False)
        print('Missing keys when loading visual model:')
        print(missing_keys)
        print('Unexpected keys when loading visual model:')
        print(unexpected_keys)

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    start_time = time.time()
    best_accu = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        val_stats = validate(args, model, data_loader_val, device)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop
            if (epoch + 1) == args.lr_drop:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                best_accu = val_stats['accu']
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
