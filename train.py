import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from utils import my_optim
from utils.Restore import get_model_para_number
from utils.Restore import get_save_dir
from utils.Restore import save_model
from data.LoadDataSeg import all_data_loader ## data_loader
from utils import NoteLoss
from networks.PPMs import network

np.random.seed(300)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_arguments():
    parser = argparse.ArgumentParser(description='shared-attributes')
    parser.add_argument("--start_step", type=int, default=1)  #
    parser.add_argument("--max_step", type=int, default=201) #
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--snapshot_dir", type=str, default='exp/001')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='voc')
    return parser.parse_args()


def get_model(args):
    model = network(num_heads=29)
    opti_A = my_optim.get_finetune_optimizer(args, model)
    model = model.cuda()
    print('Number of Parameters: %d' % (get_model_para_number(model)))

    return model, opti_A

def train(args):
    train_loader = all_data_loader(args)
    model, optimizer = get_model(args)
    model.train()
    losses = NoteLoss.Loss_total(args)

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))

    print('Start training')
    pbar = tqdm(range(1, args.max_step))

    for epoch in pbar:
        for i, data in enumerate(train_loader):

            my_optim.adjust_learning_rate_poly(args, optimizer, epoch, power=0.9)

            img, mask, part_mask,_ = data
            img, mask, part_mask = img.cuda(), mask.cuda(), part_mask.cuda()

            cos_map = model(img)
            loss = model.get_loss(cos_map, part_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.updateloss(loss)

            model.save_pred(cos_map.detach(), mask.detach(), part_mask.detach(), i, mode=args.mode)

            pbar.set_description("[Epoch %d/%d][Batch %d/%d][Loss: %f][LR: %f]"
                                 % (epoch, args.max_step-1, i, len(train_loader), loss.item(), optimizer.param_groups[0]["lr"]))

        save_model(args, epoch, model, optimizer)

        losses.miou, losses.part_miou = model.get_pred(mode='train')
        save_log_dir = get_save_dir(args)
        log_file = open(os.path.join(save_log_dir, 'log.txt'), 'a')
        losses.logloss(log_file, epoch)
        losses.reset()
        log_file.close()

if __name__ == '__main__':
    args = get_arguments()
    args.snapshot_dir = 'exp/001'
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
