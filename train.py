import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import FSRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_rr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-folder', type=str, required=True)
    # parser.add_argument('--eval-folder', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    train_LR_phiG_folder = '/home/vicky/soapy/Dataset_254x254/Train/LR_phigrad/'
    train_HR_phi_folder = '/home/vicky/soapy/Dataset_254x254/Train/HR_phi/'
    
    val_LR_phiG_folder = '/home/vicky/soapy/Dataset_254x254/Val/LR_phigrad/'
    val_HR_phi_folder = '/home/vicky/soapy/Dataset_254x254/Val/HR_phi/'
    
    
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = FSRCNN(scale_factor=args.scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(train_LR_phiG_folder, train_HR_phi_folder)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    
    eval_dataset = EvalDataset(val_LR_phiG_folder, val_HR_phi_folder)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_rr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_rr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            epoch_rr.update(calc_rr(preds, labels), len(inputs))

        print('eval rr: {}'.format(epoch_rr.avg))

        if epoch_rr.avg > best_rr:
            best_epoch = epoch
            best_rr = epoch_rr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, relative error: {:.2f}'.format(best_epoch, best_rr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
