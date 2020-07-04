import os, sys
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchsummary import  summary

from config.config import load_config
from utils import logger_utils
from model.UNet2D import UNet2D
from dataset.Dataset_train import CT2DDatasetTrain
from dataset.Dataset_val import CT2DDatasetVal
from loss.Dice import DiceLoss
from utils import metric

logger = None

def train_per_epoch(epoch_idx, train_loader, model, loss_func, opt, args=None, config=None):
    loss_l = []
    running_loss = 0.

    dice_c1_l = []
    dice_c2_l = []
    running_dice_c1 = 0.
    running_dice_c2 = 0.

    time_now = time.time()
    for i_batch, (image_tensor, label_tensor) in enumerate(train_loader):
        opt.zero_grad()
        image_tensor = image_tensor.to(config.get('device'))
        #logger.debug(f'image_tensor shape: {image_tensor.shape}')
        out = model(image_tensor)

        loss = loss_func(out, label_tensor.long().to(config.get('device')))
        loss_value = loss.data.cpu().numpy()
        loss_l.append(loss_value)
        running_loss += loss_value
        loss.backward()
        opt.step()

        #compute dice for batch
        out_arg = out.argmax(dim=1)
        logger.debug(f'label_tensor shape: {label_tensor.shape}')
        logger.debug(f'out_arg_shape: {out_arg.shape}')

        c1_dice, c2_dice = metric.dice_2d(label_tensor.detach().cpu().numpy(), out_arg.detach().cpu().numpy())
        dice_c1_l.append(c1_dice)
        dice_c2_l.append(c2_dice)
        running_dice_c1 += np.mean(c1_dice)
        running_dice_c2 += np.mean(c2_dice)

        if i_batch % args.step_size == 0 and i_batch != 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            logger.info('Train, Epoch : {}, Step : {}/{}, Loss : {:.4f}, Dice : ({:.4f}, {:.4f}) '
                         'Run Time : {:.2f} sec'
                         .format(epoch_idx, i_batch, len(train_loader), running_loss/args.step_size
                                 , running_dice_c1/args.step_size
                                 , running_dice_c2/args.step_size
                                 , time_spent))
            running_loss = 0.
            running_dice_c1 = 0.
            running_dice_c2 = 0.
        #break
    train_dice_c1 = np.mean(dice_c1_l)
    train_dice_c2 = np.mean(dice_c2_l)
    logger.info('Train, Epoch : {}, Loss : {:.4f}, Dice : ({:.4f}, {:.4f})'
                .format(epoch_idx, np.mean(loss_l), train_dice_c1, train_dice_c2)
                )
    return train_dice_c1, train_dice_c2

def val_per_epoch(epoch_idx, val_loader, model, loss_func, opt, args=None, config=None):
    loss_l = []
    running_loss = 0.

    dice_c1_l = []
    dice_c2_l = []
    running_dice_c1 = 0.
    running_dice_c2 = 0.
    for i_batch, (image_tensor, label_tensor) in enumerate(val_loader):
        image_tensor = image_tensor.squeeze(0)
        label_tensor = label_tensor.squeeze(0)
        image_tensor = image_tensor.to(config.get('device'))
        #logger.debug(f'image_tensor shape: {image_tensor.shape}')
        out = model(image_tensor)

        loss = loss_func(out, label_tensor.long().to(config.get('device')))
        loss_value = loss.data.cpu().numpy()
        loss_l.append(loss_value)

        #compute dice for batch
        out_arg = out.argmax(dim=1)

        c1_dice, c2_dice = metric.dice_2d(label_tensor.detach().cpu().numpy(), out_arg.detach().cpu().numpy())
        dice_c1_l.append(c1_dice)
        dice_c2_l.append(c2_dice)

    val_dice_c1 = np.mean(dice_c1_l)
    val_dice_c2 = np.mean(dice_c2_l)
    logger.info('Val, Epoch : {}, Loss : {:.4f}, Dice : ({:.4f}, {:.4f})'
                .format(epoch_idx, np.mean(loss_l), val_dice_c1, val_dice_c2)
                )
    return val_dice_c1, val_dice_c2

def main():
    args, config = load_config()

    global logger
    logger = logger_utils.get_logger('main')

    # create model
    logger.info('creating model...')
    model = UNet2D(config.get('model').get('in_dim')).to(config.get('device'))
    if args.verbose:
        summary(model, (1, 256, 256))

    # setup dataset
    data_train = CT2DDatasetTrain(args=args, config=config)
    train_loader = DataLoader(data_train
                              , batch_size=args.batch_size
                              , shuffle=True
                              , num_workers=4
                              )

    data_val = CT2DDatasetVal(args=args, config=config)
    val_loader = DataLoader(data_val
                              , batch_size=1
                              , shuffle=False
                              , num_workers=4
                              )

    # loss function
    loss_func = DiceLoss(args=args, config=config)

    # optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    logger.info(f'Total epoch: {args.epoch}')
    start_epoch = 0
    for epoch in range(start_epoch, args.epoch):

        # train epoch
        model.train()
        train_dice_c1, train_dice_c2 = train_per_epoch(epoch, train_loader, model, loss_func, opt, args=args, config=config)

        # val epoch
        model.eval()
        val_dice_c1, val_dice_c2 = val_per_epoch(epoch, val_loader, model, loss_func, opt
                                                 , args=args
                                                 , config=config)

        ## save model
        state = {
            'epoch': epoch,
            'Left_Lung_dice': val_dice_c1,
            'Right_Lung_dice': val_dice_c2,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}.pth'.format(epoch)))



if __name__ == '__main__':
    main()