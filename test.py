import os, sys

import numpy as np
from time import time
import SimpleITK as sitk
import torch
import yaml
import argparse

from dataset.Dataset_test import CT2DDatasetTest
from torch.utils.data import DataLoader
from model.UNet2D import UNet2D

checkpoint_dir = '/home/zli17/data/nifti/output-test-unet2d/epoch_60.pth'
prediction_save_dir = '/data/zli17/nifti/output-test-unet2d/testset'
test_file = '/home/zli17/work/projects/VNet/dataset/test_new.txt'

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/zli17/data/nifti/output-test-unet2d/epoch_60.pth'
                        , help='Path for pretrained model')
    parser.add_argument('--prediction_save_dir', type=str, default='/data/zli17/nifti/output-test-unet2d/testset'
                        , help='Path for save result')
    parser.add_argument('--test_file', type=str, default='/home/zli17/work/projects/VNet/dataset/test_new.txt',
                        help='Path for test set')
    parser.add_argument('--config_file', type=str, default='/home/zli17/work/projects/UNet2D/config/template.yaml',
                        help='Path for configuration file')
    parser.add_argument('--num_workers', type=int, default=4, help='Num of the workers [default: 4]')
    return parser.parse_args()

def saveimg(pred_array, path):
    pred_seg = sitk.GetImageFromArray(pred_array)
    sitk.WriteImage(pred_seg, path)

def predict(test_loader, model, args=None, config=None):
    if not os.path.exists(args.prediction_save_dir):
        os.mkdir(args.prediction_save_dir)

    for i_batch, (image_tensor, label_tensor, image_id) in enumerate(test_loader):
        image_tensor = image_tensor.squeeze(0)
        label_tensor = label_tensor.squeeze(0)
        image_tensor = image_tensor.cuda()
        out = model(image_tensor)

        out_arg = out.argmax(dim=1)

        id = image_id[0]
        print(id)
        saveimg(image_tensor.squeeze(1).cpu().detach().numpy()
                , os.path.join(prediction_save_dir, f'{id}_img.nii'))
        saveimg(out_arg.cpu().detach().numpy().astype(np.int8)
                , os.path.join(prediction_save_dir, f'{id}_mask_pred.nii'))
        saveimg(label_tensor.cpu().detach().numpy().astype(np.int8)
                , os.path.join(prediction_save_dir, f'{id}_mask_real.nii'))


def main():
    args = parse_args()
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if not os.path.exists(args.prediction_save_dir):
        os.makedirs(args.prediction_save_dir)

    # load testset
    data_test = CT2DDatasetTest(args=args, config=config)
    test_loader = DataLoader(data_test
                            , batch_size=1
                            , shuffle=False
                            , num_workers=4
                            )

    # load model
    # create model
    print('loading model...')
    model = UNet2D(in_dim=config.get('model').get('in_dim')).cuda()
    model.load_state_dict(torch.load(args.checkpoint_dir))
    model.eval()

    predict(test_loader, model, args=args, config=config)


if __name__ == '__main__':
    main()