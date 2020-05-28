import os, sys
import argparse
import logging
import yaml
from utils import logger_utils
from shutil import copy
import torch

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--epoch',  default=100, type=int, help='Epoch to run [default: 20]')
    parser.add_argument('--batch_size', default=64, type=int, help='Batchsize for each update [default: 64]')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--save_dir', type=str, default='/data/zli17/nifti/output-test-unet2d-test',help='Path for saving model')
    parser.add_argument('--train_file', type=str, default='/home/zli17/data/nifti/split2d_train/img_slice_list.txt',
                        help='Path for training set')
    parser.add_argument('--val_file', type=str, default='/home/zli17/work/projects/VNet/dataset/val_new.txt',
                        help='Path for validation set')
    parser.add_argument('--config_file', type=str, default='/home/zli17/work/projects/UNet2D/config/template.yaml',
                        help='Path for configuration file')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=50, help='Decay step for lr decay [default: every 200 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.3]')
    parser.add_argument('--num_workers', type=int, default=2, help='Num of the workers [default: 4]')
    parser.add_argument('--verbose', default=False, action="store_true", help="print information")
    parser.add_argument('--debug', default=False, action="store_true", help="Debug mode")
    return parser.parse_args()

def _load_config_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_config():
    args = parse_args()
    config = _load_config_yaml(args.config_file)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ## setup logging
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logger = logger_utils.set_logger('main', level=log_level, log_path=os.path.join(args.save_dir, 'main.log'))
    logger.info('setup logger successfully!')
    ## record command
    logger.info('recording command...')
    with open(os.path.join( args.save_dir,'command.txt'), 'w') as f:
        f.write(" ".join(sys.argv) + "\n")

    ## copy config file
    logger.info('backuping configuration file')
    copy(args.config_file, os.path.join(args.save_dir, 'config.yaml'))

    ## setup device
    # Get a device to train on
    device_str = args.device
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device

    return args, config

if __name__ == '__main__':
    args, config = load_config()
    print(config.get('data').get('spacing'))