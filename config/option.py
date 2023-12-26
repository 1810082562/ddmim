import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Pre-training')
    parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet2012',
                    help='path to dataset (default: imagenet)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=52, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
    parser.add_argument('--resume', default='ddmim/log/seed3407/version_30/checkpoints/last.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--lpresume', default='ddmim/log/seed3407/version_14/checkpoints/last.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--seed', default=3407, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
    parser.add_argument('--log_dir', metavar='DIR', nargs='?', default='ddmim/log',
                    help='path to log (default: ddmim/log)')
    
    return parser.parse_args()