from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import six
import os
import os.path as osp
import math
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_data_path', nargs='?', type=str, metavar='PATH', required=True)
parser.add_argument('--train_gt_path', nargs='?', type=str, metavar='PATH', required=True)

parser.add_argument('--target_data_path', nargs='?', type=str, metavar='PATH', required=True)
parser.add_argument('--target_gt_path', nargs='?', type=str, metavar='PATH', required=True)

parser.add_argument('--val_data_path', nargs='?', type=str, metavar='PATH', required=False)
parser.add_argument('--val_gt_path', nargs='?', type=str, metavar='PATH', required=False)

parser.add_argument('--checkpoint', nargs='?', type=str, metavar='PATH', required=True,
                    default='./checkpoint/', help='path to save folder')
parser.add_argument('--log', nargs='?', type=str, metavar='PATH', required=True,
                    default='./log/', help='path to save log')
parser.add_argument('--fold', nargs='?', type=str, metavar='PATH', required=True,
                    default='fold/', help='path to save each folder')

parser.add_argument('-b', '--batch_size', type=int, default=15)
parser.add_argument('--lr', type=float, default=1,
                    help="learning rate of new parameters, for pretrained "
                         "parameters it is 10 times smaller than this")
parser.add_argument('--num_workers', type=int, default=2,
                    help='dataloader worker number')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--pretrained_model_path', type=str, default='', metavar='PATH')




def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args


