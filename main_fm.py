import torch
import sys
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataset.dataset import custom_dataset, valid_dataset, IC13_dataset
# from model_resnet import EAST  # resnet
from models.model_fm_cls import EAST #vgg16
from loss import Loss, DiceLoss
from utils.logger_util import Logger
from utils.utils import AverageMeter, get_learning_rate
import os
import time
import random
import numpy as np
import copy
import warnings
from tensorboardX import SummaryWriter
from config import get_args
from datetime import datetime

global_args = get_args(sys.argv[1:])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensorboardX writer
writer = SummaryWriter(comment=global_args.fold)
# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# 输入数据维度和类型上变化不大时，这个设置可以增加运行效率
# torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# create folder
if not os.path.exists(global_args.checkpoint):
    os.mkdir(global_args.checkpoint)
if not os.path.exists(global_args.log):
    os.mkdir(global_args.log)


save_folder = os.path.join(global_args.checkpoint, global_args.fold)
log_folder = os.path.join(global_args.log, global_args.fold)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

# log
log = Logger()
log.open(os.path.join(log_folder, 'log_train.txt'), mode='a')
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))


m = torch.nn.Sigmoid().to(device)

def train(source_train_loader, target_train_loader, model, criterion, loss_domain, optimizer, epoch):
    epoch_loss = AverageMeter()
    east_total_loss = AverageMeter()
    source_domain_loss = AverageMeter()
    target_domain_loss = AverageMeter()

    model.train()
    epoch_time = time.time()

    target_train_iter = iter(target_train_loader)


    for i, (s_img, s_gt_score, s_gt_geo, s_ignored_map) in enumerate(source_train_loader):
        start_time = time.time()

        try:
            t_img, t_gt_score, t_gt_geo, t_ignored_map = next(target_train_iter)
        except StopIteration:
            target_train_iter = iter(source_train_loader)
            t_img, t_gt_score, t_gt_geo, t_ignored_map = next(target_train_iter)

        s_img, s_gt_score, s_gt_geo, s_ignored_map = s_img.to(device), s_gt_score.to(device), s_gt_geo.to(
            device), s_ignored_map.to(device)

        pred_score, pred_geo, pred_cls = model(s_img, False)

        # source label
        domain_s = Variable(torch.zeros(pred_cls.size()).float().cuda())
        loss_domain_s = loss_domain(m(pred_cls), domain_s)

        target_cls = model(t_img, True)
        # target label
        domain_t = Variable(torch.ones(pred_cls.size()).float().cuda())
        loss_domain_t = loss_domain(m(target_cls), domain_t)

        east_loss = criterion(s_gt_score, pred_score, s_gt_geo, pred_geo, s_ignored_map)
        loss = east_loss + 0.1*loss_domain_s + 0.1*loss_domain_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        epoch_loss.update(loss.item(), s_img.size(0))
        east_total_loss.update(east_loss.item(), s_img.size(0))
        source_domain_loss.update(loss_domain_s.item(), s_img.size(0))
        target_domain_loss.update(loss_domain_t, s_img.size(0))


        # output tensorboardX
        writer.add_scalar('iter/batch_loss', epoch_loss.val, epoch*int(len(source_train_loader))+i)
        writer.add_scalar('iter/detection_loss', east_total_loss.val, epoch*int(len(source_train_loader))+i)
        writer.add_scalar('iter/source_domain_loss', source_domain_loss.val, epoch*int(len(source_train_loader))+i)
        writer.add_scalar('iter/target_domain_loss', target_domain_loss.val, epoch*int(len(source_train_loader))+i)



        log.write('Epoch is [{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
            epoch + 1, i + 1, int(len(source_train_loader)), time.time() - start_time, loss.item()), is_terminal=0)
        log.write("\n",is_terminal=0)

    # output tensorboardX
    writer.add_scalar('epoch/total_loss', epoch_loss.avg, epoch)
    writer.add_scalar('epoch/detection_loss', east_total_loss.avg, epoch)
    writer.add_scalar('epoch/source_domain_loss', source_domain_loss.avg, epoch)
    writer.add_scalar('epoch/target_domain_loss', target_domain_loss.avg, epoch)

    log.write('Epoch[{}], Train, epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch, epoch_loss.avg,
                                                                            time.time() - epoch_time))
    log.write("\n")
    return epoch_loss.avg


def eval(model, val_dataloader, criterion, loss_domain, epoch):
    model.eval()
    epoch_loss = AverageMeter()
    detection_total_loss = AverageMeter()
    val_domain_loss = AverageMeter()

    epoch_time = time.time()
    log.write('start eval')
    log.write('\n')
    with torch.no_grad():
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(val_dataloader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
                device)
            pred_score, pred_geo, pred_cls = model(img)
            east_loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            # domain label
            domain_t = Variable(torch.ones(pred_cls.size()).float().cuda())
            loss_domain_t = loss_domain(m(pred_cls), domain_t)


            loss = east_loss+0.1*loss_domain_t

            # record loss
            epoch_loss.update(loss.item(), img.size(0))
            detection_total_loss.update(east_loss.item(), img.size(0))
            val_domain_loss.update(loss_domain_t.item(), img.size(0))

            log.write('Epoch[{}], Eval, mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                     epoch, i + 1, int(len(val_dataloader)), time.time() - start_time, loss.item()), is_terminal=0)


    # output tensorboardX
    writer.add_scalar('eval/total_loss', epoch_loss.avg, epoch)
    writer.add_scalar('eval/detection_loss', detection_total_loss.avg, epoch)
    writer.add_scalar('eval/domain_loss', val_domain_loss.avg, epoch)

    log.write( 'Epoch[{}], Eval, epoch_loss is {:.8f}, detection_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch, epoch_loss.avg, detection_total_loss.avg, time.time() - epoch_time))
    log.write('\n')

    return detection_total_loss.avg


def main(args):
    source_train_set = custom_dataset(args.train_data_path, args.train_gt_path)
    # target_train_set = custom_dataset(args.target_data_path, args.target_gt_path)
    target_train_set = IC13_dataset(args.target_data_path, args.target_gt_path)
    valid_train_set = valid_dataset(args.val_data_path, args.val_gt_path, data_flag='ic13')

    source_train_loader = data.DataLoader(source_train_set, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, drop_last=True)
    target_train_loader = data.DataLoader(target_train_set, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_loader = data.DataLoader(valid_train_set, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers, drop_last=False)

    criterion = Loss().to(device)
    # domain loss
    # loss_domain = torch.nn.CrossEntropyLoss().to(device)
    loss_domain = torch.nn.BCELoss().to(device)
    best_loss = 1000
    best_num = 0

    model = EAST()
    if args.pretrained_model_path:
        model.load_state_dict(torch.load(args.pretrained_model_path))

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        best_loss = checkpoint['best_loss']
        current_epoch_num = checkpoint['epoch']

    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True

    model.to(device)

    total_epoch = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[total_epoch // 3, total_epoch * 2 // 3], gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=12, threshold=args.lr/100)
    current_epoch_num = 0

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume)
        scheduler.load_state_dict(checkpoint['scheduler'])


    for epoch in range(current_epoch_num, total_epoch):
        each_epoch_start = time.time()
        scheduler.step(epoch)
        # add lr in tensorboardX
        writer.add_scalar('epoch/lr',get_learning_rate(optimizer) ,epoch)

        train_loss = train(source_train_loader, target_train_loader, model, criterion, loss_domain, optimizer, epoch)

        val_loss = eval(model,  valid_loader, criterion, loss_domain, epoch)
        # scheduler.step(train_loss)

        if val_loss < best_loss:
            best_num = epoch + 1
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.module.state_dict() if data_parallel else model.state_dict())
            # save best model

            torch.save({
                'epoch': epoch+1,
                'state_dict': best_model_wts,
                'best_loss': best_loss,
                'scheduler': scheduler.state_dict(),
            }, os.path.join(save_folder, "model_epoch_best.pth"))

            log.write('best model num:{}, best loss is {:.8f}'.format(best_num, best_loss))
            log.write('\n')

        if (epoch + 1) % int(args.save_interval) == 0 :
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save({
                'epoch': epoch+1,
                'state_dict': state_dict,
                'best_loss': best_loss,
                'scheduler': scheduler.state_dict(),
            }, os.path.join(save_folder, 'model_epoch_{}.pth'.format(epoch + 1)))
            log.write('save model')
            log.write('\n')

        log.write('=' * 50)
        log.write('\n')


if __name__ == '__main__':
    # parse the config
    args = get_args(sys.argv[1:])
    main(args)
