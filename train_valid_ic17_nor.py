import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataset import custom_dataset, valid_dataset, IC13_dataset
# from model_resnet import EAST  # resnet
from EAST_model import EAST #vgg16
from loss import Loss
import os
import time
import numpy as np
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def drawLoss(train_loss, valid_loss, save_name):
    x1 = range(0,len(train_loss))
    x2 = range(0,len(valid_loss))
    # print(x1,":",x2)
    # plt.figure(1)
    plt.plot(x1, train_loss, c='red', label='train loss')
    plt.plot(x2, valid_loss, c='blue', label = 'valid loss')
    plt.xlabel('item number')
    plt.legend(loc='upper right')
    plt.savefig(save_name, format='jpg')
    plt.close()

def eval(model, val_dataloader, criterion, epoch):
    model.eval()
    epoch_loss = 0
    epoch_time = time.time()
    with torch.no_grad():
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(val_dataloader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
                device)
            pred_score, pred_geo= model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            epoch_loss += loss.item()

            # print('Epoch[{}], Eval, mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
            #          epoch, i + 1, int(len(val_dataloader)), time.time() - start_time, loss.item()))
    print( 'Epoch[{}], Eval, epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch, epoch_loss/int(len(val_dataloader)), time.time() - epoch_time))
    return epoch_loss


def train(source_img_path, source_gt_path, valid_img_path,
          valid_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, pretrain_model_path=None, scheduler_path=None, current_epoch_num=0):

    if not os.path.exists(pths_path):
        os.mkdir(pths_path)

    # source_train_set = IC13_dataset(source_img_path, source_gt_path)
    source_train_set = custom_dataset(source_img_path, source_gt_path)
    valid_train_set = valid_dataset(valid_img_path, valid_gt_path)

    source_train_loader = data.DataLoader(source_train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)

    valid_loader = data.DataLoader(valid_train_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers, drop_last=False)

    criterion = Loss().to(device)
    loss_domain = torch.nn.CrossEntropyLoss()

    model = EAST()
    if None != pretrain_model_path:
        model.load_state_dict(torch.load(pretrain_model_path))
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 3, epoch_iter * 2 // 3], gamma=0.1)
    if None != scheduler_path:
        scheduler.load_state_dict(torch.load(scheduler_path))
    best_loss = 1000
    best_model_wts = copy.deepcopy(model.state_dict())
    best_num = 0

    train_loss = []
    valid_loss = []

    for epoch in range(current_epoch_num, epoch_iter):
        model.train()

        epoch_loss = 0
        epoch_time = time.time()
        for i, (s_img, s_gt_score, s_gt_geo, s_ignored_map) in enumerate(source_train_loader):
            start_time = time.time()


            s_img, s_gt_score, s_gt_geo, s_ignored_map = s_img.to(device), s_gt_score.to(device), s_gt_geo.to(device), s_ignored_map.to(device)


            pred_score, pred_geo = model(s_img)

            loss = criterion(s_gt_score, pred_score, s_gt_geo, pred_geo, s_ignored_map)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


            # print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
            #     epoch + 1, epoch_iter, i + 1, int(len(source_train_loader)), time.time() - start_time, loss.item()))

        epoch_loss_mean = epoch_loss / len(source_train_loader)
        train_loss.append(epoch_loss_mean)
        print('Epoch[{}], Train, epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch, epoch_loss_mean,
                                                                            time.time() - epoch_time))

        val_epoch_loss = eval(model, valid_loader, criterion, epoch)
        val_loss_mean = val_epoch_loss / len(valid_loader)
        valid_loss.append(val_loss_mean)

        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)

        if val_loss_mean < best_loss:
            best_num = epoch + 1
            best_loss = val_loss_mean
            best_model_wts = copy.deepcopy(model.module.state_dict() if data_parallel else model.state_dict())
            # save best model
            print('best model num:{}, best loss is {:.8f}'.format(best_num, best_loss))
            torch.save(best_model_wts, os.path.join(pths_path, 'model_epoch_best.pth'))
        if (epoch + 1) % interval == 0 :
            savePath = pths_path  + 'lossImg' + str(epoch + 1) + '.jpg'
            drawLoss(train_loss, valid_loss, savePath)
            print(time.asctime(time.localtime(time.time())))
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            lr_state = scheduler.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))
            torch.save(lr_state, os.path.join(pths_path, 'scheduler_epoch_{}.pth'.format(epoch + 1)))
            print("save model")
            print('=' * 50)



if __name__ == '__main__':
    source_img_path = '/youedata/dengjinhong/zjw/dataset/icdar2017/train_images/'
    source_gt_path = '/youedata/dengjinhong/zjw/dataset/icdar2017/train_gts/'

    valid_img_path = '/youedata/dengjinhong/zjw/dataset/icdar2015/ch4_test_images'
    valid_gt_path = '/youedata/dengjinhong/zjw/dataset/icdar2015/Challenge4_Test_Task4_GT'
    pths_path = './checkpoint/ic17_nor/'

    batch_size = 40
    lr = 1e-3
    num_workers = 8
    epoch_iter = 100
    save_interval = 5
    train(source_img_path, source_gt_path, valid_img_path,
          valid_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, pretrain_model_path=None, scheduler_path=None, current_epoch_num=0)








