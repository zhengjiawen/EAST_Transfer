# eval ICDAR2013
import time
import torch
import subprocess
import os
from models.model_fm_cls import EAST
from detect import detect_dataset
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    print('model epoch :', checkpoint['epoch'])
    # model.load_state_dict(torch.load(model_name))
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)
    os.chdir(submit_path)
    res = subprocess.getoutput('zip -q submit13.zip *.txt')
    res = subprocess.getoutput('mv submit13.zip ../')
    os.chdir('../')
    res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt13.zip –s=./submit13.zip')
    print(res)
    os.remove('./submit13.zip')
    print('eval time is {}'.format(time.time()-start_time))

    if not save_flag:
        shutil.rmtree(submit_path)


if __name__ == '__main__':
    model_name = '/youedata/dengjinhong/zjw/code/EAST_Tansfer/checkpoint/DA_IC15_IC13_ep800_angel/model_epoch_530.pth'
    test_img_path = '/youedata/dengjinhong/zjw/dataset/icdar2013/Challenge2_Test_Task12_Images/'
    submit_path = './submit_ic13'
    eval_model(model_name, test_img_path, submit_path)
