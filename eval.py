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
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
    print(res)
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time()-start_time))

    if not save_flag:
        shutil.rmtree(submit_path)


if __name__ == '__main__': 
    model_name = '/youedata/dengjinhong/zjw/code/EAST_Tansfer/checkpoint/DA_IC15_IC13_ep800_angel/model_epoch_350.pth'
    test_img_path = '/youedata/dengjinhong/zjw/dataset/icdar2015/ch4_test_images/'
    submit_path = './submit'
    eval_model(model_name, test_img_path, submit_path)
