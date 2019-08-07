#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py
    --train_data_path /youedata/dengjinhong/zjw/dataset/icdar2017/train_images/
    --train_gt_path /youedata/dengjinhong/zjw/dataset/icdar2017/train_gts/
    --target_data_path /youedata/dengjinhong/zjw/dataset/icdar2015/ch4_training_images/
    --target_gt_path /youedata/dengjinhong/zjw/dataset/icdar2015/ch4_training_localization_transcription_gt/
    --val_data_path /youedata/dengjinhong/zjw/dataset/icdar2015/ch4_training_localization_transcription_gt/
    --val_gt_path /youedata/dengjinhong/zjw/dataset/icdar2015/Challenge4_Test_Task4_GT
    --checkpoint ./checkpoint/
    --fold baseline/
    -b 15
    --lr 1e-3
    --num_workers 8
    --epoch 100
    --save_interval 5

