#!/bin/bash

'''
directory structure:
/root/
    data/
        voc2012/
        coco2017/
    ckpt/
        checkpoint_voc.pth
        checkpoint_coco.pth
        official/
            maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
            resnet50-19c8e357.pth
    scripts/
        advance/
            pytorch_mask_rcnn/
            run.sh
            train.py
            demo.ipynb
'''


root='/input/'
script_dir='advance/'
dataset='voc'
data_dir='voc2012/VOCdevkit/VOC2012/'
ckpt_file='checkpoint_voc.pth'

python ${root}scripts/${script_dir}train.py --use-cuda --pretrained --epochs 1 --batch-size 2 --dataset ${dataset} \
--data-dir ${root}data/${data_dir} --ckpt-path ${root}ckpt/${ckpt_file} \
--offi-ckpt-dir ${root}ckpt/official

cp ${root}ckpt/${ckpt_file} /data/ckpt/

/root/shutdown.sh # shut the server down