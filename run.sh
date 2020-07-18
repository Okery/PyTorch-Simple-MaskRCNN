#!/bin/bash


dataset="coco"
print_freq=50
epochs=1
ckpt_path="maskrcnn_${dataset}.pth"
iters=200

if [ $dataset = "voc" ]
then
    data_dir="/data/voc2012/VOCdevkit/VOC2012/"
elif [ $dataset = "coco" ]
then
    data_dir="/data/coco2017/"
fi


python train.py --use-cuda --epochs ${epochs} --iters ${iters} \
--dataset ${dataset} --data-dir ${data_dir} --ckpt-path ${ckpt_path} --print-freq ${print_freq}

