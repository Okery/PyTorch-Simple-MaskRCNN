#!/bin/bash

ngpu=1
dataset="voc"
batch_size=2
epochs=1
data_dir="voc2012/VOCdevkit/VOC2012/"
ckpt_file="checkpoint_${dataset}.pth"

# training
python -m torch.distributed.launch --nproc_per_node=${ngpu} --use_env train.py \
--use-cuda --pretrained --epochs ${epochs} --batch-size ${batch_size} --dataset ${dataset} \
--data-dir ${data_dir} --ckpt-path ${ckpt_file}

