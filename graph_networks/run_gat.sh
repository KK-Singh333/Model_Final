#!/bin/bash

python train_gat.py \
    --data-path ../dataset/ADFD \
    --hidden-channels 128 \
    --num-heads 8 \
    --num-layers 3 \
    --dropout 0.5 \
    --lr 0.001 \
    --weight-decay 5e-4 \
    --epochs 100 \
    --batch-size 32 \
    --device cuda \
    --save-dir checkpoints/gat_adsz \
    --checkpoint-freq 25
