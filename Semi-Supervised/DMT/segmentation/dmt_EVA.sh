#!/bin/bash

train_set="8"
sid="1"
exp_name="dmt-galax-8-1"
ep="10"
lr="0.001"
c1="3"
c2="3"
i1="3"
i2="3"
phases=("1" "2" "3" "4" "5")
rates=("0.2" "0.4" "0.6" "0.8" "1")

# Evaluation
echo Evaluation and saving 
python main.py --exp-name=${exp_name}__p0--c --val-num-steps=350 --state=4 --epochs=10 --dataset=galax --train-set=${train_set} --sets-id=${sid} --continue-from=dmt_best.pt --coco --mixed-precision --lr=${lr} --batch-size-labeled=1 --batch-size-pseudo=0 --seed=1 --valtiny

