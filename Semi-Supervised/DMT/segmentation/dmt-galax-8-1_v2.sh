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

# Supervised-training
echo baseline_1_COCO_pretrained
python main.py --exp-name=${exp_name}__p0--c --val-num-steps=350 --state=2 --epochs=10 --dataset=galax --train-set=${train_set} --sets-id=${sid} --continue-from=galax_coco_resnet101.pt --coco --mixed-precision --lr=${lr} --batch-size-labeled=2 --batch-size-pseudo=0 --seed=1
echo baseline_2_Imagenet_Pretrained
python main.py --exp-name=${exp_name}__p0--i --val-num-steps=350 --state=2 --epochs=10 --dataset=galax --train-set=${train_set} --sets-id=${sid} --mixed-precision --lr=${lr} --batch-size-labeled=2 --batch-size-pseudo=0 --seed=2


# SSL-training
echo dmt
for i in ${!rates[@]}; do
  echo ${phases[$i]}--${rates[$i]}
  
  echo labeling_1__${$i}
  python main.py --labeling --dataset=galax --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--i.pt --mixed-precision --batch-size-labeled=2 --label-ratio=${rates[$i]}

  echo training_1__${$i}
  python main.py --exp-name=${exp_name}__p${phases[$i]}--c --dataset=galax --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--c.pt --coco --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --batch-size-labeled=2 --batch-size-pseudo=6 --seed=1 --val-num-steps=300
  
  echo labeling_2__${$i}
  python main.py --labeling --dataset=galax --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--c.pt --coco --mixed-precision --batch-size-labeled=2 --label-ratio=${rates[$i]}

  echo training_2__${$i}
  python main.py --exp-name=${exp_name}__p${phases[$i]}--i --dataset=galax --train-set=${train_set} --sets-id=${sid} --continue-from=${exp_name}__p${i}--i.pt --mixed-precision --epochs=${ep} --gamma1=${i1} --gamma2=${i2} --lr=${lr} --batch-size-labeled=2 --batch-size-pseudo=6 --seed=2 --val-num-steps=300
        
done
