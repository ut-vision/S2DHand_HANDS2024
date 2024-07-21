#!/bin/bash

gpu=0

#Initialize rotation matrix
#for setup in 0 1
#do
#  for pair in 0,1 0,2 0,3 1,2 1,3 2,3
#  do
#    python3 initialize_R.py -eid 34 --setup ${setup} --pair ${pair}
#  done
#done

#Adaptation under in-dataset scenarios
for setup in 0 #1
do
  for pair in 0,1 #0,2 0,3 1,2 1,3 2,3
  do
    python3 adapt_detnet_dual.py -trs ah -tes ah --set val --root_idx 0 --pic 1024 --resume -eid 34 --epochs 10 --start_epoch 1 --gpus ${gpu} --checkpoint in_dataset_adapt --setup ${setup} --pair ${pair}
#    python3 adapt_detnet_dual.py -trs ah -tes ah --evaluate -eid 10 --gpus 0 --pic -1 --checkpoint in_dataset_adapt --setup ${setup} --pair ${pair}
  done
done
#python3 gen_submit_json.py --evaluation_dir in_dataset_adapt/evaluation/ah -eid 10 --savepath assemblyhands_test_joint_3d_eccv24_pred.json
