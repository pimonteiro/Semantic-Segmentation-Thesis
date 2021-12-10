#!/bin/bash

python evaluate.py --model mobilenetv2 --dataset ../kitti360_dataset.csv --batch_size 4 --output ../new_evaluation_v2/mobilenetv2_norm --input_size 375 513 --norm 1
python evaluate.py --model mobilenetv2 --dataset ../kitti360_dataset.csv --batch_size 4 --output ../new_evaluation_v2/mobilenetv2_std --input_size 375 513 --norm 0
python evaluate.py --model_folder ../pruned_models/all_layers_pruning.h5 --dataset ../kitti360_dataset.csv --batch_size 4 --output ../new_evaluation_v2/pruned_all_layers_trained --input_size 375 513 --load_directly
#python evaluate.py --model_folder ../pruned_models/depth_wise.h5 --dataset ../kitti360_dataset.csv --batch_size 4 --output ../new_evaluation_v2/pruned_depth_wise_trained --input_size 375 513 --load_directly

