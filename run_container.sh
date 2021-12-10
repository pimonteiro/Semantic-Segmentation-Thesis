docker run -it --gpus all --rm -p "8888:8888" -p "6006:6006" -v "/home/pimonteiro/repos/Semantic-Segmentation-Thesis/:/workspace/:Z" -v "/mnt/7BCDA59C6DEFFE3C/:/mnt/7BCDA59C6DEFFE3C/" my_ml_container bash

# jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser .
# tensorboard --logdir --bind_all
# sudo docker exec -it <name> bash
# sudo docker build -t my_ml_container .
# python evaluate.py --model_folder ../trt_converted/xception_retrained_16bit/ --dataset ../kitti360_dataset.csv --batch_size 8 --input_size 375 513 --output ../new_evaluation_v2/xception_16bit  --tensorrt