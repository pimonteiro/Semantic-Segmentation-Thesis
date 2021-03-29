import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
import importlib  
from keras.optimizers import Adam
import os
import time
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib
from utils.keras_functions import sparse_crossentropy_ignoring_last_label, Jaccard
from viewer.label_visualizer import vis_segmentation

matplotlib.use('TkAgg')

import Keras_segmentation_deeplab_v3_1.utils


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False, help="Model to evaluate. Only accepts {'xception','mobilenetv2'}")
parser.add_argument('--model_folder', type=str, required=False, help="Model saved as .h5 .")
parser.add_argument('--OS', type=int, required=False, help="OS size for xception. Applied ONLY on exception backbone.")
parser.add_argument('--alpha', type=int, required=False, help="Alpha size for mobilenetv2. Applied ONLY on exception mobilenetv2.")
parser.add_argument('--norm', type=int, required=False, help="Normalization method. 0 -> [-1,1] | 1 -> [0,1] (default).")
parser.add_argument('--dataset', type=str, required=False, help="Dataframe containing the dataset two columns: the class (type) and respective image path (image).")
parser.add_argument('--use_crf', default=False, action='store_true', help="Use CRF to clear model output.")
parser.add_argument('--image', type=str, required=False, help="Infer on a single image.")
parser.add_argument('--type', type=int, required=False, help="Class to focus on inferention.")
parser.add_argument('--n', type=int, required=False, help="Number of images to classify (default 5).")
parser.add_argument('--override', default=False, action='store_true', help="Flag to use the whole dataset and not only test subset.")


keras_deeplab = importlib.import_module("keras-deeplab-v3-plus.model")

image_size = (512,512)

losses = sparse_crossentropy_ignoring_last_label
metrics = {'pred_mask' : [Jaccard]}


def build_model(model_name, os, alpha, norm):
    try:
        deeplab_model = keras_deeplab.Deeplabv3(backbone=model_name, input_shape=(512, 512, 3), classes=19, weights='cityscapes', OS=os, alpha=alpha, infer=True, normalization=norm)
    except:
        raise Exception("No model with given backbone: ", model_name)

    deeplab_model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
              loss = losses, metrics = metrics)
    return deeplab_model

def load_model(path):
    deeplab_model = tf.keras.models.load_model(path, custom_objects={
        'sparse_crossentropy_ignoring_last_label': sparse_crossentropy_ignoring_last_label,
        'Jaccard': Jaccard
        })

    return deeplab_model

def mIOU(gt, preds):
    ulabels = np.unique(gt)
    iou = np.zeros(len(ulabels))
    for k, u in enumerate(ulabels):
        inter = (gt == u) & (preds==u)
        union = (gt == u) | (preds==u)
        iou[k] = inter.sum()/union.sum()
    return np.round(iou.mean(), 2)


def infer_single_image(model, image_path, user_crf, model_name):
    output = "single_images_infer/" + str(time.time())
    os.makedirs(os.path.abspath(output))
    with open(output + '/details.txt', 'w') as f:
        f.write("Model specs: " + model_name)
        f.write("\n")
        f.write("Image location:" + image_path)
    
    image = cv2.imread(image_path, 3)
    old_shape = image.shape
    image = cv2.resize(image, image_size)

    pred = model.predict(np.expand_dims(image, axis=0))
    labels = np.argmax(pred.squeeze(), -1)

    cv2.imwrite(output + "/image_resized.png",image)
    cv2.imwrite(output + "/labels_resized.png",labels)

    vis_segmentation(image,labels, output + "/overlay_resized.png")

    image = cv2.resize(image, (old_shape[1], old_shape[0]))
    labels = cv2.resize(labels, (old_shape[1], old_shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output + "/image.png",image)
    cv2.imwrite(output + "/labels.png",labels)
    
    vis_segmentation(image,labels, output + "/overlay.png")

def infer_dataset_classe(model, dataset_path, use_crf, type, n, model_name, flag):
    random.seed(7)
    data = pd.read_csv(dataset_path)
    data = data[data['type'] == type]
    tmp_data = data[data['scene'].str.contains('0003') | data['scene'].str.contains('0004')]

    if data.empty:
        if not flag:
            raise Exception("Class requested not available on test subset. Use flag --override to override this condition and use the whole dataset.")
        else:
            tmp_data = data[data['type'] == type]
    
    samples = tmp_data.sample(n = n)

    output = "dataset_images_infer/" + str(time.time()) + "/"
    os.makedirs(os.path.abspath(output))

    with open(output + 'details.txt', 'w') as f:
        f.write("Model specs: " + model_name)
        f.write("\n")
        f.write("Class: " + str(type))

    for i in range(n):
        tmp_output = output + str(i)
        os.makedirs(os.path.abspath(tmp_output))
        image_path = "/mnt/7BCDA59C6DEFFE3C/KITTI-360/data_2d_raw/" + samples.iloc[i].scene + "/image_00/data_rect/" + samples.iloc[i].image
        gt_path = "/mnt/7BCDA59C6DEFFE3C/KITTI-360/data_2d_semantics/" + samples.iloc[i].scene + "/" + samples.iloc[i].image

        image = cv2.imread(image_path, 1)
        old_shape = image.shape
        image = cv2.resize(image, image_size)

        pred = model.predict(np.expand_dims(image, axis=0))
        labels = np.argmax(pred.squeeze(), -1)

        cv2.imwrite(tmp_output + "/image_resized.png",image)
        cv2.imwrite(tmp_output + "/labels_resized.png",labels)

        vis_segmentation(image,labels, tmp_output + "/overlay_resized.png")
        
        with open(tmp_output + '/details.txt', 'w') as f:
            f.write("Image location: " + str(samples.iloc[i].scene))
            f.write("\n")
            
            gt_l = cv2.imread(gt_path, 0)
            gt_l = cv2.resize(gt_l, image_size, interpolation = cv2.INTER_NEAREST)
            f.write("mIOU: " + str(mIOU(gt_l, labels)))


        image = cv2.resize(image, (old_shape[1], old_shape[0]))
        labels = cv2.resize(labels, (old_shape[1], old_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(tmp_output + "/image.png",image)
        cv2.imwrite(tmp_output + "/labels.png",labels)
        vis_segmentation(image,labels, tmp_output + "/overlay.png")



if __name__ == "__main__":
    args = parser.parse_args()

    new_os = 8
    new_alpha= 1.
    new_norm = 1
    model_name = 'xception'
    if args.model_folder  is not None:
        model = load_model(args.model_folder)
        model_name = args.model_folder

    elif args.model  is not None:
        if args.OS is not None:
            new_os = args.OS
        if args.alpha is not None:
            new_alpha = args.alpha
        if args.norm is not None:
            new_norm = args.norm

        model = build_model(args.model, new_os, new_alpha, new_norm)
    else:
        raise Exception("No model or model_folder was definied. Run --help for more details.")
    
    if args.image  is not None:
        infer_single_image(model, args.image, args.use_crf, model_name)
    elif args.dataset is not None:
        if args.type is None:
            raise Exception("Parameter --class required. Run --help for more details.")
        n = 5 if not args.n else args.n

        infer_dataset_classe(model, args.dataset, args.use_crf, args.type, n, model_name, args.override)
    else:
        raise Exception("No image or dataset provided. Run --help for more details.")