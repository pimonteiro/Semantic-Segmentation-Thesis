import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
import importlib  
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys
import time
import json

from utils.keras_functions import sparse_crossentropy_ignoring_last_label, Jaccard

keras_deeplab = importlib.import_module("keras-deeplab-v3-plus.model")

import Keras_segmentation_deeplab_v3_1.utils

image_size = (512,512)

losses = sparse_crossentropy_ignoring_last_label
metrics = {'pred_mask' : [Jaccard]}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False, help="Model to evaluate. Only accepts {'xception','mobilenetv2'}")
parser.add_argument('--OS', type=int, required=False, help="OS size for xception. Applied ONLY on exception backbone.")
parser.add_argument('--alpha', type=int, required=False, help="Alpha size for mobilenetv2. Applied ONLY on exception mobilenetv2.")
parser.add_argument('--norm', type=int, required=False, help="Normalization method. 0 -> [-1,1] | 1 -> [0,1] (default).")
parser.add_argument('--model_folder', type=str, required=False, help="Model saved as .h5 .")
parser.add_argument('--dataset', type=str, required=True, help="Dataframe containing the dataset.")
parser.add_argument('--batch_size', type=int, required=True, help="Batch size for the training.")
parser.add_argument('--output', type=str, required=True, help="Output destination for the resulted evaluation. Created if not present")
parser.add_argument('--use_crf', default=False, action='store_true', help="Use CRF to clear model output.")

def _compute_mean_iou_and_dice(total_cm, obj):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    ious = cm_diag / denominator

    tmp = {}
    #print('Intersection over Union for each class:')
    for i, iou in enumerate(ious):
        #print('    class {}: {:.2f}'.format(i, iou))
        tmp[i] = iou 
    obj['iou_per_class'] = tmp

    # If the number of valid entries is 0 (no classes) we return 0.
    m_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)
    
    dices = (ious * 2) / (ious + 1)
    tmp = {}
    for i, dice in enumerate(dices):
        #print('    class {}: {:.2f}'.format(i, iou))
        tmp[i] = dice 
    obj['dice_per_class'] = tmp

    m_dice = np.where(
        num_valid_entries > 0,
        np.sum(dices) / num_valid_entries,
        0)
    m_iou = float(m_iou)
    m_dice = float(m_dice)
    obj['mIOU'] = np.round(m_iou,2)
    obj['mDICE'] = np.round(m_dice,2)
    #print('mean Intersection over Union: {:.2f}'.format(float(m_iou)))
    #print('mean DICE: {:.2f}'.format(float()))
    
    return obj

def _compute_accuracy(total_cm, obj):
    """Compute the accuracy via the confusion matrix."""
    denominator = total_cm.sum().astype(float)
    cm_diag_sum = np.diagonal(total_cm).sum().astype(float)

    # If the number of valid entries is 0 (no classes) we return 0.
    accuracy = np.where(
        denominator > 0,
        cm_diag_sum / denominator,
        0)
    accuracy = float(accuracy)
    obj['acc'] = np.round(accuracy,2)
    #print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))

    return obj

def build_model(model_name, os, alpha, norm):
    try:
        deeplab_model = keras_deeplab.Deeplabv3(backbone=model_name, input_shape=(512, 512, 3), classes=19, weights='cityscapes', OS=os, alpha=alpha, infer=True, normalization=norm)
    except:
        raise Exception("No model with given backbone: ", model_name)

    deeplab_model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
              loss = losses, metrics = metrics)
    return deeplab_model

def load_model(path):
    deeplab_model = tf.keras.models.load_model(path)
    
    return deeplab_model

def evaluate(model, dataset_path, output, batch_size, use_crf, params):
    data = pd.read_csv(dataset_path)
    scenes = np.unique(data[data['subset'] == 'test'].scene)

    os.makedirs(os.path.abspath(output))
    SegClass = Keras_segmentation_deeplab_v3_1.utils.SegModel(dataset_path, image_size=image_size)

    metrics = {}
    metrics['params'] = params

    for s in scenes:
        test_generator = SegClass.create_generators(dataset =dataset_path, subscene=s, blur=0, mode='test',
                                                    n_classes=19, horizontal_flip=False, vertical_flip=False, 
                                                    brightness=0, rotation=False, zoom=0, batch_size=batch_size,
                                                    seed=7, do_ahisteq=False)
        
        conf_m = np.zeros((19, 19), dtype=float)
        progress = 0.0
        print("Progress: {:>3} %".format( progress * 100 / len(test_generator) ), end=' ')
        time_start = time.time()
        for n in range(len(test_generator)):
            label = np.zeros((batch_size,np.prod(image_size)), dtype='uint8')    

            x,y = test_generator.__getitem__(n)
            label = y
            #label[n,:] = y[0,:,0]
            preds = model.predict(x)
            mask = np.argmax(preds, axis=-1)
            
            if use_crf:
                for i in range(batch_size):
                    crf_mask = Keras_segmentation_deeplab_v3_1.utils.do_crf(x[i].astype('uint8'),mask[i], zero_unsure=False)
                    mask[i] = crf_mask

            flat_pred = np.ravel(mask).astype('int')
            flat_label = np.ravel(label).astype('int')
            
            for p, l in zip(flat_pred, flat_label):
                if l == 255:        #label to ignore
                    continue
                if l < 19 and p < 19:
                    conf_m[l-1, p-1] += 1
                #else:
                    #print('Invalid entry encountered, skipping! Label: ', l,
                    #       ' Prediction: ', p)
            progress += 1
            print("\rProgress: {:>3} %".format( progress * 100 / len(test_generator) ), end=' ')
            sys.stdout.flush()

        time_end = time.time()

        scene_metrics = {}
        scene_metrics['n_images'] = test_generator.true_len()
        scene_metrics['runtime'] = time_end - time_start

        scene_metrics = _compute_mean_iou_and_dice(conf_m, scene_metrics)
        scene_metrics = _compute_accuracy(conf_m, scene_metrics)

        metrics['scene' + str(s)] = scene_metrics

        classes = [c for c in Keras_segmentation_deeplab_v3_1.utils.get_CITYSCAPES_classes().values()]
        plt.figure(figsize=(12,8))
        cm1 = Keras_segmentation_deeplab_v3_1.utils.plot_confusion_matrix(conf_m, classes, normalize=True)
        plt.title('DeepLab\nScene ' + str(s))
        plt.savefig(output + '/' + 'scene ' + str(s) + '.png')
    
    with open(output + '/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.model_folder:
        model = load_model(args.model_folder)
    elif args.model:
        new_os = 16
        new_alpha= 1.
        new_norm = 1

        if args.OS:
            new_os = args.OS
        if args.alpha:
            new_alpha = args.alpha
        if args.norm:
            new_norm = args.norm

        model = build_model(args.model, new_os, new_alpha, new_norm)
    else:
        raise Exception("No model or model_folder was definied. Run --help for more details.")
    
    params = {}
    params['Backbone'] = args.model
    params['OS'] = new_os
    params['Alpha'] = new_alpha
    if new_norm == 1:
        params['Normalization'] = "[0,1]"
    else:
        params['Normalization'] = "[-1,1]"

    evaluate(model, args.dataset, args.output, args.batch_size, args.use_crf, params)
