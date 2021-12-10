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
from datetime import datetime
from tensorflow.python.saved_model import tag_constants

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from utils.keras_functions import sparse_crossentropy_ignoring_last_label, Jaccard

keras_deeplab = importlib.import_module("keras-deeplab-v3-plus.model")

import Keras_segmentation_deeplab_v3_1.utils

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
parser.add_argument('--input_size', type=int, nargs=2, help="Input size for the model.")
parser.add_argument('--tensorrt', default=False, action='store_true', help='Flag to enable tensorrt support.')
parser.add_argument('--load_directly', default=False, action='store_true', help='Flag to directly load the model instead of building it and then loading the weights. Used for the pruned model.')



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
        deeplab_model = keras_deeplab.Deeplabv3(backbone=model_name, input_shape=(image_size[0], image_size[1], 3), classes=19, weights='cityscapes', OS=os, alpha=alpha, infer=True, normalization=norm)
    except:
        raise Exception("No model with given backbone: ", model_name)

    deeplab_model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
              loss = losses, metrics = metrics)
    return deeplab_model

def load_model(name, os, alpha, norm, path, load_directly):
    if not load_directly:
        deeplab_model = build_model(name, os, alpha, norm)
        deeplab_model.load_weights(path)
    else:
        deeplab_model = tf.keras.models.load_model(path)
    
    return deeplab_model

def load_model_trt(path, batch_size):
    saved_model_loaded = tf.saved_model.load(path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    data_build = np.zeros((batch_size, image_size[0], image_size[1], 3), dtype='float32')
    inp = tf.constant(data_build)
    print("Preping model for prediction....")
    infer(inp)
    print("Model ready for prediction!")
    return infer

def evaluate(model, dataset_path, output, batch_size, use_crf, tensorrt_flag, params):
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
                                                    seed=7, do_ahisteq=False, resize_shape=image_size)
        
        conf_m = np.zeros((19, 19), dtype=float)
        
        conf_m_reduced = np.zeros((19,19), dtype=float)

        progress = 0.0
        print("Progress: {:>3} %".format( progress * 100 / len(test_generator) ), end=' ')
        time_start = time.time()
        for n in range(len(test_generator)):
            label = np.zeros((batch_size,np.prod(image_size)), dtype='uint8')    

            x,y = test_generator.__getitem__(n)
            label = y
            #label[n,:] = y[0,:,0]
            
            if tensorrt_flag:
                inp = tf.constant(x)
                preds = model(inp)
                mask = np.argmax(preds[list(preds.keys())[0]].numpy(), axis=-1)
            else:
                preds = model.predict(x)
                mask = np.argmax(preds, axis=-1)
            
            if use_crf:
                for i in range(batch_size):
                    tmp = mask[i].reshape((image_size))
                    crf_mask = Keras_segmentation_deeplab_v3_1.utils.do_crf(x[i].astype('uint8'),tmp, zero_unsure=False)
                    mask[i] = np.ravel(crf_mask)

            flat_pred = np.ravel(mask).astype('int')
            flat_label = np.ravel(label).astype('int')
            
            for p, l in zip(flat_pred, flat_label):
                if l == 255:        #label to ignore
                    continue
                if l < 19 and p < 19:
                    conf_m[l, p] += 1

                    # Group of classes
                    if p in [13,14,15] and l in [13,14,15]:
                        conf_m_reduced[13, 13] += 1
                    elif l in [13,14,15]:
                        conf_m_reduced[13, p] += 1
                    elif p in [13,14,15]:
                        conf_m_reduced[l,13] += 1
                    else:
                        conf_m_reduced[l,p] += 1

                else:
                    print('Invalid entry encountered, skipping! Label: ', l,
                          ' Prediction: ', p)
            progress += 1
            print("\rProgress: {:>3} %".format( progress * 100 / len(test_generator) ), end=' ')
            sys.stdout.flush()


        time_end = time.time()

        # Reducing the confusion matrix
        conf_m_reduced = np.delete(conf_m_reduced, [2,3,4,5,8,9,10,14,15,16], 0)
        conf_m_reduced = np.delete(conf_m_reduced, [2,3,4,5,8,9,10,14,15,16], 1)

        scene_metrics = {}
        scene_metrics['n_images'] = test_generator.true_len()
        scene_metrics['runtime'] = time_end - time_start
        standard_res = {}

        standard_res = _compute_mean_iou_and_dice(conf_m, standard_res)
        standard_res = _compute_accuracy(conf_m, standard_res)
        scene_metrics['standard'] = standard_res

        specific_res = {}
        specific_res = _compute_mean_iou_and_dice(conf_m_reduced, specific_res)
        specific_res = _compute_accuracy(conf_m_reduced, specific_res)
        scene_metrics['specific'] = specific_res

        metrics['scene' + str(s)] = scene_metrics

        classes = [c for c in Keras_segmentation_deeplab_v3_1.utils.get_CITYSCAPES_classes().values()]
        plt.figure(figsize=(12,8))
        cm1 = Keras_segmentation_deeplab_v3_1.utils.plot_confusion_matrix(conf_m, classes, normalize=True)
        plt.title('DeepLab\nScene ' + str(s))
        plt.savefig(output + '/' + 'scene ' + str(s) + '.png')

        classes = [c for c in Keras_segmentation_deeplab_v3_1.utils.get_CITYSCAPES_classes_reduced().values()]
        plt.figure(figsize=(12,8))
        cm1 = Keras_segmentation_deeplab_v3_1.utils.plot_confusion_matrix(conf_m_reduced, classes, normalize=True)
        plt.title('DeepLab\nScene ' + str(s))
        plt.savefig(output + '/' + 'scene ' + str(s) + '_reduced.png')

    
    with open(output + '/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    global image_size
    image_size = (512,512)  

    new_os = 8
    new_alpha= 1.
    new_norm = 1

    params = {}

    if args.input_size is not None:
            image_size = (args.input_size[0], args.input_size[1])

    if args.tensorrt is True:
        model = load_model_trt(args.model_folder, args.batch_size)
        params['weights'] = 'tensorrt--' + args.model_folder
    elif args.model_folder  is not None:
        model = load_model('xception', new_os, new_alpha, new_norm, args.model_folder, args.load_directly)
        params['Weights'] = args.model_folder
    elif args.model  is not None:
        if args.OS is not None:
            new_os = args.OS
        if args.alpha is not None:
            new_alpha = args.alpha
        if args.norm is not None:
            new_norm = args.norm
        
        print(new_norm)
        model = build_model(args.model, new_os, new_alpha, new_norm)
    else:
        raise Exception("No model or model_folder was defined. Run --help for more details.")
    
    params['Backbone'] = args.model
    params['OS'] = new_os
    params['Alpha'] = new_alpha
    params['Input'] = image_size
    params['Batch'] = args.batch_size
    if new_norm == 1:
        params['Normalization'] = "[0,1]"
    else:
        params['Normalization'] = "[-1,1]"

    evaluate(model, args.dataset, args.output, args.batch_size, args.use_crf, args.tensorrt, params)
