import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
import importlib  
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

from utils.keras_functions import sparse_crossentropy_ignoring_last_label, Jaccard

keras_deeplab = importlib.import_module("keras-deeplab-v3-plus.model")

import Keras_segmentation_deeplab_v3_1.utils

image_size = (512,512)

losses = sparse_crossentropy_ignoring_last_label
metrics = {'pred_mask' : [Jaccard]}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False, help="Model to evaluate. Only accepts {'xception','mobilenetv2'}")
parser.add_argument('--model_folder', type=str, required=False, help="Folder with the custom model files.")
parser.add_argument('--dataset', type=str, required=True, help="Dataframe containing the dataset.")
parser.add_argument('--batch_size', type=int, required=True, help="Batch size for the training.")
parser.add_argument('--output', type=str, required=True, help="Output destination for the resulted evaluation.")


def build_model(model_name):
    try:
        deeplab_model = keras_deeplab.Deeplabv3(backbone=model_name, input_shape=(512, 512, 3), classes=19, weights='cityscapes')
    except:
        raise Exception("No model with given backbone: ", model_name)

    deeplab_model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
              loss = losses, metrics = metrics)
    return deeplab_model

def load_model(path):
    pass

def evaluate(model, dataset_path, output, batch_size):
    data = pd.read_csv(dataset_path)
    scenes = np.unique(data[data['subset'] == 'test'].scene)

    os.mkdir(output)
    SegClass = Keras_segmentation_deeplab_v3_1.utils.SegModel(dataset_path, image_size=image_size)

    for s in scenes:
        test_generator = SegClass.create_generators(dataset =dataset_path, subscene=s, blur=0, mode='test',
                                                    n_classes=19, horizontal_flip=False, vertical_flip=False, 
                                                    brightness=0, rotation=False, zoom=0, batch_size=batch_size,
                                                    seed=7, do_ahisteq=False)
        
        conf_m = np.zeros((19, 19), dtype=float)
        print(s, ':', len(test_generator))
        for n in range(len(test_generator)):
            label = np.zeros((batch_size,np.prod(image_size)), dtype='uint8')    

            x,y = test_generator.__getitem__(n)
            label = y
            #label[n,:] = y[0,:,0]
            print("Starting prediction ", n, "...")
            preds = model.predict(x)
            mask = np.reshape(np.argmax(preds, axis=-1), (-1,) + image_size)    
            flat_pred = np.ravel(mask).astype('int')
            flat_label = np.ravel(label).astype('int')
            
            for p, l in zip(flat_pred, flat_label):
                if l == 19:
                    print("I dont understand...")
                    continue
                if l < 19 and p < 19:
                    conf_m[l-1, p-1] += 1
                #else:
                    #print('Invalid entry encountered, skipping! Label: ', l,
                    #       ' Prediction: ', p)
        
        I = np.diag(conf_m)
        U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
        IOU = I/U
        IOU = np.nan_to_num(IOU)
        meanIOU = np.mean(IOU)
        print(meanIOU)

        classes = [c for c in Keras_segmentation_deeplab_v3_1.utils.get_CITYSCAPES_classes().values()]
        plt.figure(figsize=(12,8))
        cm1 = Keras_segmentation_deeplab_v3_1.utils.plot_confusion_matrix(conf_m, classes, normalize=True)
        plt.title('DeepLab\nScene ' + str(s) + '\nMean IOU: '+ str(np.round(np.diag(cm1).mean(), 2)))
        plt.savefig(output + '/' + 'scene ' + str(s) + '.png')



if __name__ == "__main__":
    args = parser.parse_args()

    if args.model_folder:
        model = load_model(args.model_folder)
    elif args.model:
        model = build_model(args.model)
    else:
        raise Exception("No model or model_folder was definied. Run --help for more details.")
    
    evaluate(model, args.dataset, args.output, args.batch_size)
