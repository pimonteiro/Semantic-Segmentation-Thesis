import tensorflow as tf
import numpy as np
import argparse
import importlib  
from keras.optimizers import Adam
from keras.backend import get_session
import time 

import matplotlib.pyplot as plt
from utils.keras_functions import sparse_crossentropy_ignoring_last_label, Jaccard
from tensorflow.python.compiler.tensorrt import trt_convert as trt


keras_deeplab = importlib.import_module("keras-deeplab-v3-plus.model")

image_size = (375,513)

losses = sparse_crossentropy_ignoring_last_label
metrics = {'pred_mask' : [Jaccard]}


def build_model(model_name, os, alpha, norm):
    try:
        deeplab_model = keras_deeplab.Deeplabv3(backbone=model_name, input_shape=(375, 513, 3), classes=19, weights='cityscapes', OS=os, alpha=alpha, infer=True, normalization=norm)
    except:
        raise Exception("No model with given backbone: ", model_name)

    deeplab_model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
              loss = losses, metrics = metrics)
    return deeplab_model

def load_model(path):
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    deeplab_model = keras_deeplab.Deeplabv3(backbone='xception', input_shape=(375, 513, 3), classes=19, weights='cityscapes', OS=16, alpha=1, infer=True, normalization=0)
    deeplab_model.load_weights(path)

    return deeplab_model




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorRT converter')
    parser.add_argument('--model_folder', type=str, default=None,
                        required=True,
                        help='Directory containing the input saved model in h5 format.')
    parser.add_argument('--output', type=str, default=None,
                        required=True,
                        help='Directory in which the converted model is saved')
    parser.add_argument('--precision', type=str,
                        choices=['FP32', 'FP16'], default='FP32',
                        help='Precision mode to use. FP16 and INT8 only')
    parser.add_argument('--max_workspace_size', type=int, default=(1<<30),
                        help='workspace size in bytes')

    args = parser.parse_args()

    #Load and convert h5 to pb format
    model = load_model(args.model_folder)
    tmp_folder = '/tmp/pb_files_' + str(time.time())
    model.save(tmp_folder)
    
    del model       #to reduce memory usage

    print('Converting to TF-TRT ' + args.precision)
    conversion_params = tf.experimental.tensorrt.ConversionParams(precision_mode=args.precision, max_workspace_size_bytes=args.max_workspace_size)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=tmp_folder, conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir=args.output)
    print('Done converting to TF-TRT ' + args.precision)
