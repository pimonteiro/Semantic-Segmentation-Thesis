import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
import importlib  
from keras.optimizers import Adam
import time
import cv2
import matplotlib.pyplot as plt

from viewer.label_visualizer import label_to_color_image
from utils.keras_functions import sparse_crossentropy_ignoring_last_label, Jaccard

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from tensorflow.python.saved_model import tag_constants

keras_deeplab = importlib.import_module("keras-deeplab-v3-plus.model")
losses = sparse_crossentropy_ignoring_last_label
metrics = {'pred_mask' : [Jaccard]}



def build_model(model_name, os, alpha, norm):
    try:
        deeplab_model = keras_deeplab.Deeplabv3(backbone=model_name, input_shape=(image_size[0], image_size[1], 3), classes=19, weights='cityscapes', OS=os, alpha=alpha, infer=True, normalization=norm)
    except:
        raise Exception("No model with given backbone: ", model_name)

    deeplab_model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
              loss = losses, metrics = metrics)
    return deeplab_model

def load_model(path):
    deeplab_model = build_model('xception', 8, 1, 0)
    deeplab_model.load_weights(path)
    return deeplab_model

def load_model_trt(path):
    saved_model_loaded = tf.saved_model.load(path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    return infer


def get_data(dataset, fps):
    data = pd.read_csv(dataset, index_col=0)
    data = data[data['subset'] == 'test']
    data = data.sort_values(by=['x'], ascending=True)
    data = data.iloc[:1000]                      # arbitrary decision

    return data
    
def prep_data(data, fps):
    b = np.zeros((fps, 375, 513, 3))

    for (i,d) in enumerate(data.values):
        t = cv2.imread(d[0],3)
        t = cv2.resize(t, (513,375))
        b[i] = t

    return b

def save_frames(frames, mask, video):
    global masked

    if masked:
        for i in mask:
            m = label_to_color_image(i).astype(np.uint8)
            video.write(m)
    else:
        print("half half")
        for (f, m) in zip(frames, mask):
            l = label_to_color_image(m).astype(np.uint8)
            f = f.astype(np.uint8)
            t = cv2.addWeighted(f,1, l, 0.9,0)
            video.write(t)


def run_simulation(model, fps, dataset, t, isTrt):
    print("Preparing data...")
    data = get_data(dataset, fps)

    outpJson = {}
    video = cv2.VideoWriter('simulation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (513,375))
    print("Beginning simulation...")
    for i in range(1,t+1):
        print("Second ", i)
        b = data.iloc[(fps*(i-1)):(fps*i)]
        b = prep_data(b, fps)

        start = time.time()
        if isTrt:
            inp = tf.constant(b)
            preds = model(inp)
            res = np.argmax(preds[list(preds.keys())[0]].numpy(), axis=-1)
        else:
            res = model.predict(b)
            res = np.argmax(res, axis=-1)

        end = time.time()
        outpJson[i] = end-start
        
        save_frames(b, res, video)

    print(outpJson)
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, help="Model to evaluate. Only accepts {'xception','mobilenetv2'}")
    parser.add_argument('--model_folder_h5', type=str, required=False, help="Model saved as .h5 .")
    parser.add_argument('--model_folder_trt', type=str, required=False, help="Model converted for TensorRT.")
    parser.add_argument('--fps', type=int, default=25, help="Frames per second from the camera.")
    parser.add_argument('--dataset', type=str, required=False, help="Dataframe containing the dataset two columns: the class (type) and respective image path (image).")
    parser.add_argument('--t', type=int, default=10, help="Time of execution in seconds (Default 10).")
    parser.add_argument('--masked', default=False, action='store_true', help='Save masked video.')

    args = parser.parse_args()
    global masked
    global image_size
    masked = args.masked
    image_size = (375, 513)

    if args.model_folder_trt is not None:
        print("TRT Model")
        model = load_model_trt(args.model_folder_trt)
        isTrt = True
    elif args.model_folder_h5 is not None:
        print("H5 Model")
        model = load_model(args.model_folder_h5)
        isTrt = False
    else:
        raise Exception("No model_folder_h5 or model_folder_trt was defined. Run --help for more details.")


    run_simulation(model, args.fps, args.dataset, args.t, isTrt)


