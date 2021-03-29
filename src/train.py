import tensorflow as tf
import argparse
import importlib  
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing import image

from utils.keras_functions import sparse_crossentropy_ignoring_last_label, Jaccard

keras_deeplab = importlib.import_module("keras-deeplab-v3-plus.model")

import Keras_segmentation_deeplab_v3_1.utils

monitor = 'Jaccard'
mode = 'max'

losses = sparse_crossentropy_ignoring_last_label
metrics = {'pred_mask' : [Jaccard]}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False, help="Model to evaluate. Only accepts {'xception','mobilenetv2'}")
parser.add_argument('--pretrained', default=False, action='store_true', help="False to not use cityscape weights.")
parser.add_argument('--freezed', default=False, action='store_true', help="True to freeze all layers except last ones.")
parser.add_argument('--model_folder', type=str, required=False, help="Model saved as .h5 .")
parser.add_argument('--dataset', type=str, required=True, help="Dataframe containing the dataset.")
parser.add_argument('--batch_size', type=int, required=True, help="Batch size for the training.")
parser.add_argument('--epochs', type=int, required=True, help="Number of epochs for training.")
parser.add_argument('--name', type=str, required=True, help="Name for the ouput log dir.")
parser.add_argument('--input_size', type=int, nargs=2, help='Input size for the model.')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def build_model(model_name, pretrained, input_size=(512,512,3)):
    if pretrained:
        weights = 'cityscapes'
    else:
        weights = None
    try:
        deeplab_model = keras_deeplab.Deeplabv3(backbone=model_name, input_shape=input_size, classes=19, weights=weights, infer=False)
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

    """   symbolic_weights = getattr(deeplab_model.optimizer, 'weights')
        weight_values = tf.keras.backend.batch_get_value(symbolic_weights)

        grad_vars = deeplab_model.trainable_weights

        optimizer = Adam(lr=7e-4)
        zero_grads = [tf.zeros_like(w) for w in grad_vars]

        # Apply gradients which don't do nothing with Adam
        optimizer.apply_gradients(zip(zero_grads, grad_vars))

        # Set the weights of the optimizer
        optimizer.set_weights(weight_values)

        deeplab_model.optimizer.set_weights(weight_values) """
        
    deeplab_model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
            loss = losses, metrics = metrics)

    return deeplab_model
    

def build_callbacks(output_log, tf_board = False):
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs/'+output_log, histogram_freq=0,
                        write_graph=False, write_images = False)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = '../weights/'+output_log+'/'+output_log+'_model.{epoch:02d}-{val_loss:.2f}-{val_Jaccard:.2f}.h5', verbose=1, save_best_only=False, save_weights_only=False,
                                    monitor = 'val_{}'.format(monitor), mode = mode)
    stop_train = tf.keras.callbacks.EarlyStopping(monitor = 'val_{}'.format(monitor), patience=100, verbose=1, mode = mode)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_{}'.format(monitor), factor=0.5,
                patience=5, min_lr=1e-6)
    if tf_board:
        callbacks = [checkpointer, reduce_lr, stop_train, tensorboard]
    else:
        callbacks = [checkpointer, reduce_lr, stop_train]
    return callbacks


def train(model, dataset_path, batch_size, freezed, epochs, name):
    # fine-tune model (train only last conv layers)
    if freezed:
        flag = 0
        for k, l in enumerate(model.layers):
            l.trainable = False
            if l.name == 'concat_projection':
                flag = 1
            if flag:
                l.trainable = True
    

    callbacks = build_callbacks(name, tf_board = True)
    SegClass = Keras_segmentation_deeplab_v3_1.utils.SegModel(dataset_path, image_size=image_size)
    SegClass.set_seg_model(model)
    SegClass.set_num_epochs(epochs)
    SegClass.set_batch_size(batch_size)


    train_generator = SegClass.create_generators(dataset = dataset_path, blur=0, mode='train',
                                                    n_classes=19, horizontal_flip=False, vertical_flip=False, 
                                                    brightness=0, rotation=False, zoom=0, batch_size=batch_size,
                                                    seed=7, do_ahisteq=False, random_crop=True, resize_shape=image_size)

    valid_generator = SegClass.create_generators(dataset =dataset_path, blur=0, mode='val',
                                                    n_classes=19, horizontal_flip=False, vertical_flip=False, 
                                                    brightness=0, rotation=False, zoom=0, batch_size=batch_size,
                                                    seed=7, do_ahisteq=False, resize_shape=image_size)

    history = SegClass.train_generator(model, train_generator, valid_generator, callbacks, mp = True)


if __name__ == "__main__":
    args = parser.parse_args()
    global image_size
    image_size = (512,512)

    if args.model_folder:
        model = load_model(args.model_folder)
    elif args.model:
        if args.input_size:
            model = build_model(args.model, args.pretrained, (args.input_size[0], args.input_size[1], 3))
        else:
            model = build_model(args.model, args.pretrained)
    else:
        raise Exception("No model or model_folder was definied. Run --help for more details.")
    if args.input_size:
        image_size = (args.input_size[0], args.input_size[1])

    train(model, args.dataset, args.batch_size, args.freezed, args.epochs, args.name)
