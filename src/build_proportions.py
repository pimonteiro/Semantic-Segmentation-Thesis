import os
import wget
import numpy as np
import pandas as pd
from PIL import Image
import glob
import sys
import argparse
import wget


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=False, help="Folder containing the labels already converted.")
parser.add_argument('--output', type=str, required=True, help="Output file for the resulting csv.")


LABEL_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
               'traffic light', 'traffic sign', 'vegetation', 'sky', 'person', 
               'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled', "SCENE"]


def images_to_datasets(input, output):
    total_per_scene = pd.DataFrame(columns = ['type','scene'])

    scenes = os.path.join(input + "/*")
    scenes_list = glob.glob(scenes)

    for s in scenes_list:
        scene_name = s.replace(input, "")
        images = os.path.join(s + "/semantic/*.png")
        images_list = glob.glob(images)
        print("=====Processing ", s, "=====")

        progress = 0
        print("Progress: {:>3} %".format( progress * 100 / len(images_list) ), end=' ')
        for i in images_list:
            image = np.array(Image.open(i))
            uniques = np.unique(image)
            for j in uniques:
                new_row = pd.Series({"type": j, "scene": scene_name})
                total_per_scene = total_per_scene.append(new_row, ignore_index=True)

            progress += 1
            print("\rProgress: {:>3} %".format( progress * 100 / len(images_list) ), end=' ')
            sys.stdout.flush()
    print("\n")
    total_per_scene.to_csv(output, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    images_to_datasets(args.input, args.output)
    
