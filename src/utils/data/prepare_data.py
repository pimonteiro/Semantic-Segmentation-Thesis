import os
import numpy as np
import pandas as pd
from PIL import Image
import glob
import sys
from skimage.io import imsave, imread
import argparse
import wget

from labels import convert_id_to_training_id

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=False, help="Folder containing the labels. The root folder MUST be named dataset.")
parser.add_argument('--output', type=str, required=True, help="Output destination for the converted images. Will follow the same scheme as the input.")

myfunc_vec = np.vectorize(convert_id_to_training_id)

def download_kitti360_labels():
  url = "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ed180d24c0a144f2f1ac71c2c655a3e986517ed8/data_2d_semantics.zip"
  print("=====Downloading=====")
  wget.download(url)
  os.system("mkdir dataset")
  print("=====Unzipping data=====")
  os.system("unzip -d dataset/ data_2d_semantics.zip")
  os.system("rm data_2d_semantics.zip")

def prepare_labels(output, path="dataset/"):
  try: 
    os.mkdir(output)
  except:
    print("Failed creating folder " + output)
  scenes = os.path.join(path + "data_2d_semantics/train/*")
  scenes_list = glob.glob(scenes)

  for s in scenes_list:
    new_folder = s.replace("dataset", output)
    os.makedirs(new_folder, exist_ok=True)
    images = os.path.join(s + "/semantic/*.png")
    images_list = glob.glob(images)
    print("=====Processing ", s, "=====")

    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(images_list) ), end=' ')
    for i in images_list:
      image = np.array(Image.open(i))
      conv = myfunc_vec(image)
      name = i.replace("dataset", output)
      name = name.replace("semantic/", "")

      try:
        imsave(name, conv.astype(np.uint8), check_contrast=False)
      except Exception as e:
        print(e)

      progress += 1
      print("\rProgress: {:>3} %".format( progress * 100 / len(images_list) ), end=' ')
      sys.stdout.flush()
    print("\n")


if __name__ == "__main__":
  args = parser.parse_args()

  if args.input:
    if "dataset" in args.input:
      prepare_labels(args.output, args.input)
  else:
    download_kitti360_labels()
    prepare_labels(output=args.output)
    
