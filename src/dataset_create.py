import os
import glob
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="Folder containing the dataset.")
parser.add_argument('--train', type=int, nargs="+", required=True, help="Train subsets.")
parser.add_argument('--test', type=int, nargs="+", required=True, help="Test subsets.")
parser.add_argument('--val', type=int, nargs="+", required=True, help="Validation subsets.")


def generate_dataset(path, train, test, val):
    outputs = os.path.join(path, "data_2d_semantics", "*")
    scenes_outputs = glob.glob(outputs)

    df = pd.DataFrame(columns = ['x','y','subset'])

    for s in scenes_outputs:
        print("=== Processing ",os.path.basename(s), " ===")
        for i in (train + test + val):
            scene = "2013_05_28_drive_{0:04d}_sync".format(i)
            
            if i in train:
                subset = "train"
            elif i in test:
                subset = "test"
            else:
                subset = "val"

            if scene in s:
                # list all usuable files from scene output masks
                output_files_path = os.path.join(s, "*.png")
                output_files = glob.glob(output_files_path)                

                progress = 0
                print("Progress: {:>3} %".format( progress * 100 / len(output_files) ), end=' ')
                for f in output_files:
                    new_f = f.replace("data_2d_semantics","data_2d_raw")
                    new_f = new_f.replace(scene + "/", scene + "/" + "image_00/data_rect/")
                    new_row = pd.Series({"x": new_f, "y": f, "subset": subset})
                    df = df.append(new_row, ignore_index=True)
                    
                    progress += 1
                    print("\rProgress: {:>3} %".format( progress * 100 / len(output_files) ), end=' ')
                    sys.stdout.flush()
                print("\n")
                break

    df.to_csv("kitti360_dataset.csv", index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    generate_dataset(args.dataset, args.train, args.test, args.val)