# Semantic-Segmentation-Thesis

## Prepare Labels

For preparing the labels, you must run __prepare_data.py__ inside *utils*.
If you already downloaded the labels, please make sure the folder containing them is named __dataset__, if not the program will not work.

```
python prepare_data.py --input path/to/input --output path/to/output
```

## Dataset Creation

In order to create the dataset (to split into train, test and val) ensure its folder follow the following structure:

```
├── data_2d_raw                       # Input images
│   ├── 2013_05_28_drive_0000_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   ├── 2013_05_28_drive_0002_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   ├── 2013_05_28_drive_0003_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   ├── 2013_05_28_drive_0004_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   ├── 2013_05_28_drive_0005_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   ├── 2013_05_28_drive_0006_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   ├── 2013_05_28_drive_0007_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   ├── 2013_05_28_drive_0009_sync
│   │   └── image_00
│   │       └── data_rect
│   │           └── *.png
│   └── 2013_05_28_drive_0010_sync
│       └── image_00
│           └── data_rect
│   │           └── *.png
└── data_2d_semantics                  # Output images
    ├── 2013_05_28_drive_0000_sync
    │   └── *.png
    ├── 2013_05_28_drive_0002_sync
    │   └── *.png
    ├── 2013_05_28_drive_0003_sync
    │   └── *.png
    ├── 2013_05_28_drive_0004_sync
    │   └── *.png
    ├── 2013_05_28_drive_0005_sync
    │   └── *.png
    ├── 2013_05_28_drive_0006_sync
    │   └── *.png
    ├── 2013_05_28_drive_0007_sync
    │   └── *.png
    ├── 2013_05_28_drive_0009_sync
    │   └── *.png
    └── 2013_05_28_drive_0010_sync

```
The output images must be the labels already converted, as mentioned previously.

Then run:
```
python dataset_create.py --dataset path/to/dataset --test . . . --train . . . --val . . .
```

## Evaluation of the model

To evaluate the model you can use both an already available pre-trained model on cityscapes or use a refined model by your choice. To run, simply do:
```
python evaluate.py --dataset path/do/dataset --model {xception,mobilenetv2} --batch_size ... --ouput path/to/output
```

or

```
python evaluate.py --dataset path/do/dataset --model_folder ... --batch_size ... --ouput path/to/output
```

## Training of the model

Training the model follows the same logic as evaluation:

```
python evaluate.py --dataset path/do/dataset --model {xception,mobilenetv2} --pretrained --freezed  --batch_size ... --epoch ... --name ...
```

or

```
python evaluate.py --dataset path/do/dataset --model_folder ... --pretrained --freezed  --batch_size ... --epoch ... --name ...
```

The ```--pretrained``` refers to using the pretrained version of the backbone model with cityscapes weights and ```--freezed``` refers to freezing all layers up until the final ones to refine the output.

Running the training script will generate two folders: __logs__ containing the tensorboard logs from the training and __weights__ containing the best weights achieved during training.
