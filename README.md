# Semantic-Segmentation-Thesis


## Prepare Labels

For preparing the labels, you must run __prepare_data.py__ inside *utils*.
If you already downloaded the labels, please make sure the folder containing them is named __dataset__, if not the program will take care of it.

```
$ python prepare_data.py --input path/to/input --output path/to/output
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

Then run:
```
$ python dataset_create.py --dataset path/to/dataset --test . . . --train . . . --val . . .
```