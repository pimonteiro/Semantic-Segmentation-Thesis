#!/bin/bash

#Uncomment to select files do download
train_list=("2013_05_28_drive_0000_sync"
            "2013_05_28_drive_0002_sync"
	         "2013_05_28_drive_0003_sync")
           "2013_05_28_drive_0004_sync" 
           "2013_05_28_drive_0005_sync" 
           "2013_05_28_drive_0006_sync" 
           "2013_05_28_drive_0007_sync" 
           "2013_05_28_drive_0009_sync" 
	        "2013_05_28_drive_0010_sync")

# Left and Right cameras
cam_list=("00")

root_dir="${1}/KITTI-360"
data_2d_dir=data_2d_raw

mkdir -p $root_dir
mkdir -p $root_dir/$data_2d_dir

cd $root_dir 

# perspective images
for sequence in ${train_list[@]}; do
   for camera in ${cam_list[@]}; do 
	zip_file=${sequence}_image_${camera}.zip
       wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file}
	unzip -d ${data_2d_dir} ${zip_file} 
	rm ${zip_file}
   done
done

# timestamps
#zip_file=data_timestamps_perspective.zip
#wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file}
#unzip -d ${data_2d_dir} ${zip_file}
#rm $zip_file



# Download semantic labels
#zip_file=data_2d_semantics.zip
#wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ed180d24c0a144f2f1ac71c2c655a3e986517ed8/${zip_file}
#unzip ${zip_file}
#rm $zip_file


# Download calibration settings and vehicle poses
#zip_file=calibration.zip
#wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/${zip_file}
#unzip ${zip_file}
#rm $zip_file

#zip_file=data_poses.zip
#wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/${zip_file}
#unzip ${zip_file}
#rm $zip_file