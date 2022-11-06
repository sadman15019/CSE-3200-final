#!/bin/sh

##------------------------------------------generate_dataset.py------------------------------------------------
root_path="../dataset_folder/"
src_path="${root_path}raw_videos/"
save_img_path="${root_path}raw_images/"
output_imgs_path="../output_images/"
ppg_signal_dir="${root_path}ppg_signals.csv"
ppg_feats="${root_path}ppg_feats.csv"
xlsx_path="${root_path}Data.xlsx"
ppg_feats_labels="${root_path}ppg_feats_labels.csv"

python generate_dataset.py $src_path $save_img_path $output_imgs_path $ppg_signal_dir $ppg_feats $xlsx_path $ppg_feats_labels
#--------------------------------------------------------------------------------------------------------------
echo succeded

# ##------------------------------------------model_train.py-----------------------------------------------------
# root_path="../dataset_folder/"
# dataset_dir="${root_path}ppg_feats_labels.csv"

# python model_train.py $dataset_dir
# #--------------------------------------------------------------------------------------------------------------
# echo succeded