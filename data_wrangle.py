# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:46:51 2023

@author: ang
"""
import pandas as pd
import json 
import os 
import pickle

with open (r".\FR-Refactor-master_GROUP_04_ANNOTATED\FR-Refactor\FR-Refactor\src\dataset\binary_class_dataset.pickle", 'rb') as fp:
    binary_class_dataset = pickle.load(fp)

df_binary_class = pd.DataFrame({'Text':[i[0] for i in binary_class_dataset],
                   'Tags':[i[1] for i in binary_class_dataset]})

binary_labels_set = ['no_refactor_needed', 'refactor_needed']

'''
# ONLY NEED TO RUN THE CODE IN THIS COMMENT ONCE TO CREATE DIRECTORIES FOR THE 
# BINARY CNN; FOR LOOP BUILT THE DIRECTORIES FOR THE TENSORFLOW DATA FROM FR-Refactor 
# AUTHORS DATA 
for i in binary_labels_set:
    root_path = r'.\data\multi_classifier_data_'
    path = os.path.join(root_path, i) 
    os.mkdir(path) 

# ONLY NEED TO RUN ONCE TO CREATE TXT FILES FOR CNN 
# Convert the data into .txt readable by keras text_dataset_from_directory().
# Converts data and splits it. The data must be in .txt files and there must be 
# sub-directories for each respective label. 

idx_1 = 0
idx_2 = 0
for lst_s in binary_class_dataset:
    if lst_s[1] == 0: 
        with open(fr".\data\no_refactor_needed\a_text_{idx_1}.txt", "w") as output:
            output.write(lst_s[0])
        idx_1 += 1
    elif lst_s[1] == 1:
        with open(fr"E.\data\refactor_needed\b_text_{idx_2}.txt", "w") as output:
            output.write(lst_s[0])
        idx_2 += 1

'''
# THIS IS THE DATASET DERIVED FROM .\FR-Refactor-master\FR-Refactor\FR-Refactor\src\5. multi-label_classifiers.py
# df_X_Y_resampled.csv IS ALL THE DATA FOR THE MULTI-LABEL CLASSIFIER 
# FROM FR-Refactor BEFORE THE AUTHORS SPLIT THE DATA. THIS IS THE ENTIRE 
# DATASET FR-Refactor USED TO TRAIN/TEST THEIR MULTI-LABEL CLASSIFIERS 
df = pd.read_csv(r".\data\df_X_Y_resampled.csv")
refactor_labels_set = df.columns[1:]

'''
# ONLY RUN ONCE TO CREATE DIRECTORIES FOR MULTI-LABEL CNN
# FOR LOOP BUILT THE DIRECTORIES FOR THE TENSORFLOW DATA FROM FR-Refactor AUTHORS DATA 
for index, row in df.iterrows():
    row_text = row['Text']
    row_extract_inter = row['extract interface'] 
    row_extract_method = row['extract method']
    row_extract_superclass = row['extract superclass']
    row_inline_method = row['inline method'] 
    row_move_and_rename_class= row['move and rename class']
    row_move_attribute = row['move attribute']
    row_move_class = row['move class'] 
    row_move_method = row['move method']
    row_pull_up_attribute = row['pull up attribute']
    row_pull_up_method = row['pull up method'] 
    row_push_down_attribute = row['push down attribute']
    row_push_down_method = row['push down method']
    row_rename_class = row['rename class'] 
    row_rename_method = row['rename method']

    row_lst = [row_text,row_extract_inter, row_extract_method,
               row_extract_superclass,row_inline_method,
               row_move_and_rename_class,row_move_attribute,
               row_move_class,row_move_method,
               row_pull_up_attribute,row_pull_up_method,
               row_push_down_attribute,row_push_down_method,
               row_rename_class,row_rename_method]
    
    root_path = r'.\data\multi_classifier_data_'           
    for i in row_lst:
        if i == 1:
            # print(df.columns[row_lst.index(i)])
            method_label_ = df.columns[row_lst.index(i)]
            path = os.path.join(root_path, method_label_) 
            # print(path)
            with open(path+f'\\idx_{index}.txt', 'w', encoding="utf-8") as f:
                    f.write(row_text)
'''