# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:07:01 2023

@author: ang
"""
import pandas as pd 
import numpy as np 
import tensorflow as tf
import tensorflow_addons as tfa
pd.options.display.max_columns = None 

# One way TensorFlow/Keras takes data for this multi-label classifier is through 
# directories. Batch size is not relevant because we are not training. This is placing 
# all of the training data into one Tensor--for ease of access. 
batch_size = 40000
seed = 42

# INPUTS; Given a feature request what are the refactorings to be. 
# SPLIT THE DATA 80/20 SPLIT HERE
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
                                                        r'.\data\multi_classifier_data_', 
                                                        batch_size=batch_size,
                                                        validation_split=0.2,
                                                        subset='training',
                                                        seed=seed
                                                        )

# CREATE VALIDATION DATASET 
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
                                                        r'.\data\multi_classifier_data_', 
                                                        batch_size=batch_size,
                                                        validation_split=0.2,
                                                        subset='training',
                                                        seed=seed
                                                        )

# THE TRAINING AND VALIDATION DATA IS PLACED INTO MEMORY HERE FOR BETTER 
# MODEL TRAINING. THE MODEL WILL NOT NEED TO REOPEN THE FILES EVERY EPOCH 
# WITH THESE TWO LINES CACHING THE DATA TO MEMORY. 
raw_train_ds = raw_train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
raw_val_ds = raw_val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

load_multi_model = tf.keras.models.load_model(r".\src\models\multi_class_model_epoch_2302_save_0_cpu_linear",
                                              custom_objects={'HammingLoss': tfa.metrics.HammingLoss})

def get_multi_class_train_preds():
    '''
    Function builds a predictions DataFrame. Where the data 
    include, predictions index in the training data, prediction 
    actual value, prediction value predicted and prediction text, 
    and missed prediction boolean. 

    Returns
    -------
    multi_class_missed_train_preds : DataFrame
        Pandas DataFrame. 
    '''
    # ASSIGN TRAINING DATA TO LISTS KEY (k) STRINGS AND VAL (v) ACTUAL VALS 
    for k_,v_ in raw_train_ds:
        k = list(k_)
        v = list(v_)
        break
    
    # EXTRACT STRINGS AND INDEX INT FROM k AND v 
    train_str_lst = []
    act_vals_lst = []
    for i in range(len(k)):
        train_str_lst.append(k[i].numpy().decode("utf-8"))
        act_vals_lst.append(v[i].numpy())
    
    # GET PREDICTIONS FROM THE TRAINING DATA 
    train_predictions = load_multi_model.predict(train_str_lst[0:])
    train_predictions_lst = [] 
    for i in range(len(train_predictions)):
        train_predictions_lst.append(np.argmax(train_predictions[i]))
    
    # GET LISTS OF MISSED PREDICTIONS INDEX, ACTUAL VAL, AND PREDICTED VAL 
    preds_idx = []
    preds_act_val = []
    preds_val_pred = []
    missed_pred_bool = []
    for index, (first, second) in enumerate(zip(act_vals_lst, train_predictions_lst)):
        
        preds_idx.append(index)
        preds_act_val.append(first)
        preds_val_pred.append(second)
        if first != second:
            missed_pred_bool.append(True)
        else: 
            missed_pred_bool.append(False)
    # GET TEXT DATA BY INDEX OF MISSED PREDS 
    preds_text = []
    for i in preds_idx:
        preds_text.append(k[i].numpy().decode("utf-8"))
    
    # HUMAN READABLE LABELS TO DICT 
    labels_lst = ['extract interface',
                  'extract method',
                  'extract superclass',
                  'inline method',
                  'move and rename class',
                  'move attribute',
                  'move class',
                  'move method',
                  'pull up attribute',
                  'pull up method',
                  'push down attribute',
                  'push down method',
                  'rename class',
                  'rename method']
    
    labels_dict = dict(zip([i for i in range(14)], labels_lst)) 
    
    # BUILD THE DATAFRAME 
    multi_class_train_preds = pd.DataFrame({'missed_preds_idx':preds_idx,
                                                   'preds_act_val':preds_act_val,
                                                   'preds_val_pred':preds_val_pred,
                                                   'preds_text':preds_text,
                                                   'missed_pred_bool':missed_pred_bool})
    
    # HUMAN READABLE LABELS TO DICT APPLIED 
    multi_class_train_preds.preds_act_val.replace(labels_dict,inplace=True)
    multi_class_train_preds.preds_val_pred.replace(labels_dict,inplace=True)
    
    return multi_class_train_preds
multi_class_train_preds = get_multi_class_train_preds() 

def get_confusion_matrix(): 
    confusion_matrix = pd.crosstab(multi_class_train_preds['preds_act_val'], 
                                   multi_class_train_preds['preds_val_pred'], 
                                   rownames=['Actual'], 
                                   colnames=['Predicted'],
                                   # normalize='columns'
                                   )
    total = 0 
    for i in confusion_matrix.columns:
        total += confusion_matrix[i].sum()
        
    # TP is where the index matches the column name Ex. confusion_matrix.iloc[0,0];confusion_matrix.iloc[1,1];confusion_matrix.iloc[2,2]
    TP_lst = []
    for i in range(len(confusion_matrix)):
        TP_lst.append(confusion_matrix.iloc[i,i])
        confusion_matrix.iloc[i,i] = None
        
    # FN is the sum of all other values in the row. 
    FN_lst = []
    for i in range(len(confusion_matrix)):
        FN_lst.append(confusion_matrix.iloc[i,].sum(skipna=True))
        
    # FP is the sum of all other values in the column. 
    FP_lst = []
    for i in range(len(confusion_matrix)):
        FP_lst.append(confusion_matrix.iloc[0:,i].sum(skipna=True))
    
    # Place the lists into an DataFrame and calculate the TN. 
    df_each_instance = pd.DataFrame({
                              'TP':TP_lst,
                              'FN':FN_lst,
                              'FP':FP_lst,
                              },index=[i for i in confusion_matrix.index])
    
    # TN as an equation is; total - (FP + FN + TP)
    df_each_instance['TN'] = total - (df_each_instance['TP']+df_each_instance['FN']+df_each_instance['FP'])
            
    # Get True Pos Rate (TPR), True Neg Rate (TNR), False Neg Rate (FNR), and False Pos Rate (FPR)
    df_each_instance['TPR'] = round(df_each_instance['TP']/(df_each_instance['TP']+df_each_instance['FN'])*100,2)
    df_each_instance['TNR'] = round(df_each_instance['TN']/(df_each_instance['TN']+df_each_instance['FP'])*100,2)
    df_each_instance['FNR'] = round(df_each_instance['FN']/(df_each_instance['FN']+df_each_instance['TP'])*100,2)
    df_each_instance['FPR'] = round(df_each_instance['FP']/(df_each_instance['FP']+df_each_instance['FN'])*100,2)
    
    return df_each_instance
df_each_instance = get_confusion_matrix()
