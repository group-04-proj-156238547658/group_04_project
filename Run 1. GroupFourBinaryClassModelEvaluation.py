# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:07:01 2023

@author: ang
"""
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import layers

pd.options.display.max_columns = None

# One way TensorFlow/Keras takes data for this classifier is through 
# directories. Batch size does not matter we are not training. This is placing 
# all of the training data into one Tensor--for ease of access. 
BATCH_SIZE = 13365
SEED = 42

# SPLIT THE DATA 80/20 SPLIT HERE
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
                                                        r'.\data\binary_classifier_data_keras_', 
                                                        batch_size=BATCH_SIZE, 
                                                        validation_split=0.2, 
                                                        subset='training', 
                                                        seed=SEED)

# CREATE VALIDATION DATASET 
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
                                                        r'.\data\binary_classifier_data_keras_', 
                                                        batch_size=BATCH_SIZE, 
                                                        validation_split=0.2, 
                                                        subset='validation', 
                                                        seed=SEED
                                                        )

# CREATE A KERAS LAYER 
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250

# VECTORIZE/TOKENIZE THE STRING DATA 
vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def get_vectorize_text(text):
    '''
    Function returns vectorized string. 

    Parameters
    ----------
    text : String
        Text for training or predicting. 

    Returns
    -------
    String
        Vectorized string. 
    '''
    
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)

def get_binary_class_train_preds():
    '''
    Function builds a predictions DataFrame. Where the data 
    include, predictions index in the training data, prediction 
    actual value, prediction value predicted and prediction text, missed prediction 
    boolean. 

    Returns
    -------
    binary_class_missed_train_preds : DataFrame
        Pandas DataFrame. 

    '''
    load_bin_model = tf.keras.models.load_model(r".\src\models\binary_class_model_epoch_40")
    
    # ASSIGN TRAINING DATA TO LISTS KEY (k) STRINGS AND VAL (v) ACTUAL VALS
    for k,v in raw_train_ds:
        k = list(k)
        v = list(v)
        break
    
    # EXTRACT STRINGS AND INDEX INT FROM k AND v
    train_str_lst = []
    act_vals_lst = []
    for i in range(len(k)):
        train_str_lst.append(k[i].numpy().decode("utf-8"))
        act_vals_lst.append(v[i].numpy())
    
    # GET PREDICTIONS FROM THE TRAINING DATA 
    train_predictions = load_bin_model.predict(get_vectorize_text(train_str_lst[0:]))
    train_predictions_lst = []
    for i in range(len(train_predictions)):
        train_predictions_lst.append(int(train_predictions[i][0].round(0)))
    
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
    # GET TEXT DATA BY INDEX OF MISSED PREDS (missed_preds_idx) 
    preds_text = []
    for i in preds_idx:
        preds_text.append(k[i].numpy().decode("utf-8"))
    
    labels_lst = ['no_refactor_needed','refactor_needed']
    
    labels_dict = dict(zip([i for i in range(2)], labels_lst))
    
    # BUILD THE DATAFRAME 
    binary_class_train_preds = pd.DataFrame({'missed_preds_idx':preds_idx,
                                                   'preds_act_val':preds_act_val,
                                                   'preds_val_pred':preds_val_pred,
                                                   'preds_text':preds_text,
                                                   'missed_pred_bool':missed_pred_bool})
    
    # HUMAN READABLE LABELS TO DICT APPLIED 
    binary_class_train_preds.preds_act_val.replace(labels_dict,inplace=True)
    binary_class_train_preds.preds_val_pred.replace(labels_dict,inplace=True)
    
    return binary_class_train_preds

binary_class_train_preds = get_binary_class_train_preds()