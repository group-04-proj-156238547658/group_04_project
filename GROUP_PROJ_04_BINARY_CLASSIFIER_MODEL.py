# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 18:24:25 2023

@author: ang
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

BATCH_SIZE = 32
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

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# CHECK VECTORIZATION 
text_batch, label_batch = next(iter(raw_train_ds))
first_feat_req, first_label = text_batch[0], label_batch[0]

print("\nString Vectorized Example: ")
print("\n\nString to Vectorize: ", first_feat_req)
print("\n\nLabel: ", raw_train_ds.class_names[first_label])
print("\n\nVectorized String Example", vectorize_text(first_feat_req, first_label))

# APPLY TEXT VECTORIZED LAYER; THIS OCCURS BEFORE MODEL TRAINING 
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)

# CONFIG FOR PERFORMANCE 
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# CREATE THE MODEL 
embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(MAX_FEATURES , embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1, activation='sigmoid')
  ])

# create model
model.summary()

# LOSS FUNCTION OPTIMIZER 
METRICS = [
            tf.metrics.Precision(name='precision'),
            tf.metrics.Recall(name='recall'),
            tf.metrics.BinaryAccuracy(name='Accuracy'),
            tf.metrics.TruePositives(name='TP'),
            tf.metrics.TrueNegatives(name='TN'),
            tf.metrics.FalseNegatives(name='FN'),
            tf.metrics.FalsePositives(name='FP'),
            ]

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=METRICS,
              )

# FIT THE MODEL 
epochs = 40
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

def pretty_cm(TP, TN, FN, FP):
    '''
    Function prints the confusion matrix a little prettier and prints the 
    matrix and type one/two error and the AUC score. 
    '''
    print("\n\nCONFUSION MATRIX:\n")
    
    type_one = round(FP / (TN + FP),4)
    type_two = round(FN / (TP + FN),4)
    auc = round((1 - type_two) * (1 - type_one) / 2, 4)
    
    print(f'                 Predicted\n                  Pos   Neg \n Act     Pos  {TP}   {FP} \n          Neg  {FN}   {TN}')
    print(f'\nType One Err: {type_one}')
    print(f'Type Two Err: {type_two}')
    
    # 
    accuracy = (TP + TN)/(TP+TN+FN+FP)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1_score = 2*(precision*recall)/(precision+recall)
    hamming_loss=1-accuracy
    print('Accuracy: ', round(accuracy,4))
    print('F1 Score: ', round(f1_score,4))
    print('Precision: ', round(precision,4))
    print('Recall: ', round(recall,4))
    print('Hamming loss:', round(hamming_loss,4))
    
    return

