# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:00:08 2023

@author: ang
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers import TextVectorization

# The way TensorFlow/Keras takes data for this multi-label classifier is through 
# directories. 
batch_size = 256*32
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

# SET VOCABULARY SIZE
VOCAB_SIZE = 10000

# INSTANTIATE TEXT VECTORIZE LAYER 
_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize='lower_and_strip_punctuation',
    output_mode='multi_hot')

MAX_SEQUENCE_LENGTH = 1000

# RAW TRAINING DATASET IN MEMORY 
train_text = raw_train_ds.map(lambda text, labels: text)
_vectorize_layer.adapt(train_text)

# BUILD THE MODEL WITH LAYERS 
_model = tf.keras.Sequential([
                              _vectorize_layer,
                              layers.Dense(128, activation="relu", name="dense_1"),
                              layers.Dense(128, activation="relu", name="dense_2"),
                              layers.Dense(128, activation="relu", name="dense_3"),
                              layers.Dense(14, activation="softmax", name="output_preds")
                              ])

'''
# Ran a few tests with the SGD; went with adam opt.
# _sgd = SGD(learning_rate=0.96, momentum=0.96) 
Epoch 2000/2000
4/4 [==============================] - 1s 211ms/step - loss: 0.3893 - SubsetAccuracy: 0.8405 - HammingLoss: 0.0714 - val_loss: 0.3889 - val_SubsetAccuracy: 0.8414 - val_HammingLoss: 0.0714
'''

# MODEL 'Accuracy' AND SparseCategoricalAccuracy AND SubsetAccuracy HERE ARE SYNONYMOUS
_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=[
             tf.metrics.SparseCategoricalAccuracy(name="SubsetAccuracy"),
             tfa.metrics.HammingLoss(name='HammingLoss',mode='multiclass'),
             ])

_epochs = 2302
bin_history = _model.fit(
    raw_train_ds, validation_data=raw_val_ds, epochs=_epochs)

print('Done.')