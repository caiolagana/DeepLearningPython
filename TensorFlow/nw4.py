"""
Following tutorial for word embedding techniques:
https://www.tensorflow.org/text/guide/word_embeddings
"""

import io
import os
import re
import shutil
import string
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.layers import TextVectorization



dataset_dir = os.path.join('.', 'aclImdb')
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

"""
The Embedding layer can be understood as a lookup table that maps from integer indices (which stand for specific words) to dense vectors (their embeddings). The dimensionality (or width) of the embedding is a parameter you can experiment with to see what works well for your problem, much in the same way you would experiment with the number of neurons in a Dense layer.
"""

# Embed a 1,000 word vocabulary into 5 dimensions.
# https://chat.openai.com/share/b9d6a82e-bca1-425f-bd63-db9d7b80d94f
embedding_layer = tf.keras.layers.Embedding(1000, 5)

