# Following https://www.tensorflow.org/tutorials/keras/text_classification
import tensorflow as tf
import os


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#dataset = tf.keras.utils.get_file("aclImdb_v1", url,untar=True, cache_dir='.',cache_subdir='')
#dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')


dataset_dir = os.path.join('.', 'aclImdb')
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

"""
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())
"""


batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

"""
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])
"""

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)


raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)