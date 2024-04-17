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

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

