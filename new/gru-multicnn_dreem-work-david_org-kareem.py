#!/usr/bin/env python
# coding: utf-8

# In[4]:


# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import Sequential
# # from mne.decoding import SPoC
# import tensorflow.keras.backend as K
# from keras import Input ,Model
# from keras.layers import RepeatVector, Permute ,Activation
# from tensorflow.keras.backend import squeeze, dot, expand_dims,tanh, exp
# from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
# from tensorflow.keras.layers import Layer
# from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Flatten, Conv2D, MaxPooling2D, Conv1D,MaxPool2D,GRU, Dropout, MaxPool1D, ConvLSTM2D,MaxPooling1D, ReLU, LeakyReLU, RNN, SimpleRNN, GRU, LSTM, TimeDistributed
# from tensorflow.keras.losses import sparse_categorical_crossentropy
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.layers import BatchNormalization
# from sklearn.model_selection import KFold
# from scipy.interpolate import interp1d
# import tensorflow as tf
# from scipy.signal import stft, spectrogram
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import h5py # Read and write HDF5 files from Python
# import os
# import matplotlib.pyplot as plt
# import matplotlib
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from sklearn.metrics import confusion_matrix
# from keras import Input ,Model
# from tensorflow.keras.layers import concatenate
import h5py  # Read and write HDF5 files from Python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras import Input, Model
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.signal import stft, spectrogram
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalMaxPool1D,LSTM,BatchNormalization,MaxPooling1D, Dense, Flatten, Conv1D, Dropout, GRU, TimeDistributed, LeakyReLU, MaxPool1D, GlobalAveragePooling1D
# from mne.decoding import SPoCMaxPooling1D,
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import scipy.io as sio
# from hyperas.distributions import choice, uniform
data_path = r"C:\Users\eidan\Documents\BME_Dreem\new"
file_xtrain = data_path + r"/X_train.h5"
file_xtest = data_path + r"/X_test.h5"
file_ytrain = data_path + r"/y_train.csv"

a = pd.read_csv('1.csv')