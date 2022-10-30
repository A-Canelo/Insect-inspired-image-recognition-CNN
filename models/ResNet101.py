# ResNet101 CNN

# Angel Canelo 2022.10.06

# K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,”
# presented at the Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
# 2016, pp. 770–778. Accessed: Oct. 24, 2022. [Online].
# Available: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

###### import ######################
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pymatreader import read_mat
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as cb
from tensorflow.keras import backend
from scipy.io import savemat
import seaborn as sns
##################################
##################################
tf.keras.backend.clear_session()
######## Load data ###############
data = read_mat('../data/data_pattern_train.mat')
# print(data.keys())
input_im = np.expand_dims(data['Images']/255.0,axis=3)
n_perc = 0.3    # noise strength
im_noisy = input_im + n_perc * np.random.rand(input_im.shape[0], 244, 324, 1)  # adding noise to training data
im_label = data['Image_label']
# TEST data
data_test = read_mat('../data/data_pattern_test.mat')
# print(data.keys())
input_im_test = np.expand_dims(data_test['Images']/255.0,axis=3)
im_test_noisy = input_im_test + n_perc * np.random.rand(input_im_test.shape[0], 244, 324, 1)  # adding noise to test data
im_label_test = data_test['Image_label']
HEIGHT = 244
WIDTH = 324
n_out = 3
##################################
########### CNN Model ############
cnn_model = tf.keras.applications.resnet.ResNet101(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=[HEIGHT, WIDTH, 1],
    pooling=None,
    classes=n_out)
lr = 1e-3; bz = 20; nb_epochs = 100
# Loss function and optimizer algorithm
cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
# define model callbacks
checkpoint_filepath = '../weights/ResNet101_weights.h5'
checkpoint_callback = cb.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy',
    mode='max', save_best_only=True)
cbs = [cb.EarlyStopping(monitor='loss', min_delta=0, patience=30, restore_best_weights=True), checkpoint_callback]
# train
history = cnn_model.fit(im_noisy, im_label, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_data=(im_test_noisy, im_label_test), shuffle=True)
cnn_model.load_weights("../weights/ResNet101_weights.h5")
#########################################
######## Plotting results ###############
to_mat = {"hist_acc": history.history['accuracy'], "hist_testacc":history.history['val_accuracy'],
          "topmax": np.max(history.history['accuracy']), "topmax_test": np.max(history.history['val_accuracy'])}
savemat("../data/ResNet101_perf.mat",  to_mat)
sns.set()
plt.figure()
plt.plot(history.history['accuracy'],color="blue"); plt.plot(history.history['val_accuracy'],color="red")
plt.title('Performance on pattern dataset'); plt.ylabel('Accuracy'); plt.xlabel('Iterations')
plt.legend(['train accuracy (3000 frames)', 'test accuracy (300 frames)'], loc='lower right')
plt.show()