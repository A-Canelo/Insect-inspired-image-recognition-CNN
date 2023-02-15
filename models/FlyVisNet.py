# FlyVisNet CNN

# Angel Canelo 2022.10.06

# References:
# From Photons to Behaviors: Neural Implementations of Visual Behaviors in Drosophila. Ryu et ai. 2022
# Object-Detecting Neurons in Drosophila. Mehmet F. Keles et ai. 2017
# Non-canonical Receptive Field Properties and Neuromodulation of Feature-Detecting Neurons in Flies. Stadele et ai. 2020
# Inhibitory Interactions and Columnar Inputs to an Object Motion Detector in Drosophila. Mehmet F. Keles et ai. 2020
# Neural Basis for Looming Size and Velocity Encoding in the Drosophila Giant Fiber Escape Pathway. Jan M. Ache et ai. 2019
# Ultra-selective looming detection from radial motion opponency. Nathan C. Klapoetke et ai. 2017

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
filter_exc = RandomUniform(minval=0.01, maxval=0.1)
filter_inh = RandomUniform(minval=-0.1, maxval=-0.01)
inputs = layers.Input(shape=[HEIGHT, WIDTH, 1])      # (height, width, channels)

# LAMINA
L1 = layers.Conv2D(1, 5, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_inh,
                  activation='linear', padding='same', name='L1')(inputs)
L1 = layers.MaxPooling2D(pool_size=(2, 2))(L1)
L2 = layers.Conv2D(1, 5, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='L2')(inputs)
L2 = layers.MaxPooling2D(pool_size=(2, 2))(L2)
L3 = layers.Conv2D(1, 5, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='L3')(inputs)
L3 = layers.MaxPooling2D(pool_size=(2, 2))(L3)
# MEDULLA
Mi1Tm3 = layers.Conv2D(2, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='Mi1Tm3')(L1)
Mi1Tm3 = layers.MaxPooling2D(pool_size=(2, 2))(Mi1Tm3)

C3 = layers.Concatenate()([L1, L3])
C3 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_inh,
                  activation='linear', padding='same', name='C3')(C3)
C3 = layers.MaxPooling2D(pool_size=(2, 2))(C3)
Tm124 = layers.Conv2D(3, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='Tm124')(L2)
Tm124 = layers.MaxPooling2D(pool_size=(2, 2))(Tm124)
Mi9 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_inh,
                  activation='linear', padding='same', name='Mi9')(L3)
Mi9 = layers.MaxPooling2D(pool_size=(2, 2))(Mi9)
Tm9 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='Tm9')(L3)
Tm9 = layers.MaxPooling2D(pool_size=(2, 2))(Tm9)

T2 = layers.Concatenate()([Tm124,Tm9,Mi1Tm3,C3,Mi9])
T2 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='T2')(T2)  # All medullar inputs since T2 is ON/OFF sensitive
T2 = layers.MaxPooling2D(pool_size=(2, 2))(T2)                                              # Mehmet F. Keles et ai 2020

T3 = layers.Concatenate()([Tm124,Tm9,Mi1Tm3,C3,Mi9])
T3 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_inh,
                  activation='linear', padding='same', name='T3')(T3)  # All medullar inputs since T3 is ON/OFF sensitive
T3 = layers.MaxPooling2D(pool_size=(2, 2))(T3)                                              # Mehmet F. Keles et ai 2020

T4 = layers.Concatenate()([Mi1Tm3,C3,Mi9])
T4 = layers.Conv2D(1, 5, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='T4')(T4)
T4 = layers.MaxPooling2D(pool_size=(2, 2))(T4)
# LOBULA
T5 = layers.Concatenate()([Tm124,Tm9])
T5 = layers.Conv2D(1, 5, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='T5')(T5)
T5 = layers.MaxPooling2D(pool_size=(2, 2))(T5)

LC11 = layers.Concatenate()([T2, T3])
LC11 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='LC11')(LC11)     # Mehmet F. Keles et ai 2020 (Direct conection between T2/T3 to LC11)
LC11 = layers.MaxPooling2D(pool_size=(2, 2))(LC11)                              # LC11 responds to small objects (spot/square)

LC15 = layers.Concatenate()([T4, T5])
LC15 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='LC15')(LC15)     # Stadele et ai 2020 (Indirect conection between T4/T5 to LC15 but unknown atm)
LC15 = layers.MaxPooling2D(pool_size=(2, 2))(LC15)                              # LC15 responds to bar-like objects (rectangle)

LC4 = layers.Concatenate()([T2, T3])
LC4 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='LC4')(LC4)     # Jan M. Ache et ai. 2019 (Direct conection between Lo2, Lo4 to LC4)
LC4 = layers.MaxPooling2D(pool_size=(2, 2))(LC4)                              # LC4 responds to expansion speed

LPLC2 = layers.Concatenate()([T4, T5])
LPLC2 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                  kernel_initializer=filter_exc,
                  activation='linear', padding='same', name='LPLC2')(LPLC2)     # Nathan C. Klapoetke et ai. 2017 (Direct conection between T4/T5 to LPLC2)
LPLC2 = layers.MaxPooling2D(pool_size=(2, 2))(LPLC2)                              # LPLC2 responds to expansion size

# OPTIC GLOMERULI (Collecting axons to central brain)
CB = layers.Concatenate()([LC11, LC15, LC4, LPLC2])
CB = layers.Flatten()(CB)
# CENTRAL BRAIN. Classification: Rectangle, Square
CB = layers.Dense(512, kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3),
                       activity_regularizer=l1(1e-3), activation='relu')(CB)
CB = layers.Dense(256, kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3),
                       activity_regularizer=l1(1e-3), activation='relu')(CB)
CB = layers.Dense(n_out, kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3),
                       activity_regularizer=l1(1e-3), activation='softmax', name='classification')(CB)

cnn_model = Model(inputs=inputs, outputs=CB, name='FlyVisNet')
lr = 1e-3; bz = 20; nb_epochs = 100
##################################
# Loss function and optimizer algorithm
cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
# define model callbacks
checkpoint_filepath = '../weights/FlyVisNet_weights.h5'
checkpoint_callback = cb.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy',
    mode='max', save_best_only=True)
cbs = [cb.EarlyStopping(monitor='loss', min_delta=0, patience=30, restore_best_weights=True), checkpoint_callback]
# train
history = cnn_model.fit(im_noisy, im_label, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_data=(im_test_noisy, im_label_test), shuffle=True)
#########################################
###### Quantize to INTEGER for GAP8######
cnn_model.load_weights("../weights/FlyVisNet_weights.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_data_gen():
    data = tf.data.Dataset.from_tensor_slices(tf.cast(im_test_noisy, tf.float32)).batch(1).take(300)
    for input_value in data:
        yield [input_value]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.experimental_new_converter = False
converter.experimental_new_quantizer = False
tflite_model = converter.convert()
open('../weights/FlyVisNet_8bit_weights.tflite', 'wb').write(tflite_model)
interpreter = tf.lite.Interpreter(model_path='../weights/FlyVisNet_8bit_weights.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
output_data = []
for i in range (300):
  interpreter.set_tensor(input_details[0]['index'], np.expand_dims(255*im_test_noisy[i,:,:,:].astype(np.uint8),axis=0))
  interpreter.invoke()
  output_data.append(interpreter.get_tensor(output_details[0]['index']))

true_max = np.argmax(im_label_test, axis=1)
test_max = np.argmax(np.squeeze(np.array(output_data)), axis=1)
matches = np.count_nonzero(true_max == test_max)/300
#########################################
######## Plotting results ###############
to_mat = {"hist_acc": history.history['accuracy'], "hist_testacc":history.history['val_accuracy'],
          "topmax": np.max(history.history['accuracy']), "topmax_test": np.max(history.history['val_accuracy']), "topmax_lite":matches}
savemat("../data/FlyVisNet_perf.mat",  to_mat)
sns.set()
plt.figure()
plt.plot(history.history['accuracy'],color="blue"); plt.plot(history.history['val_accuracy'],color="red")
plt.title('Performance on pattern dataset'); plt.ylabel('Accuracy'); plt.xlabel('Iterations')
plt.legend(['train accuracy (3000 frames)', 'test accuracy (300 frames)'], loc='lower right')
plt.show()