# Drosophila-inspired CNN for image recognition

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
import seaborn as sns
##################################
##################################
tf.keras.backend.clear_session()
######## Load data ###############
data = read_mat('../data/Pattern_data_CollRectSqu_train.mat')
input_im = np.expand_dims(data['Images']/255.0,axis=3)
n_perc = 0.3    # noise strength
im_noisy = input_im + n_perc * np.random.rand(input_im.shape[0], 244, 324, 1)  # adding noise to training data
im_label = data['Image_label']
# TEST data
data_test = read_mat('../data/Pattern_data_CollRectSqu_test.mat')
input_im_test = np.expand_dims(data_test['Images']/255.0,axis=3)
im_test_noisy = input_im_test + n_perc * np.random.rand(input_im_test.shape[0], 244, 324, 1)  # adding noise to test data
im_label_test = data_test['Image_label']
##################################
########### CNN Model ############
filter_exc = RandomUniform(minval=0.01, maxval=0.1) # 'glorot_uniform'
filter_inh = RandomUniform(minval=-0.1, maxval=-0.01)
inputs = layers.Input(shape=[244, 324, 1])      # (height, width, channels)

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
                  activation='linear', padding='same', name='LC4')(LC4)     # Jan M. Ache et ai. 2019 (Direct conection between Lo2, Lo4 to LC11)
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
CB = layers.Dense(3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3),
                       activity_regularizer=l1(1e-3), activation='softmax', name='classification')(CB)

fly_dnn = Model(inputs=inputs, outputs=CB, name='Fly_DNN')
lr = 1e-3; bz = 20; nb_epochs = 100 #val_split = 0.15

# Loss function and optimizer algorithm
fly_dnn.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
# define model callbacks
checkpoint_filepath = '../weights/fly_CNN.h5'
checkpoint_callback = cb.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy',
    mode='max', save_best_only=True)
cbs = [cb.EarlyStopping(monitor='loss', min_delta=0, patience=30, restore_best_weights=True), checkpoint_callback]    # monitor='val_loss'
# train
history = fly_dnn.fit(im_noisy, im_label, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_data=(im_test_noisy, im_label_test), shuffle=True)    #validation_split=val_split
#########################################
###### Quantize to INTEGER for GAP8######
fly_dnn.load_weights("../weights/fly_CNN.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(fly_dnn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_data_gen():
    data = tf.data.Dataset.from_tensor_slices(tf.cast(im_test_noisy, tf.float32)).batch(1).take(300)
    for input_value in data:
        yield [input_value]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.experimental_new_converter = False
converter.experimental_new_quantizer = False
tflite_model = converter.convert()
open('../weights/fly_CNN_8bit.tflite', 'wb').write(tflite_model)
#########################################
######## Plotting results ###############
sns.set()
plt.figure()
plt.plot(history.history['accuracy'],color="blue"); plt.plot(history.history['val_accuracy'],color="red")
plt.title('Performance on classification (EXC/INH initialization)'); plt.ylabel('Accuracy'); plt.xlabel('Iterations')  #Performance on classification (random initialization)
plt.legend(['train accuracy (3000 frames)', 'test accuracy (300 frames)'], loc='lower right')
plt.show()