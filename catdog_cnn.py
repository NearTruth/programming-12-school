import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import os

'''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)#force the model to take up more vram
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''

NAME = "catdog-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir=f'logs\\{NAME}')





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = pickle.load(open("datasets/X.pickle", "rb"))
Y = pickle.load(open("datasets/Y.pickle", "rb"))

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])
