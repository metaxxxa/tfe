#!/usr/bin/python3

import tensorflow as tf
from neuralnet_visualize.visualize import visualizer as nnviz

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(5,5),(1,1),input_shape=(128,128,3),activation='swish'),
    tf.keras.layers.Conv2D(64,(3,3),(1,1),activation='swish'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='swish'),
    tf.keras.layers.Dense(128, activation='swish'),
    tf.keras.layers.Dense(5, activation='softmax'),
])

model.compile()
net = nnviz()
net.from_tensorflow(model)
net.summarize()