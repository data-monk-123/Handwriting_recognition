#import the necessary libraries
import tensorflow as tf
from keras.models import Sequential
from keras.datasets import mnist
# from tf.keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score
import gzip
import sys
import pickle
#
# f = gzip.open('mnist.pkl.gz', 'rb')
# if sys.version_info < (3,):
#     data = pickle.load(f)
# else:
#     data = pickle.load(f, encoding='bytes')
# f.close()
# (x_train, _), (x_test, _) = data

#load the data in the training and testing set
path = "C:\\Users\\Rajesh.Pandey\\PycharmProjects\\ImageAPI\\mnist.npz"
(X_train, y_train), (X_test, y_test)= tf.keras.datasets.mnist.load_data(path)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# X_train = X_train[:200]
# y_train = y_train[:200]

# # print(X_train.shape)
# # Plot the image
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
#
# plt.show()

shape = (28, 28, 1)
batch_size = 200

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# Now let's create the model
def create_model():
    left_input = tf.keras.layers.Input(shape = shape)
    filter = 32
    x = left_input
    for i in range(3):
        x = tf.keras.layers.Conv2D(filter,3, activation = "relu", padding = 'same')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        filter*=2
    right_input = tf.keras.layers.Input(shape = shape)
    y = right_input
    filter = 32
    for i  in range(3):
        y = tf.keras.layers.Conv2D(filter,3, activation = "relu", padding = 'same', dilation_rate = 2)(y)
        y = tf.keras.layers.Dropout(0.4)(y)
        y = tf.keras.layers.MaxPooling2D()(y)
        filter*=2
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    model = tf.keras.layers.concatenate([x, y])
    model = tf.keras.layers.Dropout(0.4)(model)
    output = tf.keras.layers.Dense(10, activation = "softmax")(model)
    model = tf.keras.models.Model([left_input, right_input], output)
    return model

model = create_model()
model.summary()

# tf.keras.utils.plot_model(model, show_shapes = True)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss= "categorical_crossentropy", metrics = ['accuracy'])

# print(X_train.shape)
# print(y_train.shape)
history = model.fit([X_train, X_train], y_train, batch_size=500, epochs=2, verbose=True, validation_split = 0.2,
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience= 20, monitor='val_loss', mode = 'min',
                                                                  restore_best_weights=True)])


#plotting the results
# import matplotlib.pyplot as plt
plt.figure(figsize = (20, 5))
plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['val_accuracy'], label = "val_accuracy")
plt.legend()

plt.figure(figsize = (20, 5))
plt.plot(history.history['loss'], label = "loss")
plt.plot(history.history['val_loss'], label= "val_loss")
plt.legend()

model.evaluate([X_train, X_train], y_train, batch_size= 128)

X_test = X_test.reshape(X_test.shape[0], 28, 28)
y_pred = model.predict([X_test, X_test])
# print(y_pred)


#Saving the model
model.save("ImageRecog.h5")









