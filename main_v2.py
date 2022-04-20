import warnings
from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

warnings.filterwarnings('ignore')
np.random.seed(42)

data = np.load('etc_data_3d2.npz')
x_train = data['x_train']
y_train = tf.keras.utils.to_categorical(data['y_train'], num_classes=3)
scaler = StandardScaler()
shape1 = x_train.shape[1]
shape2 = x_train.shape[2]
x_train = np.reshape(x_train, (x_train.shape[0], shape1 * shape2))

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)

x_train = np.reshape(x_train, (x_train.shape[0], shape1, shape2))
x_valid = np.reshape(x_valid, (x_valid.shape[0], shape1, shape2))

dump(scaler, open('./crypto_scaler2.pkl', 'wb'))
# np.savez('etc_data_3d_split.npz', x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)

# data = np.load('etc_data_3d_split.npz')
# x_train = data['x_train']
# x_valid = data['x_valid']
# y_train = data['y_train']
# y_valid = data['y_valid']

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

print(x_train.shape[0], x_train.shape[1], x_train.shape[2])

model = Sequential()

model.add(InputLayer(input_shape=(x_train.shape[1], x_train.shape[2])))

model.add(LSTM(64, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(64))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss=['categorical_crossentropy'], metrics="accuracy", optimizer='adam')

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=5,
    restore_best_weights=True)

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5',
                                                monitor='val_loss',
                                                save_best_only=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,  patience=5)

BATCH_SIZE = 64

history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=500, callbacks=[reduce_lr, early_stopping, checkpoint_cb])

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')  # ‘bo’는 파란색 점을 의미합니다.
plt.plot(epochs, val_loss, 'b', label='Validation loss')  # ‘b’는 파란색 실선을 의미합니다.
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # 그래프를 초기화합니다.
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

model.save("crypto_model2")