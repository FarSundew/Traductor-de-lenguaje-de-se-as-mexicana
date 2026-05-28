import pickle

from sklearn.model_selection import train_test_split 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore

with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asanyarray(data_dict['data'])
labels = np.asanyarray(data_dict['labels']).astype(int)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

num_classes = np.max(labels) + 1

model = Sequential([
    Dense(128, activation='relu', input_shape=(len(data[0]),)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test)
print('{}% of samples were classified correctly!'.format(accuracy * 100))

model.save('model.keras')
