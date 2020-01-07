import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

seed = 3
np.random.seed(seed)


def one_hot_encode_object_array(arr):
    return np_utils.to_categorical(arr)


datas = np.array([[5.1, 3.5, 1.4, 0.2],
                  [4.9, 3., 1.4, 0.2],
                  [4.7, 3.2, 1.3, 0.2],
                  [4.6, 3.1, 1.5, 0.2]
                  ])

label = np.array([[0],
                  [1],
                  [0],
                  [1]
                  ])
print(datas)

train_X, test_X, train_y, test_y = train_test_split(datas, label, test_size=0.2, random_state=seed)

train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)

model = Sequential()

model.add(Dense(16, input_shape=(4,), activation='sigmoid'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(train_X, train_y_ohe, epochs=100, batch_size=1, verbose=0)
loss, accuracy = model.evaluate(test_X, test_y_ohe, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
