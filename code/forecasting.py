import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# fixing the random seed to have a better reproducibility
seed = 3
np.random.seed(seed)

# input datas
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

train_y_ohe = np_utils.to_categorical(train_y)
test_y_ohe = np_utils.to_categorical(test_y)


# building the model
def createModel(inputLayer, lLayer):
    model = Sequential()
    model.add(Dense(inputLayer[1], input_shape=inputLayer[0], activation='sigmoid'))
    for foo in lLayer:
        if foo[0] == 'D':
            if foo[2] == "sig":
                model.add(Dense(foo[1], activation='sigmoid'))
            elif foo[2] == "soft":
                model.add(Dense(foo[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model


model = createModel([(4,), 16], [('D', 16, 'sig'), ('D', 2, 'soft')])

# training it
model.fit(train_X, train_y_ohe, epochs=100, batch_size=1, verbose=0)

# testing it
loss, accuracy = model.evaluate(test_X, test_y_ohe, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))

# plotting

