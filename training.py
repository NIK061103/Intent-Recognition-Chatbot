import json
import numpy as np
from nlp import tokenize, stemming
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

with open("chatbot project/dataset.json") as f:
    data = json.load(f)

words = []
tags = []
x_data = []
y_data = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # print(pattern)
        temp_tok_wrds = tokenize(pattern)
        words.extend(temp_tok_wrds)
        # print(words)
        x_data.append(temp_tok_wrds)
        # print(x_data)
        y_data.append(intent["tag"])

        if intent["tag"] not in tags:
            tags.append(intent["tag"])

words = stemming(words)
words = sorted(list(set(words)))
tags = sorted(list(tags))

# print(x_data)
# print("\n\n")
# print(y_data)
# print("\n\n")
# print(words)

x_train = []
y_train = []
output = [0 for i in range(len(tags))]

for x, x_dox in enumerate(x_data):
    bag = []
    temp_stemmed = stemming(x_dox)
    for word in words:
        if word in temp_stemmed:
            bag.append(1)
        else:
            bag.append(0)

    output_row = output[:]
    output_row[tags.index(y_data[x])] = 1

    x_train.append(bag)
    y_train.append(output_row)

x_train = np.array(x_train)
# x_train = np.reshape(x_train, -1)
y_train = np.array(y_train)
# y_train = np.reshape(y_train, -1)

# print(x_train)
# print("\n\n")
# print(y_train)


modelf = Sequential()
modelf.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
modelf.add(Dropout(0.5))
modelf.add(Dense(64, activation='relu'))
modelf.add(Dropout(0.5))
modelf.add(Dense(len(y_train[0]), activation='softmax'))

# model = load_model('chatbot_model.h5')


sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
modelf.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


hist = modelf.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
modelf.save('chatbot_model_f.h5', hist)
