from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import TimeDistributedDense
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

nb_word = 1000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
train, test = imdb.load_data(nb_words=nb_word, test_split=0.2)

# discard the label since we are training language model
train = train[0]
test = test[0]
train = train[0:100]
test = test[0:10]

print(len(train), 'train sequences')
print(len(test), 'test sequences')

print('Sorting the training data in ascending length')
train = sorted(train, lambda x,y: 1 if len(x)>len(y) else -1 if len(x)<len(y) else 0)

print("Creating training data and targets")
X_train = [s[0:-1] for s in train]
X_test = [s[0:-1] for s in test]
label_train = [s[1:] for s in train]
label_test = [s[1:] for s in test]

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
label_train = sequence.pad_sequences(label_train, maxlen=maxlen)
label_test = sequence.pad_sequences(label_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# make index larger be 1-based
X_train += 1
X_test += 1
label_train += 1
label_test += 1

print np.max(label_train)
print np.min(label_train)


def get_one_hot(w, nb_word):
    one_hot = [0] * nb_word
    if w > 0:
        one_hot[w-1] = 1
    return one_hot

# to one-hot for Y
Y_train = []
for s in label_train:
    hot = []
    for w in s:
        one_hot = get_one_hot(w, nb_word)
        hot.append(one_hot)

    Y_train.append(hot)

Y_test = []
for s in label_test:
    hot = []
    for w in s:
        one_hot = get_one_hot(w, nb_word)
        hot.append(one_hot)
    Y_test.append(hot)


print('Build model...')
model = Sequential()
model.add(Embedding(nb_word, 300, input_length=maxlen, mask_zero=True))
model.add(GRU(512, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.2))
model.add(TimeDistributedDense(nb_word))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=20,
          validation_data=(X_test, Y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)