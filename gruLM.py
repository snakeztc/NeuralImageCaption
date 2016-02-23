from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import TimeDistributedDense
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

nb_word = 1000
maxlen = 10  # cut texts after this number of words (among top max_features most common words)
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
X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post', value=-1.0)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post', value=-1.0)
label_train = sequence.pad_sequences(label_train, maxlen=maxlen, padding='post', value=-1.0)
label_test = sequence.pad_sequences(label_test, maxlen=maxlen, padding='post', value=-1.0)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# make index larger be 1-based
X_train += 1
X_test += 1
label_train += 1
label_test += 1

print np.max(label_train)
print np.min(label_train)

# to one-hot for Y
Y_train = np.zeros((label_train.shape[0], maxlen, nb_word), dtype=np.bool)
for i, s in enumerate(label_train):
    for t, w in enumerate(s):
        if w > 0:
            Y_train[i, t, w-1] = 1


print('Build model...')
model = Sequential()
model.add(Embedding(nb_word, 200, input_length=maxlen, mask_zero=True))
model.add(GRU(128, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.2))
model.add(TimeDistributedDense(nb_word))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Train...")
nb_epoch = 20
num_samples = X_train.shape[0]
cur_index = np.arange(num_samples)
nb_batches = int(np.ceil(num_samples / float(batch_size)))


def get_perplexity(m, X, label):
    prob = m.predict_proba(X, verbose=False)
    num_tokens = 0
    sum_neg_prob = 0.0
    for i, s in enumerate(label):
        for t, w in enumerate(s):
            sum_neg_prob += np.log2(prob[i, t, w-1])
            num_tokens += 1
    return pow(2, -1 * sum_neg_prob/num_tokens)


for i_epoch in range(20):
    print 'Epoch ' + str(i_epoch)
    # shuffle data
    np.random.shuffle(cur_index)
    for iter_idx in range(nb_batches):
        start_idx = iter_idx * batch_size
        end_idx = np.min([(iter_idx+1) * batch_size, num_samples])
        mini_batch_index = cur_index[start_idx:end_idx]
        model.fit(X_train[mini_batch_index, :], Y_train[mini_batch_index, :, :], batch_size=batch_size,
                  nb_epoch=1, verbose=False)

    # calculate validation perplexity
    print "Training perplexity is " + str(get_perplexity(model, X_train, label_train))
    print  "Validation perplexity is " + str(get_perplexity(model, X_test, label_test))


