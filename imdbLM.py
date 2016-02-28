from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import TimeDistributedDense
from keras.datasets import imdb
from keras.preprocessing import sequence
from corpusFactory import CorpusFactory
import numpy as np

nb_word = 5000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
train_size = 5000
test_size = 1000
nb_epoch = 50


print('Loading data...')
train, test = imdb.load_data(nb_words=nb_word, test_split=0.2, seed=100)

# discard the label since we are training language model
train = train[0]
test = test[0]
train = train[0:train_size]
test = test[0:test_size]

print("Gathering statistics of data")
train_sent_len = np.mean([len(s) for s in train])
test_sent_len = np.mean([len(s) for s in test])
print(train_sent_len, " average train sentence length")
print(test_sent_len, " average test sentence length")


print(len(train), 'train sequences')
print(len(test), 'test sequences')

print('Sorting the training data in ascending length')
train = sorted(train, lambda x,y: 1 if len(x)>len(y) else -1 if len(x)<len(y) else 0)

(X_train, label_train, X_test, label_test) = CorpusFactory.next_token_prediction(train, test, maxlen, nb_word)

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
model.add(Embedding(nb_word+1, 100, input_length=maxlen, mask_zero=True)) # due to masking add 1
model.add(GRU(256, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.2))
model.add(TimeDistributedDense(nb_word))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Train...")
num_samples = X_train.shape[0]
cur_index = np.arange(num_samples)
nb_batches = int(np.ceil(num_samples / float(batch_size)))


def get_perplexity(m, X, label):
    prob = m.predict_proba(X, verbose=False)
    num_tokens = 0
    sum_neg_prob = 0.0
    for i, s in enumerate(label):
        for t, w in enumerate(s):
            if w > 0:
                sum_neg_prob += np.log2(prob[i, t, w-1])
                num_tokens += 1
    print "Number tokens " + str(num_tokens)
    return pow(2, -1 * sum_neg_prob/num_tokens)


for i_epoch in range(nb_epoch):
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
    print "Validation perplexity is " + str(get_perplexity(model, X_test, label_test))
    model.save_weights('./models/'+str(i_epoch)+'-gru-imdb.h5')


