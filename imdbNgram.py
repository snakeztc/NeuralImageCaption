from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.datasets import imdb
from corpusFactory import CorpusFactory
import numpy as np

nb_word = 1000
ngram = 4  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
train_size = 5000
test_size = 1000

print('Loading data...')
train, test = imdb.load_data(nb_words=nb_word, test_split=0.2)

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

(X_train, label_train, X_test, label_test) = CorpusFactory.ngram_prediction(train, test, ngram)

Y_train = np.zeros((label_train.shape[0], nb_word), dtype=np.bool)
for i, w in enumerate(label_train):
    Y_train[i, w] = 1


print('Build model...')
model = Sequential()
model.add(Embedding(nb_word, 100, input_length=ngram, mask_zero=False)) # due to masking add 1
model.add(GRU(256, return_sequences=False))  # try using a GRU instead, for fun
model.add(Dropout(0.2))
model.add(Dense(nb_word))
model.add(Activation('softmax'))
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
    for i, w in enumerate(label):
        sum_neg_prob += np.log2(prob[i, w])
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
        model.fit(X_train[mini_batch_index, :], Y_train[mini_batch_index, :], batch_size=batch_size, nb_epoch=1, verbose=False)

    # calculate validation perplexity
    print "Training perplexity is " + str(get_perplexity(model, X_train, label_train))
    print "Validation perplexity is " + str(get_perplexity(model, X_test, label_test))
    model.save_weights('./models/'+str(i_epoch)+'-model.h5')





