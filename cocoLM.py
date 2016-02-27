from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import TimeDistributedDense
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
from corpusFactory import CorpusFactory
import numpy as np
from pycocotools.coco import COCO

dataDir='.'
dataType='val2014'
annFile = '%s/annotations/captions_%s.json'%(dataDir,dataType)
caps=COCO(annFile)

anns = caps.loadAnns(caps.getAnnIds())

train_size = 1000
test_size = 10

val_data = [ann['caption'] for ann in anns[0:train_size+test_size]]

print "Tokenize the data"
val_data = [word_tokenize(s) for s in val_data]
maxlen = np.max([len(s) for s in val_data])

# get vocabulary
vocab = list(sorted(set(w for s in val_data for w in s)))
nb_word = len(vocab)

print "Validation data has a vocab size " + str(len(vocab))

# convert the data to index
val_indexes = []
for s in val_data:
    index_s = []
    for w in s:
        index_s.append(vocab.index(w))
    val_indexes.append(index_s)

print val_data[0]

print "Validation set has " + str(len(val_data)) + " sentences with max length " + str(maxlen)

batch_size = 32

# shuffle the data so evenly distribtued
train = val_indexes[0:train_size]
test = val_indexes[train_size:train_size+test_size]

print(len(train), 'train sequences')
print(len(test), 'test sequences')

print('Sorting the training data in ascending length')
train = sorted(train, lambda x,y: 1 if len(x)>len(y) else -1 if len(x)<len(y) else 0)


(X_train, label_train, X_test, label_test) = CorpusFactory.next_token_prediction(train, test, maxlen, nb_word)

print np.max(label_train)
print np.min(label_train)

print('Build model...')
model = Sequential()
model.add(Embedding(nb_word+2, 100, input_length=maxlen, mask_zero=True)) # due to masking add 1 and BOS
model.add(GRU(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(nb_word))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# to one-hot for Y
Y_train = np.zeros((label_train.shape[0], maxlen, nb_word), dtype=np.bool)
for i, s in enumerate(label_train):
    for t, w in enumerate(s):
        if w > 0:
            Y_train[i, t, w-1] = 1

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
            if w > 0:
                sum_neg_prob += np.log2(prob[i, t, w-1])
                num_tokens += 1
    print "Number tokens " + str(num_tokens)
    return pow(2, -1 * sum_neg_prob/num_tokens)


for i_epoch in range(20):
    print 'Epoch ' + str(i_epoch)
    # shuffle data
    np.random.shuffle(cur_index)
    for iter_idx in range(nb_batches):
        start_idx = iter_idx * batch_size
        end_idx = np.min([(iter_idx+1) * batch_size, num_samples])
        mini_batch_index = cur_index[start_idx:end_idx]

        model.fit(X_train[mini_batch_index, :], Y_train[mini_batch_index, :, :], batch_size=batch_size, nb_epoch=1, verbose=False)

    # calculate validation perplexity
    print "Training perplexity is " + str(get_perplexity(model, X_train, label_train))
    print  "Validation perplexity is " + str(get_perplexity(model, X_test, label_test))
    model.save_weights('./models/'+str(i_epoch)+'-gru.h5')





