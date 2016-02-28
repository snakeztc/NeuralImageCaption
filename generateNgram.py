from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
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
ngram = 4

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

train = val_indexes[0:train_size]
test = val_indexes[train_size:train_size+test_size]

print(len(train), 'train sequences')
print(len(test), 'test sequences')

(X_train, label_train, X_test, label_test) = CorpusFactory.ngram_prediction(train, test, ngram, nb_word)

Y_train = np.zeros((label_train.shape[0], nb_word), dtype=np.bool)
for i, w in enumerate(label_train):
    Y_train[i, w] = 1

print('Build model...')
model = Sequential()
model.add(Embedding(nb_word+1, 100, input_length=ngram, mask_zero=False)) # due to masking add 1
model.add(GRU(256, return_sequences=False))  # try using a GRU instead, for fun
model.add(Dropout(0.2))
model.add(Dense(nb_word))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.load_weights('models/')

print CorpusFactory.generate_caption(model, 5, maxlen, ngram, nb_word)