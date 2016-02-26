from keras.preprocessing import sequence
import numpy as np


class CorpusFactory(object):

    @staticmethod
    def next_token_prediction(train, test, maxlen):
        X_train = [s[0:-1] for s in train]
        X_test = [s[0:-1] for s in test]
        label_train = [s[1:] for s in train]
        label_test = [s[1:] for s in test]

        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='pre', value=-1.0)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='pre', value=-1.0)
        label_train = sequence.pad_sequences(label_train, maxlen=maxlen, padding='pre', value=-1.0)
        label_test = sequence.pad_sequences(label_test, maxlen=maxlen, padding='pre', value=-1.0)

        # add 1 so that words are from 1 to N and 0 is mask
        X_train += 1
        X_test += 1
        label_train += 1
        label_test += 1

        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        return X_train, label_train, X_test, label_test


    @staticmethod
    def get_ngram(text, n, step):
        sentences = []
        next_chars = []
        for s in text:
            for i in range(0, len(s) - n, step):
                sentences.append(s[i: i + n])
                next_chars.append(s[i + n])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        X = sequence.pad_sequences(sentences, maxlen=n, padding='post', value=0.0)
        y = np.array(next_chars)
        return (X, y)


    @staticmethod
    def ngram_prediction(train, test, n):
        (X_train, label_train) = CorpusFactory.get_ngram(train, n, 1)
        (X_test, label_test) = CorpusFactory.get_ngram(test, n, 1)
        return X_train, label_train, X_test, label_test

