from keras.preprocessing import sequence
import numpy as np


class CorpusFactory(object):

    @staticmethod
    def generate_caption(m, nb_caption, maxlen, ngram, nb_word):
        print "Begin to generate captions"
        eos = [nb_word] * ngram
        captions = []
        for i in range(nb_caption):
            c = []
            w = eos
            for t in range(maxlen):
                nw = m.predict(w, verbose=False)
                c.append(nw)
                w = nw
            captions.append(c)
        return captions

    @staticmethod
    def next_token_prediction(train, test, maxlen, nb_word):
        """
        :param train: (num_sent, sent_len) index from 0 to nb_word - 1
        :param test: (num_sent, sent_len) index from 0 to nb_word - 1
        :param maxlen: max_sent length
        :param nb_word: nb_words
        :return: data in train and test with index from 0 to nb_word + 1 (nb_word + 1) is BOS (total nb_word+2 token type)
        """
        (X_train, label_train) = CorpusFactory.get_next_token(train, maxlen, nb_word)
        (X_test, label_test) = CorpusFactory.get_next_token(test, maxlen, nb_word)
        return X_train, label_train, X_test, label_test

    @staticmethod
    def get_next_token(text, maxlen, nb_word):
        input = [[nb_word] + s[0:-1] for s in text]
        label = text
        print("Pad sequences")
        X = sequence.pad_sequences(input, maxlen=maxlen, padding='pre', value=-1.0)
        label = sequence.pad_sequences(label, maxlen=maxlen, padding='pre', value=-1.0)
        X += 1
        label += 1
        print('input shape:', X.shape)
        print('output shape:', label.shape)
        return X, label

    @staticmethod
    def get_ngram(text, n, step, nb_word):
        sentences = []
        next_tokens = []
        for s in text:
            for i in range(-1 * n, len(s) - n, step):
                if i < 0:
                    sentences.append([nb_word]*(np.abs(i)) + s[0: i+n])
                    next_tokens.append(s[i + n])
                else:
                    sentences.append(s[i: i + n])
                    next_tokens.append(s[i + n])
        print('nb sequences:', len(sentences))
        X = sequence.pad_sequences(sentences, maxlen=n, padding='pre', value=0.0)
        y = np.array(next_tokens)
        return (X, y)


    @staticmethod
    def ngram_prediction(train, test, n, nb_word):
        (X_train, label_train) = CorpusFactory.get_ngram(train, n, 1, nb_word)
        (X_test, label_test) = CorpusFactory.get_ngram(test, n, 1, nb_word)
        return X_train, label_train, X_test, label_test

