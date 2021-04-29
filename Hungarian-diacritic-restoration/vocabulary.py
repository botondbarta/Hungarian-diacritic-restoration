import os
import pickle


class Vocabulary:
    def __init__(self, **kwargs):
        if 'df' in kwargs:
            df = kwargs['df']
            self.chars = self.create_vocab(df)
            self.char_indices = dict((c, i + 1) for i, c in enumerate(self.chars))
            self.char_indices['<PAD>'] = 0
            with open(os.getcwd() + '/vocab.pkl', 'wb') as file:
                pickle.dump(self.char_indices, file)
            self.indices_char = dict((i + 1, c) for i, c in enumerate(self.chars))
            self.indices_char[0] = '<PAD>'
            with open(os.getcwd() + '/vocab_dec.pkl', 'wb') as file:
                pickle.dump(self.indices_char, file)
        elif 'vocab' in kwargs and 'vocab_dec' in kwargs:
            self.char_indices = kwargs['vocab']
            self.indices_char = kwargs['vocab_dec']
        self.vowels = [self.char_indices[v] for v in 'aeiou']

    def create_vocab(self, df):
        vocab = set()
        # vocab.add('<SOS>')
        vocab.add('<EOS>')
        vocab.add('<UNK>')

        for token in df.source:
            vocab |= set(token)

        for token in df.target:
            vocab |= set(token)

        return vocab

    def encode(self, source):
        ids = []
        for c in source:
            ids.append(self.char_indices.get(c, self.char_indices['<UNK>']))
        ids.append(self.char_indices['<EOS>'])
        return ids

    def decode_output(self, output):
        chars = []
        for y in output:
            if y == self.char_indices['<EOS>']:
                break
            chars.append(self.indices_char.get(y, '<UNK>'))

        return ''.join(chars)

