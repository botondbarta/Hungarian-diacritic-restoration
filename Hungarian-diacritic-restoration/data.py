import pandas as pd
import torch


def remove_diacritics(sentence):
    return sentence.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ö', 'o') \
        .replace('ő', 'o').replace('ú', 'u').replace('ü', 'u')


def get_df(train_file, dev_file):
    train_df = pd.read_table(train_file, header=None, names=['target'])
    dev_df = pd.read_table(dev_file, header=None, names=['target'])

    train_df = train_df.iloc[:3000]
    dev_df = dev_df.iloc[:500]

    train_df['source'] = train_df.apply(lambda x: remove_diacritics(x.target), axis=1)
    dev_df['source'] = dev_df.apply(lambda x: remove_diacritics(x.target), axis=1)

    return train_df, dev_df


def get_data(train_df, dev_df, vocab):
    train_df['tgt_encoded'] = train_df.apply(lambda x: vocab.encode(x.target), axis=1)
    train_df['src_encoded'] = train_df.apply(lambda x: vocab.encode(x.source), axis=1)

    dev_df['tgt_encoded'] = dev_df.apply(lambda x: vocab.encode(x.target), axis=1)
    dev_df['src_encoded'] = dev_df.apply(lambda x: vocab.encode(x.source), axis=1)

    X_train = train_df.src_encoded.to_numpy()
    y_train = train_df.tgt_encoded.to_numpy()

    X_dev = dev_df.src_encoded.to_numpy()
    y_dev = dev_df.tgt_encoded.to_numpy()

    return X_train, y_train, X_dev, y_dev


def pad_data(batch, pad):
    seq_len = list(map(len, batch))
    length = max(seq_len)
    data = torch.tensor([xi + [pad] * (length - len(xi)) for xi in batch])
    return data


def get_mask(batch, vowels):
    return ~sum(batch == vowel for vowel in vowels).bool()
