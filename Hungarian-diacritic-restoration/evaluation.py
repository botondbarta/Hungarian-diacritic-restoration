import hydra
from omegaconf import DictConfig
import pandas as pd
import os
import pickle
import torch
import numpy as np
from vocabulary import Vocabulary
from data import remove_diacritics, pad_data
from batched_iterator import BatchedIterator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_name="evaluation")
def main(cfg: DictConfig):
    model_file = os.path.join(cfg.model_dir, 'model.pt')
    model = torch.load(model_file, map_location=torch.device(device)).to(device)
    model.eval()

    vocab_file = os.path.join(cfg.model_dir, 'vocab.pkl')
    vocab_dec_file = os.path.join(cfg.model_dir, 'vocab_dec.pkl')
    with open(vocab_file, 'rb') as file:
        vocab_enc = pickle.load(file)
    with open(vocab_dec_file, 'rb') as file:
        vocab_dec = pickle.load(file)
    vocab = Vocabulary(vocab=vocab_enc, vocab_dec=vocab_dec)

    eval_df = pd.read_table(cfg.dev_file, header=None, names=['target'])
    eval_df = eval_df.iloc[100:102]
    eval_df['source'] = eval_df.apply(lambda x: remove_diacritics(x.target), axis=1)
    eval_df['src_encoded'] = eval_df.apply(lambda x: vocab.encode(x.source), axis=1)

    target = eval_df.target.to_numpy(dtype=str)

    target_words = np.hstack(np.char.split(target, sep=' '))
    target_words = np.array(list(filter(lambda x: len(x) > 1, target_words)))

    print(eval_df.iloc[0].source)
    print(eval_df.iloc[1].source)

    X_dev = eval_df.src_encoded.to_numpy()

    predicted = []
    test_iter = BatchedIterator(X_dev, batch_size=10)

    for bi, src in enumerate(test_iter.iterate_once()):
        src_padded = pad_data(src[0], vocab_enc['<PAD>']).to(device)

        outputs = model(src_padded)
        print(outputs.shape)
        outputs_pred = outputs.argmax(-1)

        for output in outputs_pred:
            decodec_sentence = vocab.decode_output(output.tolist())
            print(decodec_sentence)
            predicted.append(decodec_sentence)


    predicted = np.hstack(np.char.split(predicted, sep=' '))
    predicted = np.array(list(filter(lambda x: len(x) > 1, predicted)))

    print(predicted.shape)
    print(target_words.shape)
    correct = (target_words == predicted).sum()
    accuracy = correct / len(predicted)
    print(accuracy)


if __name__ == '__main__':
    main()
