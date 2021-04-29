import os
import pickle
import hydra
import torch
import pandas as pd
from data import pad_data
from omegaconf import DictConfig
from vocabulary import Vocabulary
from batched_iterator import BatchedIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_processed_data(file, vocab):
    train_df = pd.read_table(file, header=None, names=['source'])

    train_df = train_df.iloc[:3000]
    train_df['src_encoded'] = train_df.apply(lambda x: vocab.encode(x.source), axis=1)

    X_train = train_df.src_encoded.to_numpy()

    return X_train


@hydra.main(config_name="inference")
def main(cfg: DictConfig):
    model_file = os.path.join(cfg.exp_dir, 'model.pt')
    model = torch.load(model_file, map_location=torch.device(device)).to(device)

    model.eval()

    vocab_file = os.path.join(cfg.exp_dir, 'vocab.pkl')
    vocab_dec_file = os.path.join(cfg.exp_dir, 'vocab_dec.pkl')
    with open(vocab_file, 'rb') as file:
        vocab_enc = pickle.load(file)
    with open(vocab_dec_file, 'rb') as file:
        vocab_dec = pickle.load(file)

    vocab = Vocabulary(vocab=vocab_enc, vocab_dec=vocab_dec)

    if cfg.use_file:
        source = get_processed_data(cfg.file, vocab)
        predicted = []
        test_iter = BatchedIterator(source, batch_size=128)

        for bi, src in enumerate(test_iter.iterate_once()):
            src_padded = pad_data(src[0], vocab_enc['<PAD>']).to(device)

            outputs = model(src_padded)

            outputs_pred = outputs.argmax(-1)

            for output in outputs_pred:
                predicted.append(vocab.decode_output(output.tolist()))

        pred_file = os.path.join(cfg.exp_dir, f'inference/{cfg.lang}_predicted.txt')
        os.makedirs(os.path.dirname(pred_file), exist_ok=True)

        with open(pred_file, 'w') as file:
            file.write('\n'.join(predicted))
    else:
        sentence = input("Sentence: ")
        while sentence != "exit":
            sentence = sentence.lower()
            encoded = vocab.encode(sentence)
            encoded = torch.tensor(encoded)
            encoded = torch.unsqueeze(encoded, 0).to(device)
            output = model(encoded)
            output = output.argmax(-1).to('cpu').tolist()
            decoded = vocab.decode_output(output[0])
            print(f"Restored diacritics version: {decoded}")
            sentence = input("Sentence: ")


if __name__ == '__main__':
    main()
