import torch
import os
import yaml
import logging
import numpy as np
from torch import optim, nn
from datetime import datetime
from vocabulary import Vocabulary
from model import Model
from data import get_data, get_df, pad_data, get_mask
from batched_iterator import BatchedIterator


class Experiment:
    def __init__(self, cfg):
        self.result = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __enter__(self):
        self.result["start_time"] = datetime.now()
        self.result["running_time"] = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result["running_time"] = (datetime.now() - self.result["start_time"]).total_seconds()
        result_file = os.path.join(os.getcwd(), "result.yaml")
        with open(result_file, 'w+') as file:
            yaml.dump(self.result, file)

    def train(self, model, X_train, y_train, X_dev, y_dev, config):
        log = logging.getLogger('train')
        optimizer = optim.Adam(model.parameters(), lr=config.lrate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        train_iter = BatchedIterator(X_train, y_train, batch_size=config.batch_size)

        dev_iter = BatchedIterator(X_dev, y_dev, batch_size=config.batch_size)

        all_dev_loss = []
        all_dev_acc = []
        all_train_loss = []

        patience = config.patience
        epochs_no_improve = 0
        min_loss = np.Inf
        early_stopping = False
        best_epoch = 0

        for epoch in range(config.epoch):
            model.train()
            train_loss = 0
            i = 0
            for batch_x, batch_y in train_iter.iterate_once():
                batch_x = pad_data(batch_x, 0).to(self.device)
                batch_y = pad_data(batch_y, 0).to(self.device)

                output = model(batch_x)

                output = output.reshape(-1, output.shape[2])
                batch_y = batch_y.reshape(-1)

                optimizer.zero_grad()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss
                i += 1

            model.eval()
            with torch.no_grad():
                dev_acc, dev_loss = self.evaluate_model(model, dev_iter, criterion)

                all_train_loss.append(train_loss)
                all_dev_loss.append(dev_loss)
                all_dev_acc.append(dev_acc)

                log.info(f"Epoch: {epoch}")
                log.info(f"  train loss: {train_loss}")
                log.info(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")
            torch.save(model, os.path.join(os.getcwd(), "model_latest.pt"))
            if min_loss - dev_loss > 0.001:
                epochs_no_improve = 0
                min_loss = dev_loss
                best_epoch = epoch
                torch.save(model, os.path.join(os.getcwd(), "model.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    early_stopping = True
                    print("Early stopping")
            if early_stopping:
                break

        return all_train_loss[best_epoch], all_dev_acc[best_epoch], all_dev_loss[best_epoch]

    def evaluate_model(self, model, iterator, criterion):
        loss = 0
        correct_guesses = 0
        all_guesses = 0
        bi = 0
        for bi, (batch_x, batch_y) in enumerate(iterator.iterate_once()):
            batch_x = pad_data(batch_x, 0).to(self.device)
            batch_y = pad_data(batch_y, 0).to(self.device)
            output = model(batch_x)

            output = output.reshape(-1, output.shape[2])
            loss += criterion(output, batch_y.reshape(-1))

            label = batch_y.reshape(-1)
            output_pred = output.argmax(-1)
            eq = torch.eq(output_pred, label)
            mask_x = get_mask(batch_x, self.vocab.vowels)
            eq[mask_x] = 0
            correct_guesses += eq.sum().float()
            all_guesses += torch.sum(mask_x == False)
        loss /= (bi + 1)
        acc = correct_guesses / all_guesses
        return acc, loss

    def run(self, cfg):
        train_df, dev_df = get_df(cfg.data.train_file, cfg.data.dev_file)

        self.vocab = Vocabulary(df=train_df)

        X_train, y_train, X_dev, y_dev = get_data(train_df, dev_df, self.vocab)

        embedding_dim = cfg.model.emb_size
        hidden_dim = cfg.model.hidden_size
        n_layers = cfg.model.num_layers
        vocab_size = len(self.vocab.chars)

        model = Model(embedding_dim, hidden_dim, n_layers, vocab_size).to(self.device)

        train_loss, dev_acc, dev_loss = self.train(model, X_train, y_train, X_dev, y_dev, cfg.model)

        self.result["train_loss"] = float(train_loss)
        self.result["dev_acc"] = float(dev_acc)
        self.result["dev_loss"] = float(dev_loss)

