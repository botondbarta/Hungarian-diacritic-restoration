import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, vocab_size, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.ReLu = nn.ReLU()
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
        self.softmax = nn.Softmax(vocab_size)

    def forward(self, sentence):
        x = self.dropout(self.embeddings(sentence))
        outputs, (hidden, cell) = self.rnn(x)

        # outputs = self.ReLu(outputs)
        outputs = self.output(outputs)
        return outputs
        # return self.softmax(outputs)
