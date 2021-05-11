import os
import sys

import torch
import torch.nn as nn

from models.baseRNN import BaseRNN

class EncoderRNN(BaseRNN):

    def __init__(self, vocab_size, max_len, hidden_size,
                 embedding_size, input_dropout_p=0, dropout_p=0, pos_embedding_size=None, pos_embedding=None,
                 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=False,
                 embedding=None, update_embedding=True):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.pos_embedding = pos_embedding
        if self.pos_embedding != None:
            self.rnn = self.rnn_cell(embedding_size+pos_embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_partition, input_lengths=None):
        batch_size = input_var.size(0)
        seq_len = input_var.size(1)

        if self.pos_embedding == None:
            embedded = self.embedding(input_var)
        else:
            embedded = torch.cat((self.embedding(input_var), self.pos_embedding(input_partition)), dim=2)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden
