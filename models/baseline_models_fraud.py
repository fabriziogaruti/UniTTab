from loguru import logger
import torch
from torch import nn


def build_baseline_model(model_type, use_embeddings, hidden_size, sequence_len, num_layers, input_size, field_input_size, vocab_size, equal_parameters_baselines=False):

    logger.debug('BUILDING BASELINE MODEL WITH:')
    logger.debug(f"use_embeddings = {use_embeddings}")
    logger.debug(f"hidden_size = {hidden_size}")
    logger.debug(f"sequence_len = {sequence_len}")
    logger.debug(f"num_layers = {num_layers}")
    logger.debug(f"input_size = {input_size}")
    logger.debug(f"field_input_size = {field_input_size}")
    logger.debug(f"vocab_size = {vocab_size}")
    logger.debug(f"equal_parameters_baselines = {equal_parameters_baselines}")

    if model_type=='lstm':
        if equal_parameters_baselines:
            model = LSTMModelEqual(use_embeddings, sequence_len, num_layers, hidden_size, input_size, field_input_size, vocab_size)
        else:
            model = LSTMModel(use_embeddings, sequence_len, num_layers, hidden_size, input_size, field_input_size, vocab_size)
    # elif model_type=='mlp':
    #     model = MLPModel(use_embeddings, sequence_len, num_layers, hidden_size, input_size, equal_parameters_baselines=equal_parameters_baselines)
    else:
        raise NotImplementedError()

    return model


class LSTMModelEqual(nn.Module):

    def __init__(self, use_embeddings, sequence_len, num_layers, hidden_size, input_size, field_input_size, vocab_size):
        super().__init__()

        self.use_embeddings = use_embeddings
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.field_input_size = field_input_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        if not self.use_embeddings:
            # input shape (batch, seq_len*n_col)
            self.embedding_layer = nn.Embedding(self.vocab_size, self.field_input_size)
            self.embedding_layer.requires_grad = False
            self.has_embedding_layer=True
        else:
            self.has_embedding_layer=False

        self.linear = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.hidden_size, num_layers=self.num_layers, hidden_size=self.hidden_size, dropout=0.1,
                            batch_first=True)
        self.head = nn.Linear(self.hidden_size, 1)

        self.loss_fct = nn.BCEWithLogitsLoss()
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def init_embedding_layer(self, weights):
        assert not self.use_embeddings
        self.embedding_layer.weight.data = weights
        self.embedding_layer.weight.requires_grad = False

    def forward(self, input_ids, labels=None):
        if self.has_embedding_layer:
            with torch.no_grad():
                input_ids = self.embedding_layer(input_ids)

        expected_sz = [input_ids.shape[0], self.sequence_len, -1]
        input_ids = input_ids.view(expected_sz)
        embeddings = self.linear(input_ids)
        embeddings, _ = self.lstm(embeddings)
        last_embedding = embeddings[:,-1,:]

        output = self.head(last_embedding).squeeze(dim=1)

        if labels is not None and input_ids.shape[0]>0:
            aggregated_labels = torch.any(labels, dim=1).to(torch.float32)
            loss = self.loss_fct(output, aggregated_labels)
            return (loss, output)
        return (output,)


class LSTMModel(nn.Module):

    def __init__(self, use_embeddings, sequence_len, num_layers, hidden_size, input_size, field_input_size, vocab_size):
        super().__init__()

        self.use_embeddings = use_embeddings
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_input_size = field_input_size
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        if not self.use_embeddings:
            # input shape (batch, seq_len*n_col)
            self.embedding_layer = nn.Embedding(self.vocab_size, self.field_input_size)
            self.has_embedding_layer=True
        else:
            self.has_embedding_layer=False

        self.linear = nn.Linear(self.input_size, self.hidden_size)

        self.lstm = nn.LSTM(input_size=self.hidden_size, num_layers=self.num_layers, hidden_size=self.hidden_size, dropout=0.1,
                            batch_first=True)
        self.head = nn.Linear(self.hidden_size, 1)

        self.loss_fct = nn.BCEWithLogitsLoss()
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.use_embeddings:
            self.embedding_layer.bias.data.zero_()
            self.embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, labels=None):
        if self.has_embedding_layer:
            input_ids = self.embedding_layer(input_ids)
        expected_sz = [input_ids.shape[0], self.sequence_len, -1]
        input_ids = input_ids.view(expected_sz)
        embeddings = self.linear(input_ids)

        embeddings, _ = self.lstm(embeddings)
        last_embedding = embeddings[:,-1,:]

        output = self.head(last_embedding).squeeze(dim=1)

        if labels is not None and input_ids.shape[0]>0:
            aggregated_labels = torch.any(labels, dim=1).to(torch.float32)
            loss = self.loss_fct(output, aggregated_labels)
            return (loss, output)
        return (output,)
