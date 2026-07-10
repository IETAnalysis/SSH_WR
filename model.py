import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size, n_layers=1, dropout=0.5, cell_type='lstm'):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.cell_type = cell_type.lower()

        recurrent_cls = nn.LSTM if self.cell_type == 'lstm' else nn.GRU
        self.recurrent = recurrent_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(self, src, hidden=None):
        outputs, hidden = self.recurrent(src, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.empty(hidden_size))
        stdv = 1.0 / math.sqrt(hidden_size)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        steps = encoder_outputs.size(0)
        hidden = hidden.repeat(steps, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        energy = self.score(hidden, encoder_outputs)
        return F.softmax(energy, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers=1, dropout=0.2, cell_type='lstm'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cell_type = cell_type.lower()

        self.embedding = nn.Embedding(
            config.MODEL_CONFIG['decoder_num_embeddings'],
            embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_size)

        recurrent_cls = nn.LSTM if self.cell_type == 'lstm' else nn.GRU
        self.recurrent = recurrent_cls(
            input_size=embedding_dim + hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size * 2, 2)

    def forward(self, input_token, last_hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input_token).unsqueeze(0))
        hidden_state = last_hidden[0] if isinstance(last_hidden, tuple) else last_hidden
        attn_weights = self.attention(hidden_state[-1], encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        recurrent_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.recurrent(recurrent_input, last_hidden)

        output = output.squeeze(0)
        context = context.squeeze(0)
        logits = self.output_layer(torch.cat([output, context], dim=1))
        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.class_probability = nn.Softmax(dim=-1)

    def _trim_bidirectional_hidden(self, hidden):
        if isinstance(hidden, tuple):
            hidden_state, cell_state = hidden
            return hidden_state[:self.decoder.n_layers], cell_state[:self.decoder.n_layers]
        return hidden[:self.decoder.n_layers]

    def forward(self, src, target_burst_labels, teacher_forcing_ratio):
        device = src.device
        batch_size = src.size(1)
        sequence_length = target_burst_labels.size(0)
        num_classes = 2

        logits = torch.zeros(sequence_length, batch_size, num_classes, device=device)
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden = self._trim_bidirectional_hidden(encoder_hidden)

        input_token = torch.full(
            (batch_size,),
            config.MODEL_CONFIG['decoder_start_token_id'],
            dtype=torch.long,
            device=device,
        )

        for step in range(sequence_length):
            step_logits, decoder_hidden, _ = self.decoder(input_token, decoder_hidden, encoder_outputs)
            logits[step] = step_logits

            teacher_force = random.random() < teacher_forcing_ratio
            predicted_token = step_logits.argmax(dim=1)
            input_token = target_burst_labels[step] if teacher_force else predicted_token

        burst_probabilities = self.class_probability(logits.permute(1, 0, 2))[:, :, -1]
        flow_probabilities = self.global_max_pool(burst_probabilities).squeeze(-1)
        return burst_probabilities, flow_probabilities
