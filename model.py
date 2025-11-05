import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder for sequential input.
    Reduces bidirectional output by summing forward and backward hidden states.
    """
    def __init__(self, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, src, hidden=None):
        outputs, hidden = self.lstm(src, hidden)
        # Combine bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attention(nn.Module):
    """
    Additive attention mechanism to compute attention weights over encoder outputs.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class Decoder(nn.Module):
    """
    Decoder with attention mechanism and token embedding.
    Predicts per-step output over 2 classes.
    """
    def __init__(self, embed_size, hidden_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(3, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)

        self.lstm = nn.LSTM(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout
        )

        self.out = nn.Linear(hidden_size * 2, 2)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embedding(input).unsqueeze(0)
        embedded = self.dropout(embedded)

        hidden_only = last_hidden[0]
        attn_weights = self.attention(hidden_only[-1], encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.lstm(rnn_input, last_hidden)

        output = output.squeeze(0)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], dim=1))

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model with encoder, decoder, and attention.
    Outputs:
      - burst_res: per-step predictions (BCE-style)
      - flow_res: sentence-level/global prediction (via adaptive pooling)
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = 2

        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_outputs, hidden = self.encoder(src)
        hidden_state, cell_state = hidden

        # Truncate hidden states if encoder is bidirectional
        hidden_state = hidden_state[:self.decoder.n_layers]
        cell_state = cell_state[:self.decoder.n_layers]
        decoder_hidden = (hidden_state, cell_state)

        start_token = torch.ones(batch_size).long().cuda() * 2
        output = Variable(start_token)

        for t in range(max_len):
            output, decoder_hidden, attn_weights = self.decoder(output, decoder_hidden, encoder_outputs)
            outputs[t] = output

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if use_teacher_forcing else top1).cuda()

        outputs_softmax = self.softmax(outputs.permute(1, 0, 2))
        burst_res = outputs_softmax[:, :, -1]
        flow_res = self.gmp(burst_res).squeeze(-1)

        return burst_res, flow_res
