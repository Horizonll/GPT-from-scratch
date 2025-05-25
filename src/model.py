import math
import copy
import torch
import torch.nn as nn


class LMModel_RNN(nn.Module):
    """
    RNN-based language model:
    1) Embedding layer
    2) Vanilla RNN network (no nn.RNN, manual implementation)
    3) Output linear layer
    """

    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=4, dropout=0.5):
        super(LMModel_RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        self.hidden_size = hidden_size
        self.input_size = dim
        self.num_layers = num_layers
        self.W_ih = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        hidden_size, self.input_size if l == 0 else hidden_size
                    )
                )
                for l in range(num_layers)
            ]
        )
        self.W_hh = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                for l in range(num_layers)
            ]
        )
        self.b_ih = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.b_hh = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        for l in range(self.num_layers):
            nn.init.uniform_(self.W_ih[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.W_hh[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_ih[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_hh[l], -init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        embeddings = self.drop(self.encoder(input))
        seq_len, batch_size, _ = embeddings.size()
        if hidden is None:
            hidden = [
                embeddings.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        else:
            hidden = [h for h in hidden]
        outputs = []
        for t in range(seq_len):
            x = embeddings[t]
            h_t = []
            for l in range(self.num_layers):
                h_prev = hidden[l]
                W_ih = self.W_ih[l]
                W_hh = self.W_hh[l]
                b_ih = self.b_ih[l]
                b_hh = self.b_hh[l]
                h = torch.tanh(
                    torch.matmul(x, W_ih.t())
                    + b_ih
                    + torch.matmul(h_prev, W_hh.t())
                    + b_hh
                )
                x = h
                h_t.append(h)
            hidden = h_t
            outputs.append(h_t[-1].unsqueeze(0))
        output = torch.cat(outputs, dim=0)
        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(1)

        # [seq_len, batch, d_model] -> [seq_len, batch, nhead, d_k] -> [nhead, seq_len, batch, d_k]
        q = self.W_q(query).view(-1, batch_size, self.nhead, self.d_k).transpose(0, 2)
        k = self.W_k(key).view(-1, batch_size, self.nhead, self.d_k).transpose(0, 2)
        v = self.W_v(value).view(-1, batch_size, self.nhead, self.d_k).transpose(0, 2)

        # [nhead, seq_len_q, batch, d_k] × [nhead, seq_len_k, batch, d_k] -> [nhead, seq_len_q, batch, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == float("-inf"), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [nhead, seq_len_q, batch, seq_len_k] × [nhead, seq_len_v, batch, d_k] -> [nhead, seq_len_q, batch, d_k]
        context = torch.matmul(attn_weights, v)

        # [nhead, seq_len, batch, d_k] -> [seq_len, batch, nhead, d_k] -> [seq_len, batch, d_model]
        context = (
            context.transpose(0, 2).contiguous().view(-1, batch_size, self.d_model)
        )

        output = self.W_o(context)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [seq_len, batch, d_model] -> [seq_len, batch, d_ff] -> [seq_len, batch, d_model]
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = LayerNorm(encoder_layer.feed_forward.w_2.out_features)

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return self.norm(output)


class LMModel_transformer(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a transformer network and an output layer.
    # The word embedding layer has input as a sequence of word indices (in the vocabulary) and outputs a sequence of vectors where each one is a word embedding.
    # The transformer network has input of each word embedding and outputs a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and outputs the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, nhead=8, num_layers=4):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, dim)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        self.dim = dim
        self.pos_encoder = PositionalEncoding(dim, dropout=0.1)

        encoder_layer = TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=0.1
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        ########################################

        self.decoder = nn.Linear(dim, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        # print(input.device)
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        L = embeddings.size(0)
        src_mask = torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1).to(
            input.device
        )
        src = embeddings * math.sqrt(self.dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask)
        ########################################
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded.view(output.size(0), output.size(1), decoded.size(1))


class LMModel_LSTM(nn.Module):
    """
    LSTM-based language model:
    1) Embedding layer
    2) LSTM network (manual implementation, no nn.LSTM)
    3) Output linear layer
    """

    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=4, dropout=0.5):
        super(LMModel_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        # Manual LSTM implementation
        self.hidden_size = hidden_size
        self.input_size = dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.W_ii = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        hidden_size, self.input_size if l == 0 else hidden_size
                    )
                )
                for l in range(num_layers)
            ]
        )
        self.W_hi = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                for l in range(num_layers)
            ]
        )
        self.b_ii = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.b_hi = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )

        self.W_if = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        hidden_size, self.input_size if l == 0 else hidden_size
                    )
                )
                for l in range(num_layers)
            ]
        )
        self.W_hf = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                for l in range(num_layers)
            ]
        )
        self.b_if = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.b_hf = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )

        self.W_io = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        hidden_size, self.input_size if l == 0 else hidden_size
                    )
                )
                for l in range(num_layers)
            ]
        )
        self.W_ho = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                for l in range(num_layers)
            ]
        )
        self.b_io = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.b_ho = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )

        self.W_ig = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        hidden_size, self.input_size if l == 0 else hidden_size
                    )
                )
                for l in range(num_layers)
            ]
        )
        self.W_hg = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                for l in range(num_layers)
            ]
        )
        self.b_ig = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.b_hg = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )

        self.dropouts = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])
            if num_layers > 1
            else None
        )
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

        for l in range(self.num_layers):
            nn.init.uniform_(self.W_ii[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.W_hi[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_ii[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_hi[l], -init_uniform, init_uniform)

            nn.init.uniform_(self.W_if[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.W_hf[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_if[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_hf[l], -init_uniform, init_uniform)
            self.b_if[l].data.fill_(1.0)
            self.b_hf[l].data.fill_(1.0)

            nn.init.uniform_(self.W_io[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.W_ho[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_io[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_ho[l], -init_uniform, init_uniform)

            nn.init.uniform_(self.W_ig[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.W_hg[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_ig[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_hg[l], -init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)
        seq_len, batch_size, _ = embeddings.size()

        ########################################
        if hidden is None:
            h_0 = [
                embeddings.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
            c_0 = [
                embeddings.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
            hidden = (h_0, c_0)
        else:
            h_0, c_0 = hidden
            h_0 = [h for h in h_0]
            c_0 = [c for c in c_0]

        outputs = []
        h_n, c_n = [], []

        for t in range(seq_len):
            x = embeddings[t]  # (batch, dim)
            h_t, c_t = [], []

            for l in range(self.num_layers):
                h_prev = h_0[l]
                c_prev = c_0[l]

                i_t = torch.sigmoid(
                    torch.matmul(x, self.W_ii[l].t())
                    + self.b_ii[l]
                    + torch.matmul(h_prev, self.W_hi[l].t())
                    + self.b_hi[l]
                )

                f_t = torch.sigmoid(
                    torch.matmul(x, self.W_if[l].t())
                    + self.b_if[l]
                    + torch.matmul(h_prev, self.W_hf[l].t())
                    + self.b_hf[l]
                )

                o_t = torch.sigmoid(
                    torch.matmul(x, self.W_io[l].t())
                    + self.b_io[l]
                    + torch.matmul(h_prev, self.W_ho[l].t())
                    + self.b_ho[l]
                )

                g_t = torch.tanh(
                    torch.matmul(x, self.W_ig[l].t())
                    + self.b_ig[l]
                    + torch.matmul(h_prev, self.W_hg[l].t())
                    + self.b_hg[l]
                )

                c_next = f_t * c_prev + i_t * g_t

                h_next = o_t * torch.tanh(c_next)

                h_t.append(h_next)
                c_t.append(c_next)

                x = h_next

                if l < self.num_layers - 1 and self.dropouts is not None:
                    x = self.dropouts[l](x)

            h_0 = h_t
            c_0 = c_t

            outputs.append(h_t[-1].unsqueeze(0))

        output = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_size)

        for l in range(self.num_layers):
            h_n.append(h_0[l])
            c_n.append(c_0[l])

        hidden = (h_n, c_n)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden
