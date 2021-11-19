import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, 
                dense_embedding=False, dense_embedding_n=650, dense_embedding_type='default', dense_embedding_initrange=0.1):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)

        self.encoder_weight = None
        if dense_embedding:
            if not isinstance(dense_embedding_n, list) and not isinstance(dense_embedding_n, tuple):
                dense_embedding_n = [dense_embedding_n] 
            print('using %s dense embedding layer with size %d' % (dense_embedding_type, dense_embedding_n[0]))
            self.dense_embedding_n = dense_embedding_n
            self.dense_embedding_initrange = dense_embedding_initrange
            if dense_embedding_type == 'random_sphere':  
                print('generating +1, -1 sphere')  
                # self.encoder_weight = (torch.randint(0, 2, (ntoken, dense_embedding_n)) * 2 - 1) / math.sqrt(dense_embedding_n)
                # self.encoder_weight = (torch.randint(0, 2, (ntoken, dense_embedding_n)) * 2 - 1.0) * 0.1
                self.encoder_weight = (torch.randint(0, 2, (ntoken, dense_embedding_n[0])) * 2 - 1) * dense_embedding_initrange
            elif dense_embedding_type == 'normal':
                print('generating normal embedding')
                self.encoder_weight = torch.empty(ntoken, dense_embedding_n[0])
                nn.init.normal_(self.encoder_weight, std=dense_embedding_initrange)
            elif dense_embedding_type == 'uniform':
                # self.encoder_weight = (torch.rand(ntoken, dense_embedding_n) * 2 - 1) * 0.1
                # self.encoder_weight = torch.empty(ntoken, dense_embedding_n)
                self.encoder_weight = torch.empty(ntoken, dense_embedding_n[0])
                nn.init.uniform_(self.encoder_weight, -dense_embedding_initrange, dense_embedding_initrange)    
                # self.encoder_weight = self.encoder_weight / (torch.sqrt(torch.sum(torch.square(self.encoder_weight), 1)).unsqueeze(-1)) 
            # rescaled with initrange
            # self.encoder_weight = self.encoder_weight * 0.1
            self.encoder_weight.requires_grad = False
            # self.encoder = nn.Embedding(ntoken, dense_embedding_n, _weight=self.encoder_weight)
            self.encoder = nn.Embedding(ntoken, dense_embedding_n[0], _weight=self.encoder_weight)
        else:
            self.encoder = nn.Embedding(ntoken, ninp, _weight=self.encoder_weight)

        # # debug
        # self.encoder.weight.requires_grad = False
        print(self.encoder.weight.shape)
        # print(self.encoder.weight.requires_grad)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        # experimental dense embedding
        self.dense_embedding = dense_embedding
        if dense_embedding:
            self.encoder.weight.requires_grad = False
            # self.encoder_linear = nn.Linear(dense_embedding_n, ninp, bias=False)
            self.encoder_linears = nn.ModuleList()
            for i in range(len(dense_embedding_n)-1):
                self.encoder_linears.append(nn.Linear(dense_embedding_n[i], dense_embedding_n[i+1], bias=True))
            self.encoder_linears.append(nn.Linear(dense_embedding_n[-1], ninp, bias=True))
        # debug
        # print(self.encoder.weight.requires_grad)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        if not self.dense_embedding:
            nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        else:
            # nn.init.uniform_(self.encoder_linear.weight, -initrange, initrange)
            # nn.init.zeros_(self.encoder_linear.weight)
            # nn.init.eye_(self.encoder_linear.weight)
            for i in range(len(self.dense_embedding_n)):
                nn.init.xavier_uniform_(self.encoder_linears[i].weight)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        encoder_output = self.encoder(input)
        if self.dense_embedding:
            for i in range(len(self.dense_embedding_n) - 1):
                encoder_output = self.encoder_linears[i](encoder_output)
                encoder_output = F.relu(encoder_output)
            encoder_output = self.encoder_linears[-1](encoder_output)
        emb = self.drop(encoder_output)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
