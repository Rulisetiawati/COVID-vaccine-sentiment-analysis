import torch
import torch.nn as nn

class SeqModel(nn.Module):
    def __init__(
        self,
        embedding_size=200,
        vocab_size=300,
        output_size=5,
        hidden_size=200,
        num_layers=2,
        nonlin="tanh",
        dropout_rate=0.7,
        mode=0,
        unit="lstm",
        more_features=False,
    ):
        # add glove param here? ^^

        super(SeqModel, self).__init__()
        self.mode = mode
        self.unit = unit
        self.more_features = more_features

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_size
        )
        self.embedding.weight.data.normal_(0.0, 0.05)  # mean=0.0, mu=0.05

        if mode == 0:
            if unit == "lstm":
                self.lstm_rnn = nn.LSTM(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
            elif unit == "gru":
                self.gru_rnn = nn.GRU(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
            else:
                # baseline: unidirectional rnn
                self.rnn = nn.RNN(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    nonlinearity=nonlin,
                )
            if more_features:
                self.linear_layer = nn.Linear(
                    hidden_size + len(numeric_features), output_size
                )
            else:
                self.linear_layer = nn.Linear(hidden_size, output_size)

        # model with dropout:
        if mode == 1:
            if unit == "lstm":
                self.lstm_rnn = nn.LSTM(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout_rate,
                )
            elif unit == "gru":
                self.gru_rnn = nn.GRU(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout_rate,
                )
            else:
                self.rnn = nn.RNN(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    nonlinearity=nonlin,
                    dropout=dropout_rate,
                )

            if more_features:
                self.linear_layer = nn.Linear(
                    hidden_size + len(numeric_features), output_size
                )
            else:
                self.linear_layer = nn.Linear(hidden_size, output_size)

        # Bidirectional model
        if mode == 2:
            if unit == "lstm":
                self.lstm_rnn = nn.LSTM(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout_rate,
                    bidirectional=True,
                )
            elif unit == "gru":
                self.gru_rnn = nn.GRU(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout_rate,
                    bidirectional=True,
                )
            else:
                self.rnn = nn.RNN(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    nonlinearity=nonlin,
                    dropout=dropout_rate,
                    bidirectional=True,
                )

            if more_features:
                self.linear_layer = nn.Linear(
                    hidden_size * 2 + len(numeric_features), output_size
                )
            else:
                self.linear_layer = nn.Linear(hidden_size * 2, output_size)

        self.activation_fn = nonlin
        self.softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, x, x_concat=None):
        # permute x?
        out = self.embedding(x)
        if self.unit == "lstm":
            out, (h_state, c_state) = self.lstm_rnn(out)
        elif self.unit == "gru":
            out, h_state = self.gru_rnn(out)
        else:
            out, h_state = self.rnn(out)
        out = out[-1]
        if self.more_features:
            out = torch.cat((out, x_concat.permute(1, 0)), dim=1)
        out = self.linear_layer(out)
        out = self.softmax_layer(out)
        return out
