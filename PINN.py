from torch import nn


class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()

        self.tanh = nn.Tanh()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l1.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l1.weight)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l2.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l2.weight)

        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l3.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l3.weight)

        self.l4 = nn.Linear(hidden_size, output_size)
        self.l4.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l4.weight)

    def forward(self, x):
        out = self.l1(x)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        out = self.tanh(out)
        out = self.l4(out)
        return out
