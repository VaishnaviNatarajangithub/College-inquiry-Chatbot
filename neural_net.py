import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Reshape input to [batch_size, seq_len, input_size]
        x = x.view(x.size(0), 1, -1)  # Adding seq_len dimension
        out, (hn, cn) = self.lstm(x)  # LSTM layer
        out = self.fc(hn[-1])  # Fully connected layer
        return self.softmax(out)  # Softmax output
