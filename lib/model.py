import torch
import torch.nn as nn

class ResConv1dBlock(nn.Module):
    def __init__(self, channels):
        super(ResConv1dBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm1d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = out + x
        out = self.relu(out)
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length, device):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, device=device)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out)
        return out

class Regressor(nn.Module):
    def __init__(self, input_size, device):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16, device=device)
        self.fc3 = nn.Linear(16, 1, device=device)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.fc1(x[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class DegradationModel(nn.Module):
    def __init__(self, input_channels=2, hidden_size=64, num_layers=4, seq_length=128, device=torch.device("cuda")):
        super(DegradationModel, self).__init__()
        self.conv = ResConv1dBlock(input_channels)
        self.bilstm = BiLSTM(input_channels, hidden_size, num_layers, seq_length, device)
        self.reg = Regressor(hidden_size*2, device=device)
        self.device = device

    def forward(self, x):
        x = self.conv(x.permute(0,2,1))
        x = self.bilstm(x.permute(0,2,1))
        out = self.reg(x)
        return out
    

class Generator(nn.Module):
    def __init__(self, input_channel, hidden_channel, num_layers=4):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_channel, hidden_channel, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_channel, input_channel)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channel, hidden_channel, num_layers=4):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_channel, hidden_channel, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_channel, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x