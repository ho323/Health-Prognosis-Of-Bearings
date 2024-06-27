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
    def __init__(self, channels=2):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.downsampling = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.upsampling = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.ConvTranspose1d(64, channels, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm1d(channels),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.upsampling(x)
        x = self.output_layer(x)
        x = x.permute(0,2,1)
        return x