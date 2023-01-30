import torch
from torch import nn
import torch.nn.functional as F
from hyperparameters import PITCH_NUM

HIDDEN_DIM = 128
class TransductionRNN(nn.Module):
    def __init__(self, input_dim = PITCH_NUM+1, hidden_dim = HIDDEN_DIM, target_dim = PITCH_NUM):
        super(TransductionRNN, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, target_dim, bias=True)
        self.out_activate = nn.Sigmoid()

    def forward(self, input):
        """
        Args:
          input: (batch_size, len_data, input_dim)

        Outputs:
          output: (batch_size, len_data, pitch_num)
        """
       
        (x, _) = self.gru(input)
        x = self.fc(x)
        return self.out_activate(x)
    
class TransductionFC(nn.Module):
    def __init__(self, input_dim = PITCH_NUM+1, hidden_dim = HIDDEN_DIM, target_dim = PITCH_NUM):
        super(TransductionFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )
        self.out_activate = nn.Sigmoid()

    def forward(self, input):
        """
        Args:
          input: (batch_size, len_data, input_dim)

        Outputs:
          output: (batch_size, len_data, pitch_num)
        """
       
        x = self.layers(input)
        return self.out_activate(x)