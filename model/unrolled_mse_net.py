import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(directory.parent)
from algorithm.multi_pitch_detection_fourier import get_cisoid_dict

from .model_element import init_layer, init_bn, init_gru, ConvBlock, ConvBlock2D
from hyperparameters import FIXED_LEN, PITCH_FREQUENCY_GRID, PITCH_FREQUENCY_GRID_FINER, PITCH_NUM, PROJECT_SAMPLE_RATE, TEMPLATE_PATH

_grid_under_use = PITCH_FREQUENCY_GRID_FINER

PITCH_REPEAT = 3
MID_FEAT_DIM = PITCH_NUM*PITCH_REPEAT
N_FRAME = len(_grid_under_use) # len(PITCH_FREQUENCY_GRID)
REDUCED_LEN = 37
HIDDEN_DIM = 128
class UnrolledMSENet(nn.Module):
    def __init__(self, fix_fourier = True, share_mse_weight = True, unrolled_iter = 5):
        super(UnrolledMSENet, self).__init__()
        
        # feature extractor
        self.fc_cos = nn.Linear(FIXED_LEN, N_FRAME, bias=False)
        self.fc_sin = nn.Linear(FIXED_LEN, N_FRAME, bias=False)
        
        # self.w_fourier = nn.Linear(N_FRAME, N_FRAME, bias=False)
        self.fix_fourier = fix_fourier
        self.fourier_init()
        
        # unrolled mse solver
        self.unrolled_iter = unrolled_iter
        self.template_matrix = torch.nn.Parameter(torch.zeros(N_FRAME, MID_FEAT_DIM), requires_grad=True)
        self.fc_out1 = nn.Linear(MID_FEAT_DIM, MID_FEAT_DIM, bias=True)
        self.fc_out2 = nn.Linear(MID_FEAT_DIM, PITCH_NUM, bias=True)
        self.fc_out = nn.Linear(PITCH_NUM, PITCH_NUM, bias=True)
        self.unrolled_matrix_init()
    
    def fourier_init(self):
        W_fourier_init = np.exp(
            -2*np.pi*1j*_grid_under_use.reshape(-1, 1)/PROJECT_SAMPLE_RATE @ np.arange(FIXED_LEN).reshape(1, -1)
        ).copy()
        W_cos_tensor = torch.tensor(np.real(W_fourier_init).astype(np.float32)) / FIXED_LEN
        W_sin_tensor = torch.tensor(np.imag(W_fourier_init).astype(np.float32)) / FIXED_LEN
        if self.fix_fourier:
            self.fc_cos.weight = torch.nn.Parameter(W_cos_tensor, requires_grad=False)
            self.fc_sin.weight = torch.nn.Parameter(W_sin_tensor, requires_grad=False)
        else:
            self.fc_cos.weight = torch.nn.Parameter(W_cos_tensor, requires_grad=True)
            self.fc_sin.weight = torch.nn.Parameter(W_sin_tensor, requires_grad=True)
        
    def unrolled_matrix_init(self):
        cisoid_dict = np.abs(get_cisoid_dict(TEMPLATE_PATH,  frequency_grid = _grid_under_use).cisoid_dict) ** 2 # (PITCH_NUM, N_FRAME)
        cisoid_dict = np.repeat(cisoid_dict, PITCH_REPEAT, axis=0) # (MID_FEAT_DIM, N_FRAME)
        
        # torch.nn.init.eye_(self.feat_transform_layers[0].weight)
        self.template_matrix = torch.nn.Parameter(
            torch.tensor(cisoid_dict.copy().astype(np.float32).T), requires_grad=True
        )
        
        self.fc_out1.weight = torch.nn.Parameter(
            torch.tensor(np.eye(MID_FEAT_DIM).astype(np.float32)), requires_grad=True
        )
        self.fc_out2.weight = torch.nn.Parameter(
            torch.tensor(2*np.repeat(np.eye(PITCH_NUM), PITCH_REPEAT, axis=0).astype(np.float32).T), requires_grad=True
        )
        self.fc_out.weight = torch.nn.Parameter(
            torch.tensor(4*np.eye(PITCH_NUM).astype(np.float32)), requires_grad=True
        )
        self.fc_out1.bias.data.fill_(0)
        self.fc_out2.bias.data.fill_(0)
        self.fc_out.bias.data.fill_(-3)
    
    def forward(self, input_this, input_prev):
        feat = self.get_feat(input_this)
        velocity_estimate = torch.sigmoid(self.fc_out(self.process_feat(feat)))
        return velocity_estimate
    
    def get_feat(self, input_this):
        if len(input_this.shape) == 2:
            N_valid_points = torch.count_nonzero(input_this, dim=1).unsqueeze(1)
        elif len(input_this.shape) == 1:
            N_valid_points = torch.count_nonzero(input_this, dim=0)
        else:
            assert 0, "input diminsion is invalid"
        N_valid_points = N_valid_points + 1
        x_cos = self.fc_cos(input_this)*(FIXED_LEN/N_valid_points)
        x_sin = self.fc_sin(input_this)*(FIXED_LEN/N_valid_points)
        feat = x_cos.pow(2) + x_sin.pow(2)
        return feat
    
    def process_feat(self, feat):
        xi_x = 2*feat
        running_feat = 1*feat
        for i_iter in range(self.unrolled_iter):
            if i_iter == 0:
                xi_x = running_feat * (-1)
                m_y = F.relu(-10*torch.matmul(xi_x, self.template_matrix))
            else:
                xi_x = 0.8*2*(torch.matmul(m_y, self.template_matrix.transpose(0,1)) - running_feat) + 0.2*xi_x
                m_y = F.relu(-0.8*10*torch.matmul(xi_x, self.template_matrix) + 0.2*m_y)
        return F.relu(self.fc_out2(F.relu(self.fc_out1(m_y))))
