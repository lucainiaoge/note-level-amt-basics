import os
import numpy as np
from scipy import signal
from datetime import datetime, date

import torch
from torch import nn
import torch.nn.functional as F

def midi_to_hz(midi_num):
    return np.power(2.0, (midi_num-69)/12) * 440.0

def normalize(x, std = 1):
    x_out = x - np.mean(x)
    if np.std(x_out) > 0:
        x_out /= np.std(x_out)*std
    return x_out

def smooth(x, window_len=5, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    x_normalized = normalize(x, std = 1)
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    
    if len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1):
        s = np.r_[x_normalized[window_len-1:0:-1],x_normalized,x_normalized[-2:-window_len-1:-1]]
        y = np.convolve(w/w.sum(),s,mode='valid')
        return y - y.mean()
    elif len(x.shape) == 2:
        w = w[np.newaxis,:]
        y = signal.convolve2d(w/w.sum(),x_normalized,mode='same')
        return y - y.mean(axis=1)
    else:
        raise (ValueError, "Input signal should have shape either (L,) or (B,L)")

def move_data_to_device(x, device = "cpu"):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x
    return x.to(device)

def move_list_data_to_device(list_x, device = "cpu"):
    list_len = len(list_x)
    list_x_torch = [None]*list_len
    for i in range(list_len):
        list_x_torch[i] = move_data_to_device(list_x[i], device)
    return list_x_torch

LOSS_SCALING = 10
def masked_mse_loss(velocity_roll_gt, velocity_roll_estimate, lambda_L1 = 0.1, use_mask = True):
    if use_mask:
        mask = (velocity_roll_gt > 0)
        loss_distance = F.mse_loss(velocity_roll_gt[mask] * LOSS_SCALING, velocity_roll_estimate[mask] * LOSS_SCALING)
    else:
        loss_distance = F.mse_loss(velocity_roll_gt * LOSS_SCALING, velocity_roll_estimate * LOSS_SCALING)
    loss_L1 = torch.norm(velocity_roll_estimate, p=1)
    return loss_distance + lambda_L1 * loss_L1

def bce(output, target, mask):
    eps = 1e-7
    output2 = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output2) - (1. - target) * torch.log(1. - output2)
    return torch.sum(matrix * mask) / torch.sum(mask)

def save_checkpoint(model, checkpoint_dir, i_iter, model_name = "AcousticE2E"):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    checkpoint_name = model_name+"_"+str(i_iter)+"_"+dt_string+".pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)

def save_checkpoint_np(np_array, checkpoint_dir, i_iter, model_name = "AcousticE2E"):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    checkpoint_name = model_name+"_"+str(i_iter)+"_"+dt_string+".npy"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    with open(checkpoint_path, 'wb') as f:
        np.save(f, np_array)
    