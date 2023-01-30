import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, momentum=0.1):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, 
                              stride=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, 
                              stride=1, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels, momentum)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=2, pool_type='avg'):
        """
        Args:
            input: (batch_size, in_channels, time_steps)

        Outputs:
            output: (batch_size, out_channels, new_time_steps)
        """

        x = F.relu_(self.bn1(self.conv1(input))) #x: (batch_size, out_channels, time_steps1)
        x = F.relu_(self.bn2(self.conv2(x))) #x: (batch_size, out_channels, time_steps2)
        
        #x: (batch_size, out_channels, time_steps3)
        if pool_type == 'avg':
            x = F.avg_pool1d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool1d(x, kernel_size=pool_size)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.1):
        
        super(ConvBlock2D, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
            input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
            output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input))) #x: (batch_size, out_channels, time_steps, freq_bins)
        x = F.relu_(self.bn2(self.conv2(x))) #x: (batch_size, out_channels, time_steps, freq_bins)
        
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
            #x: (batch_size, out_channels, int(time_steps/pool_size[0]), int(freq_bins/pool_size[1]))
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
            #x: (batch_size, out_channels, time_steps*pool_size[0], freq_bins*pool_size[1])
        return x
