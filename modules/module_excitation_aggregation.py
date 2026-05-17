# author: JingXiaolun
# date: 2023.05.11
# description: excitation and aggregation block (including squeeze and expand)
# paper: https://arxiv.org/pdf/1709.01507.pdf(Squeeze-and-Excitation Networks)

import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

################################################################# Excitation Block ###################################################
class Excitation_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio, activation):
        super().__init__()
        ''' params
        frame_length: length of extracted frame embeddings
        reduction_ratio: ratio used to reduce the length of video frames
        activation: activationivation between fc layers (reduction_ratio > 1 -> activation=nn.ReLU() / reduction_ratio<1 -> activation=nn.ReLU6())
        '''
        reduction_frame_length = int(frame_length // reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(frame_length, reduction_frame_length),
                activation,
                nn.Linear(reduction_frame_length, frame_length),
                nn.Sigmoid()
                )

    def forward(self, x):
        b, n = x.shape[:2]
        # step1: squeeze 
        weight = self.avgpool(x).contiguous().view(b, -1)
        # step2: excitation
        weight = self.fc(weight).contiguous().view(b, n, 1)
        #print('weight_exc:', weight.squeeze().data)
        # step3: scale
        out = x * weight
        return out

################################################################# Aggregation Block ###################################################
class Aggregation_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio, activation):
        super().__init__()
        ''' params
        frame_length: length of extracted frame embeddings
        reduction_ratio: ratio used to reduce the length of video frames
        activation: activation between fc layers (reduction_ratio > 1 -> activation=nn.ReLU() / reduction_ratio<1 -> activation=nn.ReLU6())
        '''
        reduction_frame_length = int(frame_length // reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(frame_length, reduction_frame_length),
                activation,
                nn.Linear(reduction_frame_length, frame_length),
                nn.Softmax(dim=-1)
                )

    def forward(self, x):
        b, n = x.shape[:2]
        # step1: squeeze 
        weight = self.avgpool(x).contiguous().view(b, -1)
        # step2: excitation
        weight = self.fc(weight).contiguous().view(b, n)
        #print('weight_agg:', weight.data)
        # step3: aggregate
        out = torch.einsum('b n d, b n -> b d', x, weight)
        return out

################################################################# Excitation and Aggregation Block ###################################################
## Squeeze Excitation_and Squeeze Aggregation_Block /  Expand Excitation_and Expand Aggregation_Block
class Excitation_Aggregation_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio):
        super().__init__()
        ''' params
        frame_length: length of extractivationed frame embeddings
        reduction_ratio: ratio used to reduce the length of video frames
        '''
        #activation = nn.ReLU(inplace=True) if reduction_ratio > 1 else nn.ReLU6(inplace=True)
        activation = nn.ReLU(inplace=True) 
        self.scale = Excitation_Block(frame_length, reduction_ratio, activation)
        self.aggregate = Aggregation_Block(frame_length, reduction_ratio, activation)

    def forward(self, x):
        return self.aggregate(self.scale(x))

## Squeeze Excitation and Expand Aggregation
class Squeeze_Excitation_Expand_Aggregation_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio):
        super().__init__()
        ''' params
        frame_length: length of extractivationed frame embeddings
        reduction_ratio: ratio used to reduce the length of video frames
        '''
        assert reduction_ratio > 1, 'The reduction ratio must be larger than 1'
        scale_act, aggregate_act = nn.ReLU(inplace=True), nn.ReLU6(inplace=True)
        self.scale = Excitation_Block(frame_length, reduction_ratio, scale_act)
        self.aggregate = Aggregation_Block(frame_length, 1/reduction_ratio, aggregate_act)

    def forward(self, x):
        return self.aggregate(self.scale(x))

## Expand Excitation and Squeeze Aggregation
class Expand_Excitation_Squeeze_Aggregation_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio):
        super().__init__()
        ''' params
        frame_length: length of extractivationed frame embeddings
        reduction_ratio: ratio used to reduce the length of video frames
        '''
        assert reduction_ratio < 1, 'The reduction ratio must be smaller than 1'
        scale_act, aggregate_act = nn.ReLU6(inplace=True), nn.ReLU(inplace=True)
        self.scale = Excitation_Block(frame_length, reduction_ratio, scale_act)
        self.aggregate = Aggregation_Block(frame_length, 1/reduction_ratio, aggregate_act)

    def forward(self, x):
        return self.aggregate(self.scale(x))

############################################# Main Function ###############################################################
if __name__ == '__main__':
    # step1: define frame_length and reduction_ratio
    frame_length, reduction_ratio = 12, 0.25

    # step2: construct Excitation_Aggregation_Block
    excitation_aggregation_block = Excitation_Aggregation_Block(frame_length, reduction_ratio)
    lstm_visual = nn.LSTM(input_size=512, hidden_size=512,
                                           batch_first=True, bidirectional=False, num_layers=1)

    model = lstm_visual

    # step3: define input
    input = torch.randn(1, 12, 512)
    
    # step4: 
    summary(model, input_size=(12, 512), batch_size=-1, device='cpu')
    #flops, params = profile(model, (input,))
    #print('GFLOPs: %.2f B, params: %.2f M' % (flops , params))
