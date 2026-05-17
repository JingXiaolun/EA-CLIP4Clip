# author: JingXiaolun
# date: 2022.04.06
# description: SE Block(squeeze and excitation)
# paper: https://arxiv.org/pdf/1709.01507.pdf(Squeeze-and-Excitation Networks)

import torch
import torch.nn as nn

################################################################# Squeeze and Excitation in frame dimension ###################################################
class Squeeze_Excitation_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio):
        super().__init__()
        ''' params
        frame_length: length of extracted frame embeddings
        reduction_ratio: ratio used to reduce the length of video frame to constrcut nonlinear relations between different frames
        '''
        reduction_frame_length = int(frame_length // reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(frame_length, reduction_frame_length),
                nn.ReLU(inplace=True),
                nn.Linear(reduction_frame_length, frame_length),
                nn.Sigmoid()
                )

    def forward(self, x):
        b, n = x.shape[:2]
        # step1: squeeze 
        weight = self.avgpool(x).contiguous().view(b, -1)
        # step2: excitation
        weight = self.fc(weight).contiguous().view(b, n, 1)
        # step3: scale
        out = x * weight
        return out

class Squeeze_Aggregation_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio):
        super().__init__()
        ''' params
        frame_length: length of extracted frame embeddings
        reduction_ratio: ratio used to reduce the length of video frame to constrcut nonlinear relations between different frames
        '''
        reduction_frame_length = int(frame_length // reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(frame_length, reduction_frame_length),
                nn.ReLU(inplace=True),
                nn.Linear(reduction_frame_length, frame_length),
                nn.Softmax(dim=-1)
                )

    def forward(self, x):
        b, n = x.shape[:2]
        # step1: squeeze 
        weight = self.avgpool(x).contiguous().view(b, -1)
        # step2: excitation
        weight = self.fc(weight).contiguous().view(b, n)
        # step3: aggregate
        out = torch.einsum('b n d, b n -> b d', x, weight)
        return out

class SE_Block(nn.Module):
    def __init__(self, frame_length, reduction_ratio):
        super().__init__()
        ''' params
        frame_length: length of extracted frame embeddings
        reduction_ratio: ratio used to reduce the length of video frame to constrcut nonlinear relations between different frames
        '''
        self.scale = SE_Scale_Block(frame_length, reduction_ratio)
        self.aggregate = SE_Aggregate_Block(frame_length, reduction_ratio)

    def forward(self, x):
        return self.aggregate(self.scale(x))
