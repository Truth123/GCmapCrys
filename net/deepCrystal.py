import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Union, Tuple, Optional


def single_value(x: Union[int, Tuple[int]]):
    if isinstance(x, int):
        return x
    else:
        return x[0]


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Conv1d_same_padding(nn.Conv1d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int]], 
        stride: Union[int, Tuple[int]] = 1, 
        bias: bool = True
    ):
        super(Conv1d_same_padding, self).__init__(in_channels, out_channels, kernel_size, stride, bias=bias)
    
    def forward(self, input: Tensor) -> Tensor:
        input_rows = input.size(2)
        filter_rows = self.weight.size(2)
        stride = single_value(self.stride)
        out_rows = (input_rows + stride - 1) // stride
        padding_rows = max(0, (out_rows - 1) * stride + filter_rows - input_rows)
        rows_odd = (padding_rows % 2 != 0)

        if rows_odd:
            input = F.pad(input, [0, int(rows_odd)])

        return F.conv1d(input, self.weight, self.bias, self.stride,
                    padding=padding_rows // 2)


class MaxPool1D_same_padding(nn.MaxPool1d):
    def __init__(
        self, 
        kernel_size: Union[int, Tuple[int]], 
        stride: Optional[Union[int, Tuple[int]]] = None, 
        return_indices: bool = False, 
        ceil_mode: bool = False
    ) -> None:
        super(MaxPool1D_same_padding, self).__init__(kernel_size, stride, return_indices=return_indices, ceil_mode=ceil_mode)
    

    def forward(self, input: Tensor) -> Tensor:
        input_rows = input.size(2)
        kernel_size = single_value(self.kernel_size)
        stride = single_value(self.stride)
        out_rows = (input_rows + stride - 1) // stride
        padding_rows = max(0, (out_rows - 1) * stride + kernel_size - input_rows)
        rows_odd = (padding_rows % 2 != 0)

        if rows_odd:
            input = F.pad(input, [0, int(rows_odd)])
        return F.max_pool1d(input, self.kernel_size, self.stride,
                            padding = padding_rows // 2, 
                            ceil_mode = self.ceil_mode, return_indices = self.return_indices)



class myConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        conv_kernel_size: Union[int, Tuple[int]], 
        pool_kernel_size: Union[int, Tuple[int]], 
        conv_stride: Union[int, Tuple[int]] = 1, 
        pool_stride: Optional[Union[int, Tuple[int]]] = None,
        relu_inplace: bool = False,
        ):
        super(myConv1D, self).__init__()
        self.layer = nn.Sequential(
            Conv1d_same_padding(in_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=relu_inplace),
            MaxPool1D_same_padding(kernel_size=pool_kernel_size, stride=pool_stride)
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return self.layer(input)


class DeepCrystal(nn.Module):
    def __init__(self, conf):
        super(DeepCrystal, self).__init__()
        self.emb_layer = nn.Embedding(168, 50)
        self.dropout = nn.Dropout(0, inplace=True)

        in_channel = 590 + 50
        out_channel = 64
        self.conv_1_1 = myConv1D(in_channel, out_channel, conv_kernel_size=2, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_1_2 = myConv1D(in_channel, out_channel, conv_kernel_size=3, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_1_3 = myConv1D(in_channel, out_channel, conv_kernel_size=8, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_1_4 = myConv1D(in_channel, out_channel, conv_kernel_size=9, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_1_5 = myConv1D(in_channel, out_channel, conv_kernel_size=4, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_1_6 = myConv1D(in_channel, out_channel, conv_kernel_size=5, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_1_7 = myConv1D(in_channel, out_channel, conv_kernel_size=6, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_1_8 = myConv1D(in_channel, out_channel, conv_kernel_size=7, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        
        in_channel = out_channel * 8
        out_channel = 64
        self.conv_2_1 = myConv1D(in_channel, out_channel, conv_kernel_size=11, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_2_2 = myConv1D(in_channel, out_channel, conv_kernel_size=13, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_2_3 = myConv1D(in_channel, out_channel, conv_kernel_size=15, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)

        in_channel = out_channel * 3
        out_channel = 64
        self.conv_3_1 = myConv1D(in_channel, out_channel, conv_kernel_size=5, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_3_2 = myConv1D(in_channel, out_channel, conv_kernel_size=9, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)
        self.conv_3_3 = myConv1D(in_channel, out_channel, conv_kernel_size=13, conv_stride=1, relu_inplace=True, pool_kernel_size=5, pool_stride=1)

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(800*out_channel*3, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.apply(weight_init)

    def forward(self, x_emb, x_feature):
        x_emb = self.emb_layer(x_emb)  ## (N, 800) ==> (N, 800, 50)
        x_emb = x_emb.permute(0, 2, 1) ## (N, 50, 800)
        x = torch.cat((x_emb, x_feature), dim=1)  ## (N, 590+50, 800)

        x_1_1 = self.conv_1_1(x)
        x_1_2 = self.conv_1_2(x)
        x_1_3 = self.conv_1_3(x)
        x_1_4 = self.conv_1_4(x)
        x_1_5 = self.conv_1_5(x)
        x_1_6 = self.conv_1_6(x)
        x_1_7 = self.conv_1_7(x)
        x_1_8 = self.conv_1_8(x)
        x_1 = torch.cat((x_1_1, x_1_2, x_1_3, x_1_4, x_1_5, x_1_6, x_1_7, x_1_8), dim=1)
        x_1 = self.dropout(x_1)

        x_2_1 = self.conv_2_1(x_1)
        x_2_2 = self.conv_2_2(x_1)
        x_2_3 = self.conv_2_3(x_1)
        x_2 = torch.cat((x_2_1, x_2_2, x_2_3), dim=1)
        x_2 = self.dropout(x_2)

        x_3_1 = self.conv_3_1(x_2)
        x_3_2 = self.conv_3_2(x_2)
        x_3_3 = self.conv_3_3(x_2)
        x_3 = torch.cat((x_3_1, x_3_2, x_3_3), dim=1)
        x_3 = self.dropout(x_3)

        x = self.flatten(x_3)
        out = self.dense(x)
        
        return out

        
