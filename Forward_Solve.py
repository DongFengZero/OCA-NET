'''PhyCRNet for solving spatiotemporal PDEs'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm
from matplotlib import cm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

# define the high-order finite difference kernels
lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

lapl_op2 = [[[[    0,   1, 0],
             [    1,   -4,   1],
             [0, 1,    0]]]]
solve = []
# generalized version
# def initialize_weights(module):
#     ''' starting from small initialized parameters '''
#     if isinstance(module, nn.Conv2d):
#         c = 0.1
#         module.weight.data.uniform_(-c*np.sqrt(1 / np.prod(module.weight.shape[:-1])),
#                                      c*np.sqrt(1 / np.prod(module.weight.shape[:-1])))

#     elif isinstance(module, nn.Linear):
#         module.bias.data.zero_()

def return_solve(t,x,y):
    return solve[t,0,x,y]

def init(x,y):
    return np.sin(np.pi*x) * np.sin(np.pi*y)

# specific parameters for burgers equation
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1  # 0.5
        module.weight.data.uniform_(-c * np.sqrt(1 / (3 * 3 * 320)),
                                    c * np.sqrt(1 / (3 * 3 * 320)))

    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, mask=None):
        x = self.embedding(src)
        x = self.pe(x)
        output = self.encoder(x, mask)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.out = nn.Linear(d_model, output_dim)

    def forward(self, trg, memory, src_mask=None, trg_mask=None):
        x = self.embedding(trg)
        x = self.pe(x)
        output = self.decoder(x, memory, tgt_mask=trg_mask, memory_mask=src_mask)
        output = self.out(output)
        return output

class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 3

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Whi =nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Whc =nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxo =nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):
        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())


class encoder_block(nn.Module):
    ''' encoder with CNN '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels,
                                          self.hidden_channels, self.input_kernel_size, self.input_stride,
                                          self.input_padding, bias=True, padding_mode='circular'))

        self.act = nn.ReLU()

        nn.init.zeros_(self.conv.bias)

    def forward(self,  x):
        return self.act(self.conv(x))


class PhyCRNet(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''

    def __init__(self, input_channels, hidden_channels,
                 input_kernel_size, input_stride, input_padding, dt, dx,
                 num_layers, upscale_factor, step=1, effective_step=[1]):

        super(PhyCRNet, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.backward_state = None
        self.forward_state = None
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.dx=dx
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        self.encoder = TransformerEncoder(input_dim=64, d_model=64, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.1)
        self.decoder = TransformerDecoder(output_dim=64, d_model=64, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.1)

        # self.fc61 = nn.Linear(1 * 64 * 64, 128)  # 第二层全连接层
        # self.fc62 = nn.Linear(128, 128)  # 第二层全连接层
        # self.fc63 = nn.Linear(128, 128)  # 第二层全连接层
        # self.fc64 = nn.Linear(128, 32 * 32)

        # self.fc66 = nn.Linear(128*2*14, 64)  #
        # self.fc67 = nn.Linear(64, 64)  # 第二层全连接层
        # self.fc68 = nn.Linear(64, 64)  # 第二层全连接层
        # self.fc69 = nn.Linear(64, 64)
        # self.fc70 = nn.Linear(64, 1)
        # self.fc71 = nn.Linear(64, 1)
        # self.fc72 = nn.Linear(28*64, 64)

        # self.fc2 = nn.Linear(32*32, 128)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, 64 * 64)

        # self.fc74 = nn.Linear(14*128, 64)  # 第二层全连接层
        # self.fc75 = nn.Linear(14*128, 64)  # 第二层全连接层
        # self.fc76 = nn.Linear(64, 64)  # 第二层全连接层
        # self.fc77 = nn.Linear(64, 64)  # 第二层全连接层
        # self.fc78 = nn.Linear(64, 64)
        # self.fc79 = nn.Linear(64, 1)
        # self.fc80 = nn.Linear(64, 1)

        # ConvLSTM(Forward)
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm_F{}'.format(i)
            cell = ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i])

            setattr(self, name, cell)
            self._all_layers.append(cell)

        #ConvLSTM(Backward)
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm_B{}'.format(i)
            cell = ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i])

            setattr(self, name, cell)
            self._all_layers.append(cell)

        # output layer

        self.output_layer = weight_norm(nn.Conv2d(2, 1, kernel_size=3, stride=1,
                                      padding=1, padding_mode='circular'))

        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        # initialize weights

        self.ref_sol = torch.load('E:/Image_Dataset/True/temp10.pt').cuda()
        self.apply(initialize_weights)

    def forward(self, forward_state,backward_state,x_t,x_tt):
        self.forward_state = forward_state
        internal_state_forward = []
        second_last_state_forward = []
        self.backward_state = backward_state
        internal_state_backward = []
        second_last_state_backward = []
        outputs4 = []
        outputs3 = []
        outputs2 = []
        outputs1 = []


        ref_speed = 45
        ref_speed = ref_speed * torch.ones((1, 1, 64, 64)).cuda()  # 生成一个波速为1的张量

        x_tt[:, :, 0, :] = 0
        x_tt[:, :, -1, :] = 0
        x_tt[:, :, :, 0] = 0
        x_tt[:, :, :, -1] = 0
        x_t[:, :, 0, :] = 0
        x_t[:, :, -1, :] = 0
        x_t[:, :, :, 0] = 0
        x_t[:, :, :, -1] = 0
        outputs4.append(x_tt)
        outputs4.append(x_t)
        outputs1.append(x_tt)
        outputs1.append(x_t)

        for step in range(0,self.step-1,1):
            x_tt = outputs1[-2]
            x_t = outputs1[-1]
            c3 = torch.squeeze(x_t, dim=1)
            x_temp2 = self.encoder(c3, mask=None)
            # convlstm forward
            x_t1=torch.unsqueeze(x_temp2, dim=1)
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm_F{}'.format(i)
                if step == 0:
                    (hx, c) = getattr(self, name).init_hidden_tensor(
                        prev_state=self.forward_state[i - self.num_encoder])
                    internal_state_forward.append((hx, c))

                # one-step forward
                (hx, c) = internal_state_forward[i - self.num_encoder]
                x_t1, new_c = getattr(self, name)(x_t1, hx, c)
                internal_state_forward[i - self.num_encoder] = (x_t1, new_c)
            # x, _ = self.lstm(x_temp2)
            x = x_t1
            x = torch.reshape(x,(1,64,64))
            x_temp2 = self.decoder(x_temp2, x, src_mask=None, trg_mask=None)
            # x = torch.tanh(self.fc2(x))
            # x = torch.tanh(self.fc3(x))
            # x_temp2 = torch.reshape(self.fc4(x),(1,1,64,64))#* (self.dt ** 2) / (self.dx ** 2)
            x_temp2 = torch.reshape(x_temp2, (1, 1, 64, 64))
            x_temp7 = (2 * x_t - x_tt) + x_temp2 * (45 ** 2) * (self.dt ** 2) / (self.dx ** 2)
            x_temp7[:,:, 0, :] = 0
            x_temp7[:, :, -1, :] = 0
            x_temp7[:, :, :, 0] = 0
            x_temp7[:, :, :, -1] = 0

            if step in self.effective_step:
                #outputs4.append(x_temp6)
                outputs1.append(x_temp7)

            if step == (self.step - 2):
                second_last_state_forward = internal_state_forward.copy()


        # for step in range(0, self.step-1, 1):
        #     x_tt = outputs3[-2]
        #     x_t = outputs3[-1]
        #
        #     c3 = torch.squeeze(x_t, dim=1)
        #     x_temp2 = self.encoder(c3, mask=None)
        #     # convlstm forward
        #     x_t1=torch.unsqueeze(x_temp2, dim=1)
        #     for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
        #         name = 'convlstm_B{}'.format(i)
        #         if step == 0:
        #             (hx, c) = getattr(self, name).init_hidden_tensor(
        #                 prev_state=self.forward_state[i - self.num_encoder])
        #             internal_state_backward.append((hx, c))
        #
        #         # one-step forward
        #         (hx, c) = internal_state_backward[i - self.num_encoder]
        #         x_t1, new_c = getattr(self, name)(x_t1, hx, c)
        #         internal_state_backward[i - self.num_encoder] = (x_t1, new_c)
        #     # x, _ = self.lstm(x_temp2)
        #     x = x_t1
        #     x = torch.reshape(x, (1, 64, 64))
        #     x_temp2 = self.decoder(x_temp2, x, src_mask=None, trg_mask=None)
        #     x_temp7 = (2 * x_t - x_tt) + x_temp2 * (45 ** 2) * (self.dt ** 2) / (self.dx ** 2)
        #     x_temp7[:, :, 0, :] = 0
        #     x_temp7[:, :, -1, :] = 0
        #     x_temp7[:, :, :, 0] = 0
        #     x_temp7[:, :, :, -1] = 0
        #
        #     if step in self.effective_step:
        #         outputs3.append(x_temp7)
        #
        #     if step == (self.step - 2):
        #         second_last_state_backward = internal_state_backward.copy()


        #outputs4 = torch.cat(tuple(outputs4), dim=0)
        # outputs3 = torch.cat(tuple(outputs3), dim=0)
        # outputs3 = torch.flip(outputs3, dims=[0])
        outputs1 = torch.cat(tuple(outputs1), dim=0)

        return outputs1, outputs2, outputs3, second_last_state_forward,second_last_state_backward,ref_speed


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt=0.01, dx=1):
        ''' Construct the derivatives, X = Width, Y = Height '''

        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.flag =False
        self.laplace = Conv2dDerivative(
            DerFilter=lapl_op,
            resol=(dx ** 2),
            kernel_size=5,
            name='laplace_operator').cuda()

        self.laplace2 = Conv2dDerivative(
            DerFilter=lapl_op2,
            resol=(dx ** 2),
            kernel_size=3,
            name='laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter=[[[-1, 0, 1]]],
            resol=(dt * 2),
            kernel_size=3,
            name='partial_t').cuda()

        self.dtt = Conv1dDerivative(
            DerFilter=[[[1, -2, 1]]],
            resol=(dt ** 2),
            kernel_size=3,
            name='partial_tt').cuda()

        self.ref_sol = torch.load('E:/Image_Dataset/True/temp10.pt').cuda()
        #self.ref_sol = torch.cat((self.ref_sol[0:1,:,:,:],self.ref_sol),dim=0)
        self.ref_sol2 = torch.load('E:/Image_Dataset/True/temp10.pt').cuda()
        self.ref_sol3 = torch.load('E:/Image_Dataset/True/temp10.pt').cuda()

    def get_ref_Loss(self):
        temp_res = self.ref_sol[:,:,:,:]
        #temp_res = torch.squeeze(temp_res,dim=1)
        return temp_res

    def get_ref_Loss2(self):
        temp_res = self.ref_sol2[:, :, :, :]
        #temp_res = torch.squeeze(temp_res, dim=1)
        return temp_res

    def get_ref_Loss3(self):
        temp_res = self.ref_sol3[:, :, :, :]
        #temp_res = torch.squeeze(temp_res, dim=1)
        return temp_res


    def get_local_Loss(self, res):
        local_res = torch.zeros((res.shape[0],1,14,14))
        for x in range(2,3,1):
            for y in range(14):
                local_res[:,:, x, y] = res[:,:,4 * (x + 1), 4 * (y + 1)]

        return local_res

    def get_local_Loss2(self, res):
        local_res = torch.zeros((res.shape[0],1, 14, 14))
        for x in range(14):
            for y in range(2,3,1):
                local_res[:,:, x, y] = res[:,:,4 * (x + 1), 4 * (y + 1)]

        return local_res

    def get_laplace_Loss(self, output, c):
        output1 = torch.squeeze(output, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        t_max = 0.01 * (int(output3.shape[0]) + 1)
        # dx = 0.01
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            #output3[:, 0, :] = output3[:, -1, :] = output3[:, :, 0] = output3[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            output3[(n + 1) - 2, :, :] = (-output1[n+1,:, :] +2 * output1[n, :, :] - output1[n - 1,:, :] )/((dt**2)*(c[2:3,:]**2))
            # output3[(n + 1) - 2, 1:-1, 1:-1] = (output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n, :-2,1:-1]) + \
            #                                    (output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n, 1:-1,:-2])

        return torch.unsqueeze(output3[:,:,:], dim=1)

    def get_laplace_Loss2(self, output, c):
        output1 = torch.squeeze(output, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        t_max = 0.01 * (int(output3.shape[0]) + 1)
        # dx = 0.01
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            #output3[:, 0, :] = output3[:, -1, :] = output3[:, :, 0] = output3[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            output3[(n + 1) - 2, :, :] = (-output1[n+1,:, :] +2 * output1[n, :, :] - output1[n - 1,:, :] )/((dt**2)*(c[:,2:3]**2))
            # output3[(n + 1) - 2, 1:-1, 1:-1] = (output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n, :-2,1:-1]) + \
            #                                    (output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n, 1:-1,:-2])

        return torch.unsqueeze(output3[:,:,:], dim=1)

    def get_laplace_Loss3(self, output, c):
        output1 = torch.squeeze(output, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        t_max = 0.01 * (int(output3.shape[0]) + 1)
        # dx = 0.01
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            output3[:, 0, :] = output3[:, -1, :] = output3[:, :, 0] = output3[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            output3[(n + 1) - 2, :, :] = (-output1[n+1,:, :] +2 * output1[n, :, :] - output1[n - 1,:, :] )/((dt**2)*(c[:,:]**2))
            # output3[(n + 1) - 2, 1:-1, 1:-1] = (output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n, :-2, 1:-1]) + \
            #                                    (output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n, 1:-1,:-2])

        return torch.unsqueeze(output3[:, :, :], dim=1)

    def get_phy_Loss(self, output, c):
        output1 = torch.squeeze(output, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        dx = dy = 1
        t_max = 0.01*(int(output3.shape[0])+1)
        r1, r2 = ((dt**2)*(45**2))/ (dx ** 2),((dt**2)*(45**2))/ (dy ** 2)
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            output1[:, 0, :] = output1[:, -1, :] = output1[:, :, 0] = output1[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            output1[(n+1), 1:-1, 1:-1] = (2 * output1[n, 1:-1, 1:-1] - output1[n - 1, 1:-1, 1:-1] )+ \
                                           r1* (
                                                   output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                       :-2, 1:-1]) + \
                                           r2*(
                                                   output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                       1:-1, :-2])
            # if ((n + 1) - 2) % 10 == 0 and n + 1!=2:
            #     t = int(((n + 1) - 2) / 10)
            #     output3[(t + 1) - 1, 1:-1, 1:-1] = output3[(n + 1)-2, 1:-1, 1:-1]

        return torch.unsqueeze(output1[:,:,:], dim=1)

    def get_phy_Loss2(self, output, c):
        output1 = torch.squeeze(output, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        dx = dy = 1
        t_max = 0.01*(int(output3.shape[0])+1)
        r1, r2 = ((dt**2)*(45**2))/ (dx ** 2),((dt**2)*(45**2))/ (dy ** 2)
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            output1[:, 0, :] = output1[:, -1, :] = output1[:, :, 0] = output1[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            output3[(n+1)-2, 1:-1, 1:-1] = (2 * output1[n, 1:-1, 1:-1] - output1[n - 1, 1:-1, 1:-1] )+ \
                                           r1* (
                                                   output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                       :-2, 1:-1]) + \
                                           r2*(
                                                   output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                       1:-1, :-2])
            # if ((n + 1) - 2) % 10 == 0 and n + 1!=2:
            #     t = int(((n + 1) - 2) / 10)
            #     output3[(t + 1) - 1, 1:-1, 1:-1] = output3[(n + 1)-2, 1:-1, 1:-1]

        return torch.unsqueeze(output3[:,:,:], dim=1)


def compute_loss(batchloss,output7, output0,output8, loss_func, id, ref_speed,bsize,ntb):
    ''' calculate the phycis loss '''
    # print(c1)
    mse_loss = nn.MSELoss(reduction='mean')

    # ref_speed2 = loss_func.get_local_Loss(ref_speed).cuda()
    # ref_speed3 = loss_func.get_local_Loss2(ref_speed).cuda()

    ref_speed = torch.squeeze(torch.squeeze(ref_speed, dim=0), dim=0)
    # ref_speed2 = torch.squeeze(torch.squeeze(ref_speed2, dim=0), dim=0)
    # ref_speed3 = torch.squeeze(torch.squeeze(ref_speed3, dim=0), dim=0)

    # print(ref_speed[0:1, 0:1])

    output11 = loss_func.get_phy_Loss2(output7, ref_speed)
    #ref_local_res2 = loss_func.get_local_Loss(output7).cuda()

    ref_local_sol = loss_func.get_ref_Loss().cuda()
    #ref_local_sol = torch.cat((ref_local_sol[0:1, :, :, :], ref_local_sol), dim=0)

    #ref_local_sol = loss_func.get_phy_Loss(ref_local_sol, ref_speed)
    #output101 = loss_func.get_phy_Loss2(ref_local_sol, ref_speed)
    #ref_local_res22 = loss_func.get_local_Loss2(output7).cuda()
    #ref_local_sol2 = loss_func.get_ref_Loss3().cuda()
    # ref_local_sol = torch.cat((ref_local_sol[0:1,:,:,:],ref_local_sol),dim=0)
    # ref_local_sol2 = torch.cat((ref_local_sol2[0:1, :, :, :], ref_local_sol2), dim=0)
    p_res2=0
    p_backward=0
    #output41=torch.cat((output7[0:2,:,:,:],output11),dim=0)
    for i in range(32):
        p_res2 += (bsize*(ntb-id-1)+i)*mse_loss(output11[i:i+1, :, :, :],output7[i+2:i+3,:,:,:]).cuda()
        # p_backward += (bsize*(ntb-id)-i)*mse_loss(output41[i:i+1, :, :, :],output8[i:i+1,:,:,:]).cuda()

    p_local2 = mse_loss(ref_local_sol[:, :, :, :], output7[:, :, :, :])
               #mse_loss(ref_local_sol[bsize*id:bsize*(id+1), :, :, :], output8[:-2, :, :, :])
    #p_local3 = mse_loss(ref_local_sol[2:, :, :, :],  output101[:, :, :, :]
    #
    # output01 = loss_func.get_local_Loss(output0).cuda()
    # output012 = loss_func.get_local_Loss2(output0).cuda()

    # output22 = loss_func.get_laplace_Loss(ref_local_sol[:, :, 2:3, :], ref_speed2)
    # output123 = loss_func.get_laplace_Loss2(ref_local_sol2[:, :, :, 2:3], ref_speed3)
    # p_laplace2 = mse_loss(output22, output01[1:, :, 2:3, :])+mse_loss(output123,output012[1:,:,:,2:3])

    # output32 = loss_func.get_phy_Loss2(loss_func.ref_sol, 45*torch.ones(64,64).cuda())
    # p_laplace5 = mse_loss(output32, loss_func.ref_sol[2:,:,:,:])

    # output32 = loss_func.get_laplace_Loss3(output7, ref_speed)
    # p_laplace3 = mse_loss(output32, output0)

    #p_speed = mse_loss(speed,ref_speed)

    loss = p_res2+p_backward+batchloss
    loss2 = p_local2
    # print(p_laplace2)
    # print(p_laplace3)
    return loss, loss2, p_res2, p_backward

def train(model, input, ref_res, forward_state, backward_state, n_iters, time_batch_size, learning_rate,
          dt, dx, save_path, pre_model_save_path, num_time_batch):
    train_loss_list = []
    second_last_state = []
    state_detached1 = []
    state_detached2 = []
    prev_output1 = []
    prev_output2 = []
    speed=[]

    batch_loss = 0.0
    batch_loss2 = 0.0
    batch_loss3 = 0.0
    batch_loss4 = 0.0
    best_loss = 1e4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load previous9 model
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.97)
    #model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler,
    #                                              pre_model_save_path)

    #for param_group in optimizer.param_groups:
        #print(param_group['lr'])

    loss_func = loss_generator(dt, dx)

    for epoch in range(n_iters):
        # input: [t,c,p,h,w]
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_loss2 = 0.0
        batch_loss3 = 0.0
        batch_loss4 = 0.0

        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                hidden_state1 = forward_state
                hidden_state2 = backward_state
                x_tt = input[1:2,:,:,:]
                x_t = input[0:1, :, :, :]
            else:
                hidden_state1 = state_detached1
                hidden_state2 = state_detached2
                x_tt = prev_output1[1:2, :, :, :].detach()  # second last output
                x_t = prev_output1[0:1, :, :, :].detach()  # second last output

            # output is a list
            output111, output0, output3, second_last_state_forward,second_last_state_backward, ref_speed= model(hidden_state1,hidden_state2,x_t,x_tt)
            # output4 = torch.cat((x_tt,x_t,output4), dim=0)
            #output3 = torch.cat((x_tt, x_t, output3), dim=0)


            # get loss
        # with torch.autograd.set_detect_anomaly(True):
            loss, loss2, loss3, loss4 = compute_loss(batch_loss,output111,output0,output3,loss_func,time_batch_id, ref_speed,time_batch_size,num_time_batch)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            batch_loss2 += loss2
            batch_loss3 += loss3
            batch_loss4 += loss4

            # update the state and output for next batch
            prev_output1 = output111
            prev_output2 = output3
            state_detached1 = []
            state_detached2 = []
            for i in range(len(second_last_state_forward)):
                (h, c) = second_last_state_forward[i]
                state_detached1.append((h.detach(), c.detach()))
            for i in range(len(second_last_state_backward)):
                (h, c) = second_last_state_backward[i]
                state_detached2.append((h.detach(), c.detach()))

            if epoch % 5000 == 0 and epoch != 0:
                torch.save(prev_output1, 'E:/Image_Dataset/True/tensor_t1_' + str(epoch)+'_'+str(time_batch_id)+ '.pt')

        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.10f loss_true: %.10f loss_res: %.10f loss_backward: %.10f' % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0),
                                            batch_loss, batch_loss2, batch_loss3, batch_loss4))
        train_loss_list.append(batch_loss)

        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss

        cmap = cm.get_cmap('jet')
        cmap1 = cm.get_cmap('jet')
        # if epoch%500==0 and epoch!=0:
        #     #speed = torch.squeeze(torch.squeeze(speed, dim=0), dim=0)
        #     torch.save(speed, 'E:/Image_Dataset/True/res/speed_' + str(epoch) + '.pt')
        #     plt.imshow(speed.detach().cpu().numpy().squeeze(), cmap=cmap1)
        #     plt.colorbar()
        #     plt.show()
        # if epoch%1000==0 and epoch!=0:
        #     torch.save(prev_output1, 'E:/Image_Dataset/True/res/tensor_LSTMF_'+str(epoch)+'.pt')
            # plt.imshow(prev_output1[i].detach().cpu().numpy().squeeze(), cmap=cmap)
            # plt.colorbar()
            # plt.show()

    return train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

if __name__ == '__main__':
    Lx = Ly = 64  # Length of the 2D domain
    dx = dy = 1  # Spatial step
    x = np.arange(0, Lx, dx)
    y = np.arange(0, Ly, dy)
    X, Y = np.meshgrid(x, y)
    input_tensor = torch.load("E:/Image_Dataset/True/temp10.pt")
    input_tensor2 = torch.load("E:/Image_Dataset/True/temp10.pt")
    input2 = torch.zeros((2,1,64,64)).cuda()
    input2[0:2,:,:,:]=input_tensor[-2:,:,:,:]

    # set initial states for convlstm
    num_convlstm = 1
    (h1, c0) = (torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64))
    backward_state = []
    forward_state=[]
    for i in range(num_convlstm):
        forward_state.append((h1, c0))
        backward_state.append((h1, c0))

    # grid parameters
    time_steps = input_tensor.shape[0]-2

    ################# build the model #####################-``
    time_batch_size = 32
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters_adam = 50001
    lr_adam = 1e-4  # 1e-3
    pre_model_save_path = './checkpoint' \
                          '500.pt'
    model_save_path = './checkpoint1000.pt'
    fig_save_path = './figures/'

    dt = 0.01
    dx = 1

    model = PhyCRNet(
        input_channels=1,
        hidden_channels=[1, 1, 1, 1,1,1],
        input_kernel_size=[4, 4, 4, 3,3,3],
        input_stride=[2, 2, 2, 1, 1,1],
        input_padding=[1, 1, 1, 1, 1,1],
        dt=dt,
        dx=dx,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps,
        effective_step=effective_step).cuda()

    start = time.time()
    train_loss = train(model, input2, solve, forward_state,backward_state, n_iters_adam, time_batch_size,
                       lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch)
    end = time.time()

    np.save('./train_loss', train_loss)
    print('The training time is: ', (end - start))