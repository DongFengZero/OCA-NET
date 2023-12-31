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
        #output = src + output
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.out = nn.Linear(d_model, output_dim)
        self.tgt_mask = torch.triu(torch.ones(nhead, d_model,d_model), diagonal=1).cuda()
        self.tgt_mask = self.tgt_mask.masked_fill(self.tgt_mask == 1, float('-inf'))
        self.src_mask = torch.zeros(nhead, d_model,d_model).cuda()
        self.src_mask = self.tgt_mask.masked_fill(self.tgt_mask == 1, float('-inf'))

    def forward(self, trg, memory, src_mask=None, trg_mask=None):
        x = self.embedding(trg)
        x = self.pe(x)
        x = torch.squeeze(x, dim=0)
        memory = torch.squeeze(memory, dim=0)
        #output = self.decoder(x, memory, tgt_mask=trg_mask, memory_mask=src_mask)
        output = self.decoder(x, memory, self.tgt_mask,self.src_mask)
        output = self.out(output)
        #output = memory + output
        output = torch.unsqueeze(output,dim=0)
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
        self.step2 = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.dx=dx
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=0, output_padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=0, output_padding=0, bias=False)

        self.encoder = TransformerEncoder(input_dim=64, d_model=64, nhead=4, num_layers=1, dim_feedforward=128)
        self.decoder = TransformerDecoder(output_dim=64, d_model=64, nhead=4, num_layers=1, dim_feedforward=128)

        self.fc91 = nn.Linear(2 * 64 * 64, 64)  # 第二层全连接层
        self.fc92 = nn.Linear(64, 64)  # 第二层全连接层
        self.fc93 = nn.Linear(64, 64)  # 第二层全连接层
        self.fc94 = nn.Linear(64, 1 * 64 * 64)

        self.fc61 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
        self.fc62 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
        self.fc63 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
        self.fc64 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
        #self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

        self.fc74 = nn.Linear(60, 64)  # 第二层全连接层
        self.fc174 = nn.Linear(30, 64)  # 第二层全连接层
        self.fc75 = nn.Linear(64, 64)  # 第二层全连接层
        self.fc81 = nn.Linear(64, 1)
        self.fc181 = nn.Linear(64, 1)

        self.fc82 = nn.Linear(200, 64)  # 第二层全连接层
        self.fc83 = nn.Linear(64, 64)  # 第二层全连接层
        self.fc84 = nn.Linear(64, 2*64*64)

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
        self.ref_sol1 = torch.load('E:/Image_Dataset/True/temp10.pt').cuda()
        self.ref_sol2 = torch.zeros((self.ref_sol1.shape[0], 1, 10, 10)).cuda()
        for x in range(10):
            for y in range(10):
                self.ref_sol2[:, :, x, y] = self.ref_sol1[:, :, 6 * (x + 1), 6 * (y + 1)]
        # self.ref_sol3 = torch.zeros((self.ref_sol.shape[0], 1, 14, 14)).cuda()
        # for x in range(14):
        #     for y in range(2,3,1):
        #         self.ref_sol3[:, :, x, y] = self.ref_sol[:, :, 4 * (x + 1), 4 * (y + 1)]

        self.apply(initialize_weights)

    def forward(self, nid, bsize, forward_state,backward_state,x_t,x_tt,ref_speed,ref_speed1,chen0,x_t2,x_tt2):
        self.forward_state = forward_state
        internal_state_forward = []
        second_last_state_forward = []
        self.backward_state = backward_state
        internal_state_backward = []
        second_last_state_backward = []

        outputs3 = []

        # c6 = torch.reshape(self.ref_sol2[0:2,:,:,:],(1,2*10*10))
        # c6 = torch.tanh(self.fc82(c6))
        # c6 = torch.tanh(self.fc83(c6))
        # c6 = torch.reshape(self.fc84(c6),(2,1,64,64))
        # x_tt = c6[0:1,:,:,:]
        # x_t = c6[1:2,:,:,:]

        ref_speed2 = 0
        # temp = torch.ones((1,120)).cuda()
        for step in range(0, self.ref_sol2.shape[0] - 2, 1):
            temp = torch.reshape(self.ref_sol2[step:step + 3, :, 1, :], (1, 3 * 10))
            temp = torch.cat((temp, torch.reshape(self.ref_sol2[step:step + 3, :, :, 1], (1, 3 * 10))), dim=0)
            temp = torch.reshape(temp, (1, 60))
            # c5 = torch.reshape(c5, (1, 3 *10 * 10))
            c5 = torch.tanh(self.fc74(temp))
            c5 = torch.tanh(self.fc75(c5))
            c2 = torch.reshape(F.softplus(self.fc81(c5)), (1, 1))
            ref_speed2 = (ref_speed2 * step + c2) / (step + 1)
            # ref_speed31 = (ref_speed31 * id + ref_speed21) / (id + 1)

        x_init2 = x_tt.clone()
        x_init = x_t.clone()
        outputs1 = []

        outputs1.append(x_tt2)
        outputs1.append(x_t2)
        outputs3.append(x_tt2)

        ref_speed1 = torch.ones((1,1,64,64)).cuda() * 45
        ref_speed = torch.ones((1,1,64,64)).cuda() * 45
        ref_speed3 = torch.ones((1,1,64,64)).cuda() * 45

        outputs4 = []
        outputs6 = []
        outputs2 = []
        outputs11 = []
        outputs62 = []

        outputs11.append(x_init2)
        outputs11.append(x_init)
        outputs6.append(x_init2)

        for step in range(0, self.step2 - 1, 1):
            x_tt = outputs11[-2].clone()
            x_t = outputs11[-1].clone()
            # for x in range(10):
            #     for y in range(10):
            #         x_tt[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[step:step + 1, :, x, y]
            #         x_t[:, :, 6*(x+1), 6*(y+1)] =self.ref_sol2[step+1:step+2, :, x, y]
            c3 = torch.squeeze(x_t, dim=1)
            x_temp2 = self.encoder(c3, mask=None)
            outputs2.append(x_temp2)
            # convlstm forward
            x_t1 = torch.unsqueeze(x_temp2, dim=1)
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
            x = torch.reshape(x, (1, 64, 64))
            x_temp3 = self.decoder(x_temp2, x, src_mask=None, trg_mask=None)
            x_temp3 = torch.reshape(x_temp3, (1, 1, 64, 64))
            x_temp4 = x_temp3.clone()
            x_temp3[:, :, 1:-1, 1:-1] = (2 * x_t[:, :, 1:-1, 1:-1] - x_tt[:, :, 1:-1, 1:-1]) + x_temp4[:, :, 1:-1,
                                                                                               1:-1] * \
                                        (ref_speed2 ** 2) * (self.dt ** 2) / (self.dx ** 2)
            x_temp3[:,:,0,:]=x_temp3[:,:,-1,:]=x_temp3[:,:,:,0]=x_temp3[:,:,:,-1]=0
            if step in self.effective_step:
                outputs11.append(x_temp3)

            if step == (self.step2 - 2):
                second_last_state_forward = internal_state_forward.copy()

        outputs4.append(outputs11[-1])
        outputs4.append(outputs11[-2])
        for step in range(0, self.step2 - 1, 1):
            x_tt1 = outputs11[-(step + 1)].clone()
            x_t2 = outputs11[-(step + 2)].clone()
            # for x in range(10):
            #     for y in range(10):
            #         x_tt1[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[self.ref_sol2.shape[0]-1:self.ref_sol2.shape[0], :, x, y]
            #         x_t2[:, :, 6*(x+1), 6*(y+1)] =self.ref_sol2[self.ref_sol2.shape[0]-2:self.ref_sol2.shape[0]-1, :, x, y]
            # x_t = outputs2[-(step + 1)]
            # c3 = torch.squeeze(x_t, dim=1)
            # x_temp2 = self.encoder1(c3, mask=None)
            # convlstm forward
            x_temp2 = outputs2[-(step + 1)]
            x_t1 = torch.unsqueeze(x_temp2, dim=1)
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm_B{}'.format(i)
                if step == 0:
                    (hx, c) = getattr(self, name).init_hidden_tensor(
                        prev_state=self.backward_state[i - self.num_encoder])
                    internal_state_backward.append((hx, c))

                # one-step forward
                (hx, c) = internal_state_backward[i - self.num_encoder]
                x_t1, new_c = getattr(self, name)(x_t1, hx, c)
                internal_state_backward[i - self.num_encoder] = (x_t1, new_c)
            # x, _ = self.lstm(x_temp2)
            x = x_t1
            x = torch.reshape(x, (1, 64, 64))
            x_temp3 = self.decoder(x_temp2, x, src_mask=None, trg_mask=None)
            x_temp3 = torch.reshape(x_temp3, (1, 1, 64, 64))
            x_temp4 = x_temp3.clone()
            x_temp3[:, :, 1:-1, 1:-1] = (2 * x_t2[:, :, 1:-1, 1:-1] - x_tt1[:, :, 1:-1, 1:-1]) + x_temp4[:, :, 1:-1,
                                                                                                 1:-1] \
                                        * (ref_speed2 ** 2) * (self.dt ** 2) / (self.dx ** 2)
            x_temp3[:,:,0,:]=x_temp3[:,:,-1,:]=x_temp3[:,:,:,0]=x_temp3[:,:,:,-1]=0
            if step in self.effective_step:
                outputs4.append(x_temp3)

            if step == (self.step2 - 2):
                second_last_state_backward = internal_state_backward.copy()

        outputs4 = torch.cat(tuple(outputs4), dim=0)
        outputs4 = torch.flip(outputs4, dims=[0])
        outputs11 = torch.cat(tuple(outputs11), dim=0)
        # outputs4 = torch.cat((outputs11[0:2, :, :, :], outputs4[1:, :, :, :]), dim=0)
        outputs6.append(x_init2)

        for step in range(0, self.step2 - 1, 1):
            x_tt = outputs6[-2].clone()
            x_t = outputs6[-1].clone()
            # for x in range(10):
            #     for y in range(10):
            #         x_tt[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[step:step + 1, :, x, y]
            #         x_t[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[step + 1:step + 2, :, x, y]
            x_temp3 = torch.cat((outputs11[step + 2:step + 3, :, :, :], outputs4[step + 2:step + 3, :, :, :]), dim=0)
            x_temp3 = torch.reshape(x_temp3, (1, 2 * 64 * 64))
            x_temp3 = torch.tanh(self.fc91(x_temp3))
            x_temp3 = torch.tanh(self.fc92(x_temp3))
            # x_temp3=torch.tanh(self.fc63(x_temp3))
            x_temp3 = torch.reshape(self.fc94(x_temp3), (1, 1, 64, 64))
            x_temp4 = x_temp3.clone()
            x_temp3[:, :, 1:-1, 1:-1] = (2 * x_t[:, :, 1:-1, 1:-1] - x_tt[:, :, 1:-1, 1:-1]) + x_temp4[:, :, 1:-1,
                                                                                               1:-1] * \
                                        (ref_speed2 ** 2) * (self.dt ** 2) / (self.dx ** 2)
            x_temp3[:,:,0,:]=x_temp3[:,:,-1,:]=x_temp3[:,:,:,0]=x_temp3[:,:,:,-1]=0

            if step in self.effective_step:
                outputs6.append(x_temp3)

            # if step == (self.step - 2):
            #     second_last_state_backward = internal_state_backward.copy()

        # outputs62 = torch.cat(tuple(outputs6), dim=0)
        #
        # outputs4 = []
        # outputs6 = []
        # outputs2 = []
        # outputs11 = []
        # #outputs62 = []
        #
        # outputs11.append(x_init2)
        # outputs11.append(x_init)
        # outputs6.append(x_init2)
        # for step in range(0, self.step-1, 1):
        #     x_tt = outputs11[-2].clone()
        #     x_t = outputs11[-1].clone()
        #     # for x in range(10):
        #     #     for y in range(10):
        #     #         x_tt[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[step:step + 1, :, x, y]
        #     #         x_t[:, :, 6*(x+1), 6*(y+1)] =self.ref_sol2[step+1:step+2, :, x, y]
        #     c3 = torch.squeeze(x_t, dim=1)
        #     x_temp2 = self.encoder(c3, mask=None)
        #     outputs2.append(x_temp2)
        #     # convlstm forward
        #     x_t1 = torch.unsqueeze(x_temp2, dim=1)
        #     for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
        #         name = 'convlstm_F{}'.format(i)
        #         if step == 0:
        #             (hx, c) = getattr(self, name).init_hidden_tensor(
        #                 prev_state=self.forward_state[i - self.num_encoder])
        #             internal_state_forward.append((hx, c))
        #
        #         # one-step forward
        #         (hx, c) = internal_state_forward[i - self.num_encoder]
        #         x_t1, new_c = getattr(self, name)(x_t1, hx, c)
        #         internal_state_forward[i - self.num_encoder] = (x_t1, new_c)
        #     # x, _ = self.lstm(x_temp2)
        #     x = x_t1
        #     x = torch.reshape(x, (1, 64, 64))
        #     x_temp3 = self.decoder(x_temp2,x, src_mask=None, trg_mask=None)
        #     x_temp3 = torch.reshape(x_temp3, (1, 1, 64, 64))
        #     x_temp4 = x_temp3.clone()
        #     x_temp3[:,:,1:-1,1:-1] = (2 *  x_t[:,:,1:-1,1:-1] - x_tt[:,:,1:-1,1:-1]) + x_temp4[:,:,1:-1,1:-1]*\
        #                              (ref_speed1[:, :, 1:-1, 1:-1] ** 2) * (self.dt ** 2) / (self.dx ** 2)
        #     #x_temp3[:,:,0,:]=x_temp3[:,:,-1,:]=x_temp3[:,:,:,0]=x_temp3[:,:,:,-1]=0
        #     if step in self.effective_step:
        #         outputs11.append(x_temp3)
        #
        #     if step == (self.step-2):
        #         second_last_state_forward = internal_state_forward.copy()
        #
        # outputs4.append(outputs11[-1])
        # outputs4.append(outputs11[-2])
        # for step in range(0, self.step-1, 1):
        #     x_tt1 = outputs11[-(step + 1)].clone()
        #     x_t2 = outputs11[-(step + 2)].clone()
        #     # for x in range(10):
        #     #     for y in range(10):
        #     #         x_tt1[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[self.ref_sol2.shape[0]-1:self.ref_sol2.shape[0], :, x, y]
        #     #         x_t2[:, :, 6*(x+1), 6*(y+1)] =self.ref_sol2[self.ref_sol2.shape[0]-2:self.ref_sol2.shape[0]-1, :, x, y]
        #     #x_t = outputs2[-(step + 1)]
        #     # c3 = torch.squeeze(x_t, dim=1)
        #     # x_temp2 = self.encoder1(c3, mask=None)
        #     # convlstm forward
        #     x_temp2 = outputs2[-(step + 1)]
        #     x_t1 = torch.unsqueeze(x_temp2, dim=1)
        #     for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
        #         name = 'convlstm_B{}'.format(i)
        #         if step == 0:
        #             (hx, c) = getattr(self, name).init_hidden_tensor(
        #                 prev_state=self.backward_state[i - self.num_encoder])
        #             internal_state_backward.append((hx, c))
        #
        #         # one-step forward
        #         (hx, c) = internal_state_backward[i - self.num_encoder]
        #         x_t1, new_c = getattr(self, name)(x_t1, hx, c)
        #         internal_state_backward[i - self.num_encoder] = (x_t1, new_c)
        #     # x, _ = self.lstm(x_temp2)
        #     x = x_t1
        #     x = torch.reshape(x, (1, 64, 64))
        #     x_temp3 = self.decoder(x_temp2,x, src_mask=None, trg_mask=None)
        #     x_temp3 = torch.reshape(x_temp3, (1, 1, 64, 64))
        #     x_temp4 = x_temp3.clone()
        #     x_temp3[:,:,1:-1,1:-1] = (2 *  x_t2[:,:,1:-1,1:-1] - x_tt1[:,:,1:-1,1:-1]) +x_temp4[:,:,1:-1,1:-1]\
        #                              *(ref_speed1[:, :, 1:-1, 1:-1] ** 2) * (self.dt ** 2) / (self.dx ** 2)
        #     #x_temp3[:,:,0,:]=x_temp3[:,:,-1,:]=x_temp3[:,:,:,0]=x_temp3[:,:,:,-1]=0
        #     if step in self.effective_step:
        #         outputs4.append(x_temp3)
        #
        #     if step == (self.step-2):
        #         second_last_state_backward = internal_state_backward.copy()
        #
        # outputs4 = torch.cat(tuple(outputs4), dim=0)
        # outputs4 = torch.flip(outputs4, dims=[0])
        # outputs11 = torch.cat(tuple(outputs11), dim=0)
        # #outputs4 = torch.cat((outputs11[0:2, :, :, :], outputs4[1:, :, :, :]), dim=0)
        # outputs6.append(x_init2)
        #
        # for step in range(0, self.step - 1, 1):
        #     x_tt = outputs6[-2].clone()
        #     x_t = outputs6[-1].clone()
        #     # for x in range(10):
        #     #     for y in range(10):
        #     #         x_tt[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[step:step + 1, :, x, y]
        #     #         x_t[:, :, 6 * (x + 1), 6 * (y + 1)] = self.ref_sol2[step + 1:step + 2, :, x, y]
        #     x_temp3 = torch.cat((outputs11[step + 2:step + 3, :, :, :], outputs4[step + 2:step + 3, :, :, :]), dim=0)
        #     x_temp3 = torch.reshape(x_temp3, (1, 2 * 64 * 64))
        #     x_temp3 = torch.tanh(self.fc91(x_temp3))
        #     x_temp3 = torch.tanh(self.fc92(x_temp3))
        #     # x_temp3=torch.tanh(self.fc63(x_temp3))
        #     x_temp3 = torch.reshape(self.fc94(x_temp3), (1, 1, 64, 64))
        #     x_temp4 = x_temp3.clone()
        #     x_temp3[:,:,1:-1,1:-1] = (2 *  x_t[:,:,1:-1,1:-1] - x_tt[:,:,1:-1,1:-1]) + x_temp4[:,:,1:-1,1:-1]*\
        #                              (ref_speed1[:, :, 1:-1, 1:-1] ** 2) * (self.dt ** 2) / (self.dx ** 2)
        #     #x_temp3[:,:,0,:]=x_temp3[:,:,-1,:]=x_temp3[:,:,:,0]=x_temp3[:,:,:,-1]=0
        #
        #     if step in self.effective_step:
        #         outputs6.append(x_temp3)
        #
        #     # if step == (self.step - 2):
        #     #     second_last_state_backward = internal_state_backward.copy()

        outputs6 = torch.cat(tuple(outputs6), dim=0)
        #outputs62 = torch.cat(tuple(outputs62), dim=0)
        return outputs11, outputs1, outputs6, outputs62, second_last_state_forward, second_last_state_backward, ref_speed2


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
        self.ref_sol1 = torch.load('E:/Image_Dataset/True/temp10.pt').cuda()
        self.speed = torch.load('./c_speed.pt').cuda()
        self.speed = torch.unsqueeze((torch.unsqueeze(self.speed,dim=0)),dim=0)
        self.zero =  torch.zeros((1,1,64)).cuda()

    def random_points(self):
        points = torch.empty(0, 2, dtype=torch.int32)
        while points.shape[0] < 90:
            new_points = torch.randint(low=1, high=63, size=(90 - points.shape[0], 2))
            new_points_unique = torch.unique(new_points, dim=0)
            points = torch.cat([points, new_points_unique], dim=0)
        return points[:90]

    def get_ref_Loss(self):
        temp_res = self.ref_sol[:,:,:,:]
        #temp_res = torch.squeeze(temp_res,dim=1)
        return temp_res

    def get_ref_Loss1(self):
        temp_res = self.ref_sol1[:,:,:,:]
        #temp_res = torch.squeeze(temp_res,dim=1)
        return temp_res

    def get_local_Loss3(self, res):
        res1 = res.clone()
        local_res = torch.zeros((res.shape[0],1, 10, 10))
        for x in range(10):
            for y in range(10):
                local_res[:,:, x, y] = res1[:,:,6 * (x + 1), 6 * (y + 1)]

        return local_res

    def get_laplace_Loss(self, output):
        output1 = torch.squeeze(output, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        t_max = 0.01 * (int(output3.shape[0]) + 1)
        # dx = 0.01
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            #output3[:, 0, :] = output3[:, -1, :] = output3[:, :, 0] = output3[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            #output3[(n + 1) - 2, :, :] = (-output1[n+1,:, :] +2 * output1[n, :, :] - output1[n - 1,:, :] )/((dt**2)*(c[2:3,:]**2))
            output3[(n + 1) - 2, 1:-1, 1:-1] = (output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n, :-2,1:-1]) + \
                                               (output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n, 1:-1,:-2])

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
            output3[(n + 1) - 2, :, :] = (-output1[n+1,:, :] +2 * output1[n, :, :] - output1[n - 1,:, :] )/(dt**2)+ \
                                         (c**2)*((output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n, :-2,1:-1]) + (output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n, 1:-1,:-2]))

        return torch.unsqueeze(output3[:,:,:], dim=1)

    def get_residual_Loss(self, output, c):
        output1 = torch.squeeze(output, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        t_max = 0.01 * (int(output3.shape[0]) + 1)
        # dx = 0.01
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            #output3[:, 0, :] = output3[:, -1, :] = output3[:, :, 0] = output3[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            output3[(n + 1) - 2, :, :] = (-output1[n+1,:, :] +2 * output1[n, :, :] - output1[n - 1,:, :] )/((dt**2)*(c[:,:]**2))
            # output3[(n + 1) - 2, 1:-1, 1:-1] = (output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n, :-2, 1:-1]) + \
            #                                    (output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n, 1:-1,:-2])

        return torch.unsqueeze(output3[:, :, :], dim=1)

    def get_phy_Loss3(self, output,c,bsize,id):
        output1 = torch.squeeze(output.clone(), dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        t_max = 0.01*(int(output3.shape[0])+1)
        output1[1:2,:,:]=output1[0:1,:,:]
        r1, r2 = ((dt**2)*(c**2))/ (dx ** 2),((dt**2)*(c**2))/ (dy ** 2)
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

        return torch.unsqueeze(output1[2:,:,:], dim=1)

    def get_phy_Loss4(self, output, c, bsize, id):
        output1 = output.clone()
        output1 = torch.squeeze(output1, dim=1)
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        dx = dy = 1
        t_max = 0.01 * (int(output3.shape[0]) + 1)
        r1, r2 = ((dt ** 2) * (c ** 2)) / (dx ** 2), ((dt ** 2) * (c ** 2)) / (dy ** 2)
        for n in range(1, int(t_max / dt)):
            # 在内部节点上使用五点差分法计算新的波场
            output1[:, 0, :] = output1[:, -1, :] = output1[:, :, 0] = output1[:, :, -1] = 0
            output3[(n + 1) - 2, 1:-1, 1:-1] = (2 * output1[n, 1:-1, 1:-1] - output1[n + 1, 1:-1, 1:-1]) + \
                                               r1 * (
                                                       output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                           :-2, 1:-1]) + \
                                               r2 * (
                                                       output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                           1:-1, :-2])

        return torch.unsqueeze(output3[:, :, :], dim=1)


    def get_phy_Loss2(self, output, c,bsize,id):
        output1=output.clone()
        output1 = torch.squeeze(output1, dim=1)
        ref_local_sol = self.get_ref_Loss1().cuda()
        output3 = torch.zeros_like(output1[2:, :, :]).cuda()
        dt = 0.01
        dx = dy = 1
        t_max = 0.01*(int(output3.shape[0])+1)
        r1, r2 = ( (dt ** 2)*(c**2))/ (dx ** 2),((dt ** 2)*(c**2))/ (dy ** 2)
        for n in range(1, int(t_max / dt)):
            # 在边界处设置固定边界条件
            output1[:, 0, :] = output1[:, -1, :] = output1[:, :, 0] = output1[:, :, -1] = 0

            # 在内部节点上使用五点差分法计算新的波场
            output3[(n + 1)-2, 1:-1, 1:-1] =(2*output1[n,1:-1,1:-1]-output1[n-1,1:-1,1:-1])+r1 * (
                                                   output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                       :-2, 1:-1]) + \
                                           r2 * (
                                                   output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                       1:-1, :-2])

        return torch.unsqueeze(output3[:, :, :], dim=1)

    def get_pressure_release_Loss(self,output,epoch):
        mse_loss = nn.MSELoss(reduction='mean')
        output4 = torch.squeeze(output[2+epoch:3+epoch, :, :, :], dim=1)
        Loss_b = 0
        zero1 = torch.zeros((1,1,64)).cuda()
        zero2 = torch.zeros((1,64,1)).cuda()
        Loss_b += (mse_loss(output4[:, 0:1, :], zero1) + mse_loss(output4[:, -1:, :],zero1) + \
                        mse_loss(output4[:,  :, 0:1], zero2) + mse_loss(output4[:, :, -1:],zero2))
        return Loss_b/4

    def get_rigid_boundary_Loss(self,output,epoch):
        mse_loss = nn.MSELoss(reduction='mean')
        output1 = torch.squeeze(output[2+epoch:3+epoch,:,:,:], dim=1)
        Loss_b=0
        Loss_b+=mse_loss(output1[:,0,1:-1]-output1[:,1,1:-1],torch.zeros_like(output1[:,0,1:-1]))
        Loss_b+=mse_loss(output1[:,-1,1:-1]-output1[:,-2,1:-1],torch.zeros_like(output1[:,0,1:-1]))
        Loss_b+=mse_loss(output1[:,1:-1,0] - output1[:,1:-1,1], torch.zeros_like(output1[:,1:-1,0]))
        Loss_b+=mse_loss(output1[:,1:-1,-1] - output1[:,1:-1,-2], torch.zeros_like(output1[:,1:-1,0]))
        return Loss_b/4

    def get_nonreflect_boundary_Loss(self,output,epoch,c):
        mse_loss = nn.MSELoss(reduction='mean')
        output1 = torch.squeeze(output[:,:,:,:], dim=1)
        Loss_b=0
        dt=0.01
        dx=1
        n=epoch+2
        Loss_b1= mse_loss(output1[n - 1, 0, 1:-1]-0.01 * c[0, 1:-1] * (
                        output1[n - 1, 0, 1:-1] - output1[n - 1, 1, 1:-1]), output1[n,0, 1:-1])

        Loss_b2= mse_loss(output1[n - 1, -1, 1:-1] + 0.01 * c[-1, 1:-1] * (
                        output1[n - 1, -2, 1:-1] - output1[n - 1, -1, 1:-1]),output1[n, -1, 1:-1])

        Loss_b3= mse_loss(output1[n - 1, 1:-1, -1] - 0.01 * c[1:-1, -1] * (
                        output1[n - 1, 1:-1, -1] - output1[n - 1, 1:-1, -2]), output1[n,1:-1, -1])

        Loss_b4= mse_loss(output1[n - 1, 1:-1, 0] + 0.01 * c[1:-1, 0] * (
                        output1[n - 1, 1:-1, 1] - output1[n - 1, 1:-1, 0]),output1[n,  1:-1, 0])
        Loss_b+=(Loss_b1+Loss_b2+Loss_b3+Loss_b4)/4

        return Loss_b



def compute_loss(batchloss,output1, output9,output4,output8,loss_func, id, ref_speed,ref_source2,chen0,bsize,ntb,epoch,omega1,omega2,omega3,omega4,omega11,omega21,omega31,omega41):
    ''' calculate the phycis loss '''
    # print(c1)
    mse_loss = nn.MSELoss(reduction='mean')
    mse_loss1 = nn.MSELoss(reduction='sum')


    #ref_speed2 = ref_speed2 * torch.ones_like(ref_speed).cuda()

    #
    ref_local_sol = loss_func.get_ref_Loss1().cuda()
    ref_res_local1 = loss_func.get_local_Loss3(ref_local_sol)

    #epoch = int((ref_res_local1.shape[0]-2)/2)
    output11 = loss_func.get_phy_Loss3(output4, ref_speed, bsize, id)
    #output13 = loss_func.get_phy_Loss2(output4, ref_speed, bsize, id)

    p_res2=0
    p_local=0
    p_b1 = 0
    p_b2 = 0
    w1=[]
    w2=[]
    p_init = 0#mse_loss(output21[1:2, :, :, :], output4[1:2, :, :, :])
    # epoch = int(output13.shape[0]/2)

    # p_local37 = loss_func.get_local_Loss3(output13)
    p_local36 = loss_func.get_local_Loss3(output4)
    w1 = []

    for i in range(output11.shape[0]):
        p_local += mse_loss(p_local36[i+2:i+3, :,3, :],
                            ref_res_local1[(bsize - 2) * id + i + 2:(bsize - 2) * id + i + 3, :,3, :])
        p_local += mse_loss(p_local36[i+2:i+3, :, :, 3],
                            ref_res_local1[(bsize - 2) * id + i + 2:(bsize - 2) * id + i + 3, :, :, 3])
        p_res2 += mse_loss(output11[i:i+1, :, :, :], output4[i + 2:i + 3, :, :, :]).cuda()

    # for i in range(epoch):
    #     p_local += mse_loss(p_local37[i:i + 1, :, :, :],
    #                         ref_res_local1[(bsize - 2) * id + i + 2:(bsize - 2) * id + i + 3, :, :, :])
    #     p_res2 += mse_loss(output13[i:i + 1, :, :, :], output4[i + 2:i + 3, :, :, :]).cuda()
    #
    #     p_res2 += loss_func.get_pressure_release_Loss(output4, i)

    # for i in range(epoch):
    #     p_local += mse_loss(p_local37[i:i +1, :, :, :],
    #                         ref_res_local1[(bsize - 2) * id + i + 2:(bsize - 2) * id + i + 3, :, :, :])
    #     p_res2 += mse_loss(output13[i:i + 1, :, :, :], output4[i + 2:i + 3, :, :, :]).cuda()
        #p_res2 += loss_func.get_pressure_release_Loss(output4, i)

    # for i in range(output11.shape[0]):
    #     # print(mse_loss(output11[i:i + 1, :, :, :], output4[i+2:i + 3, :, :, :]))
    #     p_res2 += (mse_loss(output11[i:i + 1, :, :, :], output4[i+2:i + 3, :, :, :]).cuda())

        #p_b1 += w1[i] * loss_func.get_pressure_release_Loss(output4, i)
    p_local2=mse_loss(ref_local_sol, output4)

    p_1=p_res2
    p_2=p_b1
    p_3=p_init
    p_4=p_local
    loss = p_1+p_local+batchloss
    loss2 = p_local2
    return loss, loss2, p_1, p_2, p_3, p_4

def train(model, input, ref_res, forward_state, backward_state, n_iters, time_batch_size, learning_rate,
          dt, dx, save_path, pre_model_save_path, num_time_batch):
    train_loss_list = []
    source_loc_list_x = []
    source_loc_list_y = []
    ref_speed_arr=[]
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
    #ref_speed = 65*torch.ones((1,1,64,64)).cuda()
    c=1/2
    omega1=c
    omega2=c
    omega3=c
    omega4=c

    omega11=1
    omega21=1
    omega31=1
    omega41=1

    last_loss1=0
    last_loss2=0
    last_loss3=0
    last_loss4=0

    chen0 = torch.ones((1, 1,64,64)).cuda()

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
        start = time.time()
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_loss2 = 0.0
        batch_loss3 = 0.0
        batch_loss4 = 0.0
        batch_loss5 = 0.0
        batch_loss6 = 0.0

        ref_speed = 65
        ref_speed1 = 65


        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                hidden_state1 = forward_state
                hidden_state2 = backward_state
                x_tt = input[0:1,:,:,:]
                x_t = input[1:2, :, :, :]
                x_tt2 = input[0:1, :, :, :]
                x_t2 = input[1:2, :, :, :]
            else:
                hidden_state1 = state_detached1
                hidden_state2 = state_detached2
                x_tt = prev_output2[-2:-1, :, :, :]
                x_t = prev_output2[-1:, :, :, :]
                x_tt2 = prev_output2[-2:-1, :, :, :]
                x_t2 = prev_output2[-1:, :, :, :]
                # x_tt = input[time_batch_size*time_batch_id:time_batch_size*time_batch_id+1, :, :, :].detach()  # second last output
                # x_t = input[time_batch_size*time_batch_id+1:time_batch_size*time_batch_id+2, :, :, :].detach()  # second last output
                # x_tt2 = input[time_batch_size*time_batch_id:time_batch_size*time_batch_id+1, :, :, :].detach()
                # x_t2 = input[time_batch_size*time_batch_id+1:time_batch_size*time_batch_id+2, :, :, :].detach()

            # output is a list
            output1,output2,output3,output4,second_last_state_forward,second_last_state_backward, ref_speed= model(time_batch_id,time_batch_size,hidden_state1,hidden_state2,x_t,x_tt,ref_speed,ref_speed1,chen0,x_t2,x_tt2)
            # output4 = torch.cat((x_tt,x_t,output4), dim=0)
            #output3 = torch.cat((x_tt, x_t, output3), dim=0)


            # get loss
        # with torch.autograd.set_detect_anomaly(True):
            loss, loss2, loss3, loss4, loss5, loss6= compute_loss(batch_loss,output1,output2,output3,output4,loss_func,time_batch_id,
                                                                         ref_speed,ref_speed,chen0,time_batch_size,num_time_batch,epoch,omega1,omega2,omega3,omega4,omega11,omega21,omega31,omega41)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            batch_loss2 += loss2.item()
            batch_loss3 += loss3.item()
            #batch_loss4 += loss4.item()
            #batch_loss5 += loss5.item()
            batch_loss6 += loss6.item()
            # update the state and output for next batch
            prev_output1 = output4
            prev_output2 = output3
            state_detached1 = []
            state_detached2 = []
            for i in range(len(second_last_state_forward)):
                (h, c) = second_last_state_forward[i]
                state_detached1.append((h.detach(), c.detach()))
            for i in range(len(second_last_state_backward)):
                (h, c) = second_last_state_backward[i]
                state_detached2.append((h.detach(), c.detach()))

            # if epoch % 1000 == 0 and epoch != 0:
            #     torch.save(prev_output2, 'E:/Image_Dataset/True/res/tensor_LSTMF_speed3_' + str(epoch)+'_'+str(time_batch_id)+ '.pt')
        # if epoch==0:
        #     max_order=max(batch_loss3-last_loss1,batch_loss6-last_loss4)
        #     omega11 = 2**(math.log2(max_order)-math.log2(batch_loss3-last_loss1))
        #     #omega21 = 2**(math.log2(max_order)-math.log2(batch_loss4))
        #     #omega31 = 2 ** (math.log2(max_order) - math.log2(batch_loss5))
        #     omega41 = 2 ** (math.log2(max_order) - math.log2(batch_loss6-last_loss4))
        #     print(omega11,omega41)

        if epoch>=1 and epoch%10==0:
            weight1 = batch_loss3 / (last_loss1)
            #weight2 = batch_loss4 / (last_loss2)
            #weight3 = batch_loss5 / (last_loss3)
            weight4 = batch_loss6 / (last_loss4)
            all = (weight1 + weight4)
            omega1 = weight1 / all
            #omega2 = weight3 / all

            #omega3 = weight3 / all
            omega4 = weight4 / all
            print("weight1: %.5f weight3: %.5f"
                  %(omega1,omega4))
            # omega = 10**(int(math.log10(last_loss1))-int(math.log10(last_loss2)))
            # omega11 = 10**(int(math.log10(last_loss1))-int(math.log10(last_loss3)))
        if (epoch >= 1 and epoch % 10 == 0) or epoch==1:
            last_loss1 = batch_loss3
            #last_loss2 = batch_loss4
            #last_loss3 = batch_loss5
            last_loss4 = batch_loss6


        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.5f speed: %.5f loss_true: %.5f loss_res: %.5f loss_boundary: %.5f loss_init: %.5f loss_local: %.5f'  % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0),
                                            batch_loss,ref_speed, batch_loss2, batch_loss3, batch_loss4, batch_loss5, batch_loss6))

        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss

        cmap = cm.get_cmap('jet')
        cmap1 = cm.get_cmap('jet')

        # if epoch%5000==0 and epoch!=0:
        #     #speed = torch.squeeze(torch.squeeze(ref_speed, dim=0), dim=0)
        #     np.save('E:/Papertest/Inverse/SoundSpeed/case1/speed_' + str(epoch),ref_speed_arr)
        end = time.time()
        # print('The epoch time is: ', (end - start))
        # if epoch%100==0 and epoch!=0:
        #     torch.save(ref_speed, 'E:/Papertest/Inverse/MultiObstacle/case4/case4_'+str(epoch)+'.pt')
        #     plt.imshow(ref_speed.detach().cpu().numpy().squeeze(), cmap=cmap)
        #     plt.colorbar()
        #     plt.show()

    return train_loss_list,source_loc_list_x,source_loc_list_y


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
    input2 = torch.zeros((input_tensor.shape[0],1,64,64)).cuda()
    input2[:,:,:,:]=input_tensor[:,:,:,:]

    # set initial states for convlstm
    num_convlstm = 1
    (h1, c0) = (torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64,64))
    backward_state = []
    forward_state=[]
    for i in range(num_convlstm):
        forward_state.append((h1, c0))
        backward_state.append((h1, c0))

    # grid parameters
    time_steps = input_tensor.shape[0]

    ################# build the model #####################-``
    time_batch_size =  input_tensor.shape[0]
    steps = time_batch_size - 1
    effective_step = list(range(0, steps))
    num_time_batch = int((time_steps-2) / (time_batch_size-2))
    n_iters_adam =10001
    lr_adam = 1e-3  # 1e-3
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
    train_loss,source_loc_x,source_loc_y = train(model, input2, solve, forward_state,backward_state, n_iters_adam, time_batch_size,
                       lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch)
    end = time.time()

    print('The training time is: ', (end - start))