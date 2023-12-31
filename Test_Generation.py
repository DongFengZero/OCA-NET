import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_phy_Loss(output1, c):
    print(c)
    t_max = 6.01
    dt = 0.01
    dx = 1
    dy = 1
    r1, r2 = c ** 2 * dt ** 2 / dx ** 2, c ** 2 * dt ** 2 / dy ** 2
    for n in range(1, int(t_max / dt)):
        # 在边界处设置固定边界条件
        output1[:, 0, :] = output1[:, -1, :] = output1[:, :, 0] = output1[:, :, -1] = 0
        # output1[n, 0, 1:-1] = output1[n - 1, 0, 1:-1]-0.01 * c[0, 1:-1] * (
        #             output1[n - 1, 0, 1:-1] - output1[n - 1, 1, 1:-1])
        # output1[n, -1, 1:-1] = output1[n - 1, -1, 1:-1] + 0.01 * c[-1, 1:-1] * (
        #             output1[n - 1, -2, 1:-1] - output1[n - 1, -1, 1:-1])
        # output1[n, 1:-1, -1] = output1[n - 1, 1:-1, -1] - 0.01 * c[1:-1, -1] * (
        #             output1[n - 1, 1:-1, -1] - output1[n - 1, 1:-1, -2])
        # output1[n, 1:-1, 0] = output1[n - 1, 1:-1, 0] + 0.01 * c[1:-1, 0] * (
        #             output1[n - 1, 1:-1, 1] - output1[n - 1, 1:-1, 0])
        # 在内部节点上使用五点差分法计算新的波场
        output1[(n + 1), 1:-1, 1:-1] = 2 * output1[n, 1:-1, 1:-1] - output1[n - 1, 1:-1, 1:-1] + \
                                           r1[1:-1, 1:-1] * (
                                                       output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                           :-2, 1:-1]) + \
                                           r2[1:-1, 1:-1] * (
                                                       output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                           1:-1, :-2])
        # output1[n + 1, 0, 1:-1] = output1[n+1, 1, 1:-1]
        # output1[n + 1, -1, 1:-1] = output1[n+1, -2, 1:-1]
        # output1[n + 1, 1:-1, 0] = output1[n+1, 1:-1, 1]
        # output1[n + 1, 1:-1, -1] = output1[n+1, 1:-1, -2]

    return torch.unsqueeze(output1, dim=1)
# def solve_wave_equation(t_max, c):
#     # 设置模拟参数
#     # t_max = 0.5  # 模拟时间长为0.5秒
#     Lx = Ly = 1.28  # Length of the 2D domain
#     dt = 0.01  # 时间分辨率为0.0002秒
#     dx = 0.01  # 空间步长为0.01米
#     dy = 0.01
#
#     # 从波速张量中获取网格尺寸
#     nx, ny = 128, 128
#
#     # 计算一些常量
#     #c = np.ones((128, 128))  # 生成一个波速为1的张量
#
#     r1, r2 = c ** 2 * dt ** 2 / dx ** 2, c ** 2 * dt ** 2 / dy ** 2
#
#     # 初始化时刻t=0和t=dt的波场
#     u = np.zeros((nx, ny, 2))
#     x = np.arange(0, Lx, dx)
#     y = np.arange(0, Ly, dy)
#     X, Y = np.meshgrid(x, y)
#     #u[:, :, 0] = 1e2*np.exp(-((X - 640) ** 2 + (Y - 640) ** 2)/1e6)
#     u[:, :,0] = 1e3 * np.exp(-((X - 0.64) ** 2 + (Y - 0.64) ** 2) / 0.01)
#     u[:, :, 1] = u[:, :, 0]  # t=dt时的波场
#     u = torch.from_numpy(u).cuda()
#     r1 = torch.from_numpy(r1).cuda()
#     r2 = torch.from_numpy(r2).cuda()
#
#     # 迭代计算波场
#     for n in range(1, int(t_max / dt)):
#         # 在边界处设置固定边界条件
#         u[0, :, n % 2] = u[-1, :, n % 2] = u[:, 0, n % 2] = u[:, -1, n % 2] = 0
#
#         # 在内部节点上使用五点差分法计算新的波场
#         u[1:-1, 1:-1, (n + 1) % 2] = 2 * u[1:-1, 1:-1, n % 2] - u[1:-1, 1:-1, (n - 1) % 2] + \
#                                     r1[1:-1,1:-1] * (u[2:, 1:-1, n % 2] - 2 * u[1:-1, 1:-1, n % 2] + u[:-2, 1:-1, n % 2]) + \
#                                     r2[1:-1,1:-1] * (u[1:-1, 2:, n % 2] - 2 * u[1:-1, 1:-1, n % 2] + u[1:-1, :-2, n % 2])
#
#     return u[:, :, -1]
#
# def get_phy_Loss2(output1):
#     output3 = output1[2:, :, :]
#     t_max = 0.16
#     dt = 0.01
#
#     for n in range(1, int(t_max / dt) - 2):
#         # 在边界处设置固定边界条件
#         # output1[:, 0, :] = output1[:, -1, :] = output1[:, :, 0] = output1[:, :, -1] = 0
#
#         # 在内部节点上使用五点差分法计算新的波场
#         output3[(n + 1) - 2, 1:-1, 1:-1] = 2 * output1[n,1:-1, 1:-1] - output1[n-1,1:-1, 1:-1]
#
#         return output3
#
#
Lx = Ly = 64 # Length of the 2D domain

dx = 1  # 空间步长为0.01米
dy = 1
#
# # 从波速张量中获取网格尺寸
#
# speed = torch.ones((1,2),dtype=torch.float32).cuda()
#
# output[0:2,:,:]=u[0:2,:,:]
c = 45 * torch.ones((64, 64)).cuda()  # 生成一个波速为1的张量
# x2 = torch.arange(0, Lx, dx).cuda()
# y2 = torch.arange(0, Ly, dy).cuda()
# X2, Y2 = torch.meshgrid(x2, y2, indexing='ij')
# l=5
# s_x=15
# s_y=15
# s_u = (2 * l) ** 2 / ((X2 - s_x) ** 2 + (Y2 - s_y) ** 2 + 1)
# c=c-s_u
# c[23:27,13:19]=0
# c[48:52,12:17]=0
# c[36:40,36:40]=0
# c[52:56,52:56]=0
size=34
# output_s = torch.zeros((size*5,1, 64, 64)).cuda()
x1=[30]
y1=[30]
output_s = torch.zeros((size,1, 64, 64)).cuda()
u = np.zeros((2, 64, 64))
output = torch.zeros((602, 64, 64)).cuda()
for i in range(len(x1)):
    x = np.arange(0, Lx, dx)
    y = np.arange(0, Ly, dy)
    X, Y = np.meshgrid(x, y)
    u[0,:, :] +=1e3*np.exp(-((X - x1[i]) ** 2 + (Y - y1[i]) ** 2)/100)
    u[1,:, :] +=1e3*np.exp(-((X - x1[i]) ** 2 + (Y - y1[i]) ** 2)/100)
u = torch.from_numpy(u)
output[0:2,:,:] = u[0:2,:,:]
output=get_phy_Loss(output,c)
output_s[:size,:,:,:]=output[:size,:,:,:]

#torch.save(c,'E:/Papertest/Inverse/MultiObstacle/case5/c_speed.pt')
# for t in range(0, 640, 1):
#     temp_u = solve_wave_equation(t * 0.01, c).squeeze()
#     output[t+1, :, :] = temp_u
#
# output=torch.load('E:/Image_Dataset/True/refsol6.pt')
# # output=torch.unsqueeze(output,dim=1)
# #output3 = get_phy_Loss2(output)
# local_res = torch.zeros((output.shape[0], 1, 16, 16))
# for x in range(16):
#     for y in range(16):
#         local_res[:,:, x, y] = output[:,:, 7 * (x + 1), 7 * (y + 1)]
#
# torch.save(local_res,'E:/Image_Dataset/True/refcolsol6.pt')
#output3 = torch.from_numpy(output3).cuda()
#output = torch.from_numpy(output).cuda()
# output1=torch.zeros((4,34,1,64,64))
#output1[0:2,:,:,:]=u[0:2,:,:,:]
# for t in range(0,output1.shape[0],1):
#     output1[t,:,:,:]=output_s[:,t,:,:,:]
torch.save(output_s,'E:/Image_Dataset/True/temp10.pt')
#torch.save(output_s,'E:/Papertest/Inverse/MultiObstacle/case5/case5.pt')
#
# local_res = torch.zeros((output1.shape[0], 1, 14, 14))
# for x in range(14):
#     for y in range(14):
#         local_res[:,:, x, y] = output1[:,:, 4 * (x + 1), 4 * (y + 1)]
# torch.save(local_res,'E:/Image_Dataset/True/temp99.pt')
# print(local_res.shape)
# local_res = torch.zeros((output1.shape[0], 1, 14, 14))
# for x in range(14):
#     for y in range(14):
#         local_res[:,:, x, y] = output1[:,:, 4 * (x + 1), 4 * (y + 1)]
# torch.save(local_res,'E:/Image_Dataset/True/temp999.pt')
# print(local_res.shape)
#
# print(output.shape)
cmap = cm.get_cmap('jet')
plt.imshow(c.detach().cpu().numpy().squeeze(), cmap=cmap)
plt.colorbar()
plt.show()
# for i in range(0,6,2):
#     plt.imshow(output_s[i].detach().cpu().numpy().squeeze(), cmap=cmap)
#     plt.colorbar()
#     plt.show()
