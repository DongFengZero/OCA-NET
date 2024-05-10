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
        # 在内部节点上使用五点差分法计算新的波场
        output1[(n + 1), 1:-1, 1:-1] = 2 * output1[n, 1:-1, 1:-1] - output1[n - 1, 1:-1, 1:-1] + \
                                           r1[1:-1, 1:-1] * (
                                                       output1[n, 2:, 1:-1] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                           :-2, 1:-1]) + \
                                           r2[1:-1, 1:-1] * (
                                                       output1[n, 1:-1, 2:] - 2 * output1[n, 1:-1, 1:-1] + output1[n,
                                                                                                           1:-1, :-2])


    return torch.unsqueeze(output1, dim=1)

Lx = Ly = 64 # Length of the 2D domain

dx = 1  # 空间步长为0.01米
dy = 1

c = 45 * torch.ones((64, 64)).cuda()  # 生成一个波速为45的张量（可根据测试需要调整）

# 以下为多障碍物测试时生成代码，可根据需要调整
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

#设置时间步
size=34

#设置单点（单点声速预测）或多点声源（探测多障碍物）
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

#保存文件
torch.save(output_s,'E:/Image_Dataset/True/temp10.pt')

#设置预览图
cmap = cm.get_cmap('jet')
plt.imshow(c.detach().cpu().numpy().squeeze(), cmap=cmap)
plt.colorbar()
plt.show()
