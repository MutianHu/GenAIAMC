
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


import torch.nn as nn


# 判别器网络
# class Discriminator(nn.Module):
#     def __init__(self, input_dim):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        # print("1")
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # print("2")
        noise = torch.randn_like(x_0) * 0.25
        # print("3")
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        # print("4")
        out = self.model(x_t, t, labels)
        # print("5")

        print(x_0.shape)
        for i in range(10):
            # print(i)
            # print(labels[i])
            x_axis_data = np.arange(0, 128, 1)  # x
            y_axis_data_I = np.squeeze(x_0[i:i + 1, :, 0:1, :])
            y_axis_data_Q = np.squeeze(x_0[i:i + 1, :, 1:2, :])
            plt.figure(figsize=(10, 8))

            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5, linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5, linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

            # plt.legend()  # 显示上面的label
            plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
            plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
            plt.xticks(fontsize=30, fontname='Times New Roman')
            plt.yticks(fontsize=30, fontname='Times New Roman')

            # plt.ylim(-1,1)#仅设置y轴坐标范围
            # plt.show()
            plt.savefig('./SampledSignals2/x_0_{}_{}.jpg'.format(i, labels[i]), bbox_inches='tight')
            # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # plt.show()
            plt.close()


        for i in range(10):
            # print(i)
            # print(labels[i])
            x_axis_data = np.arange(0, 128, 1)  # x
            y_axis_data_I = np.squeeze(noise[i:i + 1, :, 0:1, :])
            y_axis_data_Q = np.squeeze(noise[i:i + 1, :, 1:2, :])
            plt.figure(figsize=(10, 8))

            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5, linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5, linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

            # plt.legend()  # 显示上面的label
            plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
            plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
            plt.xticks(fontsize=30, fontname='Times New Roman')
            plt.yticks(fontsize=30, fontname='Times New Roman')

            # plt.ylim(-1,1)#仅设置y轴坐标范围
            # plt.show()
            plt.savefig('./SampledSignals2/noise_{}_{}.jpg'.format(i, labels[i]), bbox_inches='tight')
            # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # plt.show()
            plt.close()
        #
        #     # plt.figure(figsize=(20, 20), dpi=100)
        #     plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/Constellation_x_0_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()
        #
        for i in range(10):
            # print(i)
            # print(labels[i])
            x_axis_data = np.arange(0, 128, 1)  # x
            y_axis_data_I = np.squeeze(x_t[i:i + 1, :, 0:1, :])
            y_axis_data_Q = np.squeeze(x_t[i:i + 1, :, 1:2, :])
            plt.figure(figsize=(10, 8))

            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5,
                     linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5,
                     linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
            plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
            plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
            plt.xticks(fontsize=30, fontname='Times New Roman')
            plt.yticks(fontsize=30, fontname='Times New Roman')

            # plt.legend()  # 显示上面的label
            # plt.xlabel('time')  # x_label
            # plt.ylabel('number')  # y_label

            # plt.ylim(-1,1)#仅设置y轴坐标范围
            # plt.show()
            plt.savefig('./SampledSignals2/SampledGuidenceSignal_{}_{}.jpg'.format(i, labels[i]), bbox_inches='tight')
            # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # plt.show()
            plt.close()

        for i in range(10):
            # print(i)
            # print(labels[i])
            out1 = Tensor.cpu(out)
            x_axis_data = np.arange(0, 128, 1)  # x
            y_axis_data_I = np.squeeze(out1[i:i + 1, :, 0:1, :])
            y_axis_data_Q = np.squeeze(out1[i:i + 1, :, 1:2, :])
            plt.figure(figsize=(10, 8))

            plt.plot(x_axis_data, y_axis_data_I.detach().numpy(), 'b-', alpha=0.5,
                     linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            plt.plot(x_axis_data, y_axis_data_Q.detach().numpy(), 'r-', alpha=0.5,
                     linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
            plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
            plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
            plt.xticks(fontsize=30, fontname='Times New Roman')
            plt.yticks(fontsize=30, fontname='Times New Roman')

            # plt.legend()  # 显示上面的label
            # plt.xlabel('time')  # x_label
            # plt.ylabel('number')  # y_label

            # plt.ylim(-1,1)#仅设置y轴坐标范围
            # plt.show()
            plt.savefig('./SampledSignals2/Pernoise_{}_{}.jpg'.format(i, labels[i]), bbox_inches='tight')
            # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # plt.show()
            plt.close()
        # print("End", e)
        #
        #
        #
        #     # plt.figure(figsize=(20, 20), dpi=100)
        #     plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/Constellation_SampledGuidenceSignal_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()
        #
        #     x_axis_data = np.arange(0, 128, 1)  # x
        #     y_axis_data_I = np.squeeze(noise[i:i + 1, :, 0:1, :])
        #     y_axis_data_Q = np.squeeze(noise[i:i + 1, :, 1:2, :])
        #     plt.figure(figsize=(10, 8))
        #
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5,
        #              linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5,
        #              linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
        #     plt.xlabel('Time', fontsize=20, fontname='Times New Roman')  # x_label
        #     plt.ylabel('Amplitude', fontsize=20, fontname='Times New Roman')  # y_label
        #     plt.xticks(fontsize=20, fontname='Times New Roman')
        #     plt.yticks(fontsize=20, fontname='Times New Roman')
        #
        #     # plt.legend()  # 显示上面的label
        #     # plt.xlabel('time')  # x_label
        #     # plt.ylabel('number')  # y_label
        #
        #     # plt.ylim(-1,1)#仅设置y轴坐标范围
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/SampledGuidenceNoise_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()
        #
        #     # plt.figure(figsize=(20, 20), dpi=100)
        #     plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/Constellation_SampledGuidenceNoise_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()




            # y_axis_data_I = np.squeeze(x_t[i:i + 1, :, 0:1, :])
            # y_axis_data_Q = np.squeeze(x_t[i:i + 1, :, 1:2, :])
            # plt.figure(figsize=(10, 10), dpi=100)
            # plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
            # # plt.show()
            # plt.savefig('./SampledSignals/Constellation_{}_{}.jpg'.format(i, labels[i]))
            # # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # # plt.show()
            # plt.close()




        # kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # # # # log_target = F.log_softmax(torch.rand(3, 5), dim=1)
        # # # output = kl_loss(input, log_target)
        # # loss1 = F.mse_loss(out, noise, reduction='none')
        # # loss2 = kl_loss(out, noise)
        # loss = kl_loss(out, noise)
        # # print(loss1)
        # # print(loss2)
        # loss = (loss1 + loss2)/2
        # 转换为 MxM 的二维星座矩阵
        # print(out.shape)
        # b = out.shape[0]

        # print(constellation_matrix_input)
        # print(constellation_matrix_out)
        loss = F.mse_loss(out, noise, reduction='none') #/ b ** 2.
        # print("6")
        # print(loss.shape)
        # print(loss2.shape)

        # loss2 = F.kl_div(out, noise, reduction='batchmean')
        # print(loss2)
        # loss = loss1 + loss2

        # # 提取 sqrt_alphas_bar 和 sqrt_one_minus_alphas_bar 的值
        # sqrt_alphas_bar = extract(self.sqrt_alphas_bar, t, x_t.shape)
        # sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        # # #
        # # # # 根据公式还原 x_0
        # p_x_0 = (x_t - sqrt_one_minus_alphas_bar * out) / sqrt_alphas_bar
        # for i in range(10):
        #     # print(i)
        #     # print(labels[i])
        #     x_axis_data = np.arange(0, 128, 1)  # x
        #     y_axis_data_I = np.squeeze(p_x_0[i:i + 1, :, 0:1, :])
        #     y_axis_data_Q = np.squeeze(p_x_0[i:i + 1, :, 1:2, :])
        #     plt.figure(figsize=(10, 8))
        #
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I).detach().numpy(), 'b-', alpha=0.5, linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q).detach().numpy(), 'r-', alpha=0.5, linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
        #     plt.xlabel('Time', fontsize=20, fontname='Times New Roman')  # x_label
        #     plt.ylabel('Amplitude', fontsize=20, fontname='Times New Roman')  # y_label
        #     plt.xticks(fontsize=20, fontname='Times New Roman')
        #     plt.yticks(fontsize=20, fontname='Times New Roman')
        #
        #     # plt.legend()  # 显示上面的label
        #     # plt.xlabel('time')  # x_label
        #     # plt.ylabel('number')  # y_label
        #
        #     # plt.ylim(-1,1)#仅设置y轴坐标范围
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/p_x_0_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()
        #
        #     # plt.figure(figsize=(20, 20), dpi=100)
        #     plt.scatter(Tensor.cpu(y_axis_data_I).detach().numpy(), Tensor.cpu(y_axis_data_Q).detach().numpy())
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/Constellation_p_x_0_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()
        #
        # for i in range(10):
        #     # print(i)
        #     # print(labels[i])
        #     x_axis_data = np.arange(0, 128, 1)  # x
        #     y_axis_data_I = np.squeeze(p_x_0[i:i + 1, :, 0:1, :])
        #     y_axis_data_Q = np.squeeze(p_x_0[i:i + 1, :, 1:2, :])
        #     plt.figure(figsize=(10, 8))
        #
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5,
        #              linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5,
        #              linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
        #     plt.xlabel('Time', fontsize=20, fontname='Times New Roman')  # x_label
        #     plt.ylabel('Amplitude', fontsize=20, fontname='Times New Roman')  # y_label
        #     plt.xticks(fontsize=20, fontname='Times New Roman')
        #     plt.yticks(fontsize=20, fontname='Times New Roman')
        #
        #     # plt.legend()  # 显示上面的label
        #     # plt.xlabel('time')  # x_label
        #     # plt.ylabel('number')  # y_label
        #
        #     # plt.ylim(-1,1)#仅设置y轴坐标范围
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/p_x_0_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()
        #
        #     # plt.figure(figsize=(20, 20), dpi=100)
        #     plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
        #     # plt.show()
        #     plt.savefig('./SampledSignals2/Constellation_p_x_0_{}_{}.jpg'.format(i, labels[i]))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()

        # # loss2 = F.mse_loss(p_x_0, x_0, reduction='none')
        # # #添加还原星座图loss
        # constellation_matrix_out = iq_to_constellation_matrix_torch_batch(p_x_0, 64)
        # # # print(constellation_matrix_out.shape)
        # constellation_matrix_input = iq_to_constellation_matrix_torch_batch(x_0, 64)
        # # print(constellation_matrix_input.shape)
        # loss2 = F.mse_loss(constellation_matrix_out, constellation_matrix_input, reduction='none').sum() / b ** 2.
        # #
        # loss_total = loss + loss2
        # print(loss)
        return loss.sum()


import torch


def iq_to_constellation_matrix_torch_batch(iq_data, M):
    """
    将形状为 [B, 1, 2, N] 的批量 IQ 信号数据转为 [B, M, M] 的二维星座矩阵。

    参数:
        iq_data: torch.Tensor, 形状为 [B, 1, 2, N]，包含批量 I 和 Q 信号分量
        M: int, 目标矩阵的大小 (MxM)

    返回:
        constellation_matrices: torch.Tensor, 形状为 [B, M, M] 的二维星座矩阵
    """
    B = iq_data.size(0)  # 批次大小
    N = iq_data.size(-1)  # 数据点数量

    # 提取 I 和 Q 分量
    I_data = iq_data[:, 0, 0, :]  # [B, N]
    Q_data = iq_data[:, 0, 1, :]  # [B, N]

    # 获取 I 和 Q 的最小和最大值，确定每批次的坐标范围
    I_min, I_max = I_data.min(dim=-1, keepdim=True)[0], I_data.max(dim=-1, keepdim=True)[0]
    Q_min, Q_max = Q_data.min(dim=-1, keepdim=True)[0], Q_data.max(dim=-1, keepdim=True)[0]

    # 归一化到 [0, M-1] 的网格索引
    I_indices = ((I_data - I_min) / (I_max - I_min) * (M - 1)).floor().long()  # [B, N]
    Q_indices = ((Q_data - Q_min) / (Q_max - Q_min) * (M - 1)).floor().long()  # [B, N]

    # 映射到矩阵网格（Q 分量从下到上增加，因此需要翻转 Q 轴）
    Q_indices = M - 1 - Q_indices  # 翻转 Q 轴索引，保持星座图直观
    indices = torch.stack((Q_indices, I_indices), dim=-1)  # [B, N, 2]

    # 初始化批量的 MxM 二维星座矩阵
    constellation_matrices = torch.zeros((B, M, M), dtype=torch.float32, device=iq_data.device)

    # 填充矩阵
    for b in range(B):
        constellation_matrices[b].index_put_(tuple(indices[b].t()),
                                             torch.ones(N, dtype=torch.float32, device=iq_data.device), accumulate=True)

    return constellation_matrices


# # 示例输入
# B = 100  # 批次大小
# N = 128  # 单个样本的数据点数量
# M = 16  # 星座矩阵大小
#
# # 生成示例 IQ 数据 [B, 1, 2, N]
# torch.manual_seed(42)
# I = torch.rand(B, 1, N) * 2 - 1  # I 分量在 [-1, 1]
# Q = torch.rand(B, 1, N) * 2 - 1  # Q 分量在 [-1, 1]
# iq_data = torch.cat((I, Q), dim=1)  # [B, 1, 2, N]
#
# # 转换为 [B, M, M] 的二维星座矩阵
# constellation_matrices = iq_to_constellation_matrix_torch_batch(iq_data, M)
#
# # 打印结果
# print("星座矩阵形状:", constellation_matrices.shape)
# print("示例矩阵:\n", constellation_matrices[0])  # 打印第一个样本的星座矩阵


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # print(x_t.shape)
        # print(t.shape)
        # print(eps.shape)
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # print(x_t.shape)
        # print(t.shape)
        # print(labels.shape)
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        # print('eps.shape')
        # print(eps.shape)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        print(x_T.shape)
        for i in range(10):
            # print(i)
            # print(labels[i])
            x_axis_data = np.arange(0, 128, 1)  # x
            y_axis_data_I = np.squeeze(x_T[i:i + 1, :, 0:1, :])
            y_axis_data_Q = np.squeeze(x_T[i:i + 1, :, 1:2, :])
            plt.figure(figsize=(10, 8))

            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5, linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5, linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
            plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
            plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
            plt.xticks(fontsize=30, fontname='Times New Roman')
            plt.yticks(fontsize=30, fontname='Times New Roman')

            # plt.legend()  # 显示上面的label
            # plt.xlabel('time')  # x_label
            # plt.ylabel('number')  # y_label

            # plt.ylim(-1,1)#仅设置y轴坐标范围
            # plt.show()
            plt.savefig('./SampledSignals3/allGuidence_{}_{}.jpg'.format(i, labels[i]), bbox_inches='tight')
            # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # plt.show()
            plt.close()

        x_t = x_T
        for time_step in reversed(range(self.T)):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t) * 0.25
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            if time_step == 499 or time_step == 299 or time_step == 199 or time_step == 99 or time_step == 399:
                x_t = torch.clip(x_t, -1, 1)
                for i in range(10):
                    # print(i)
                    # print(labels[i])
                    x_axis_data = np.arange(0, 128, 1)  # x
                    y_axis_data_I = np.squeeze(x_t[i:i + 1, :, 0:1, :])
                    y_axis_data_Q = np.squeeze(x_t[i:i + 1, :, 1:2, :])
                    plt.figure(figsize=(10, 8))

                    plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5,
                             linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
                    plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5,
                             linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
                    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
                    plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
                    plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
                    plt.xticks(fontsize=30, fontname='Times New Roman')
                    plt.yticks(fontsize=30, fontname='Times New Roman')

                    # plt.legend()  # 显示上面的label
                    # plt.xlabel('time')  # x_label
                    # plt.ylabel('number')  # y_label

                    # plt.ylim(-1,1)#仅设置y轴坐标范围
                    # plt.show()
                    plt.savefig('./SampledSignals3/SampledGuidenceSignal_{}_{}_{}.jpg'.format(i, labels[i], time_step), bbox_inches='tight')
                    # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
                    # plt.show()
                    plt.close()
            #
                    # plt.figure(figsize=(10, 8), dpi=100)
                    plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
                    # plt.show()
                    plt.savefig('./SampledSignals3/Constellation_SampledGuidenceSignal_{}_{}_{}.jpg'.format(i, labels[i], time_step), bbox_inches='tight')
                    # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
                    # plt.show()
                    plt.close()


                    x_axis_data = np.arange(0, 128, 1)  # x
                    y_axis_data_I = np.squeeze(noise[i:i + 1, :, 0:1, :])
                    y_axis_data_Q = np.squeeze(noise[i:i + 1, :, 1:2, :])
                    plt.figure(figsize=(10, 8))
                    plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
                    plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
                    plt.xticks(fontsize=30, fontname='Times New Roman')
                    plt.yticks(fontsize=30, fontname='Times New Roman')

                    plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5,
                             linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
                    plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5,
                             linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
                    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

                    # plt.legend()  # 显示上面的label
                    # plt.xlabel('time')  # x_label
                    # plt.ylabel('number')  # y_label

                    # plt.ylim(-1,1)#仅设置y轴坐标范围
                    # plt.show()
                    plt.savefig('./SampledSignals3/SampledGuidenceNoise_{}_{}_{}.jpg'.format(i, labels[i], time_step), bbox_inches='tight')
                    # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
                    # plt.show()
                    plt.close()
            #
            #         # plt.figure(figsize=(10, 8), dpi=100)
            #         plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
            #         # plt.show()
            #         plt.savefig('./SampledSignals3/Constellation_SampledGuidenceNoise_{}_{}_{}.jpg'.format(i, labels[i],
            #                                                                                                time_step))
            #         # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            #         # plt.show()
            #         plt.close()

                    # y_axis_data_I = np.squeeze(x_t[i:i + 1, :, 0:1, :])
                    # y_axis_data_Q = np.squeeze(x_t[i:i + 1, :, 1:2, :])
                    # plt.figure(figsize=(10, 10), dpi=100)
                    # plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
                    # # plt.show()
                    # plt.savefig('./SampledSignals/Constellation_{}_{}_{}.jpg'.format(i, labels[i], time_step))
                    # # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
                    # # plt.show()
                    # plt.close()



        x_0 = x_t
        out = torch.clip(x_0, 0, 1)
        # print(ea)

        for i in range(20):
            # print(i)
            # print(labels[i])
            x_axis_data = np.arange(0, 128, 1)  # x
            y_axis_data_I = np.squeeze(out[i:i + 1, :, 0:1, :])
            y_axis_data_Q = np.squeeze(out[i:i + 1, :, 1:2, :])
            plt.figure(figsize=(10, 8))

            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5,
                     linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5,
                     linewidth=5)  # 'bo-'表示蓝色实线，数据点实心原点标注
            ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
            plt.xlabel('Time', fontsize=30, fontname='Times New Roman')  # x_label
            plt.ylabel('Amplitude', fontsize=30, fontname='Times New Roman')  # y_label
            plt.xticks(fontsize=30, fontname='Times New Roman')
            plt.yticks(fontsize=30, fontname='Times New Roman')

            # plt.legend()  # 显示上面的label
            # plt.xlabel('time')  # x_label
            # plt.ylabel('number')  # y_label

            # plt.ylim(-1,1)#仅设置y轴坐标范围
            # plt.show()
            plt.savefig('./SampledSignals4/END_SampledGuidenceSignal_{}_{}.jpg'.format(labels[i], i), bbox_inches='tight')
            # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # plt.show()
            plt.close()

            # plt.figure(figsize=(10, 8), dpi=100)
            plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
            # plt.show()
            plt.savefig(
                './SampledSignals4/END_Constellation_SampledGuidenceSignal_{}_{}.jpg'.format(labels[i], i), bbox_inches='tight')
            # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
            # plt.show()
            plt.close()
        print(ea)

        return out


