

import os
from typing import Dict
import numpy as np
import tensorflow as tf
import torch
import torch.optim as optim
from torch import Tensor, nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torch.utils.data as Data

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler

import matplotlib.pyplot as plt
import numpy as np




def TrainDataset(r):
    x = np.load(f'train_GT/x_GT.npy')
    x = x.transpose((0, 2, 1))
    y = np.load(f'train_GT/y_GT.npy')
    # x_train, x_val, y_train, y_val = train_test_split(x[:, :, 0:2], y, test_size=0.3)
    print(x.shape)
    # # print(x_val.shape)
    # print(y.shape)
    # print(y_val.shape)
    # x_train, y_train = Rotate_DA(x_train, y_train)
    # print(x_train.shape)
    # print(y_train.shape)
    x = x.transpose((0, 2, 1))
    # x_val = x_val.transpose((0, 2, 1))
    # print(x_train.shape)
    # print(x_val.shape)
    x = tf.expand_dims(x, axis=1)

    # 对每个样本归一化到 [0, 1]
    normalized_data = (x - np.min(x, 3, keepdims=True)) / (
                np.max(x, 3, keepdims=True) - np.min(x, 3, keepdims=True))

    # print("原数据最大值：\n", np.max(x, 3, keepdims=True))
    # print("原数据最小值：\n", np.min(x, 3, keepdims=True))
    # print("归一化后最大值：\n", np.max(normalized_data, 3, keepdims=True))
    # print("归一化后最小值：\n", np.min(normalized_data, 3, keepdims=True))

    # x_val = tf.expand_dims(x_val, axis=1)
    # print(x_train.shape)
    # print(x_val.shape)
    # y_train = to_categorical(y_train)
    # y_val = to_categorical(y_val)
    # print(x.shape)
    # print(y.shape)
    # print(y)
    return normalized_data, y

# def TestDataset(snr):
#     x = np.load(f"test/x_mix_snr={snr}.npy")
#     # x = tf.expand_dims(x[:, :, 0:2], axis=1)
#     # x = x.transpose((0, 2, 1))
#     print("TestDataset")
#     print(x.shape)
#     y = np.load(f"test/y_mix_snr={snr}.npy")
#     # print(y)
#     # y = to_categorical(y)
#     return x, y

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    x_train, y_train = TrainDataset(1)
    # print(x_train.shape)
    # print(y_train.shape)
    # 先转换成 torch 能识别的 Dataset
    a = np.array(x_train)
    b = np.array(y_train)
    c = torch.from_numpy(a)
    d = torch.from_numpy(b)
    dataset = Data.TensorDataset(c, d)

    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    # ckpt = torch.load(os.path.join(
    #     modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
    # net_model.load_state_dict(ckpt)
    # print("model load weight done.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 2, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    loss_a = []
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # for i in range(30):
                #     # print(i)
                #     # print(labels[i])
                #     x_axis_data = np.arange(0, 128, 1)  # x
                #     y_axis_data_I = np.squeeze(images[i:i + 1, :, 0:1, :])
                #     y_axis_data_Q = np.squeeze(images[i:i + 1, :, 1:2, :])
                #     plt.figure(figsize=(10, 8))
                #
                #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5,
                #              linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
                #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5,
                #              linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
                #     ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
                #
                #     # plt.legend()  # 显示上面的label
                #     plt.xlabel('Time', fontsize=20, fontname='Times New Roman')  # x_label
                #     plt.ylabel('Amplitude', fontsize=20, fontname='Times New Roman')  # y_label
                #     plt.xticks(fontsize=20, fontname='Times New Roman')
                #     plt.yticks(fontsize=20, fontname='Times New Roman')
                #
                #     # plt.ylim(-1,1)#仅设置y轴坐标范围
                #     # plt.show()
                #     plt.savefig('./SampledSignals5/x_{}_{}.jpg'.format(labels[i], i))
                #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
                #     # plt.show()
                #     plt.close()
                #
                #     # plt.figure(figsize=(20, 20), dpi=100)
                #     plt.scatter(Tensor.cpu(y_axis_data_I), Tensor.cpu(y_axis_data_Q))
                #     # plt.show()
                #     plt.savefig('./SampledSignals5/Constellation_x_{}_{}.jpg'.format(labels[i], i))
                #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
                #     # plt.show()
                #     plt.close()
                # train, torch.Size([80])
                # torch.Size([80, 3, 32, 32])
                # print(images.shape)
                # print(labels.shape)
                # print(labels)
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) #+ 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                # print(labels)
                # print(labels.shape)
                # print(x_0.shape)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                loss_a.append(loss)
        #         import csv
        #         with open("losses.csv", "w", newline="") as file:
        #             writer = csv.writer(file)
        #             writer.writerow(["Epoch", "Loss"])
        #             for epoch, loss in enumerate(loss_a):
        #                 writer.writerow([epoch + 1, loss.cpu().detach().numpy()])
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) #+ 1
        # print(labels)
        # print("labels: ", labels)
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 1, modelConfig["img_size1"], modelConfig["img_size2"]], device=device) #* 0.1
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, labels)
        # print(sampledImgs)
        # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # print(sampledImgs.shape)
        # print(labels.shape)
        # np.save('train_GS/x_GS.npy', sampledImgs)
        # np.save('train_GS/y_GS.npy', labels)
        # for i in range(len(sampledImgs)):
        #     # print(i)
        #     # print(labels[i])
        #     x_axis_data = np.arange(0, 128, 1)  # x
        #     y_axis_data_I = np.squeeze(sampledImgs[i:i+1, :, 0:1, :])
        #     y_axis_data_Q = np.squeeze(sampledImgs[i:i + 1, :, 1:2, :])
        #
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_I), 'b-', alpha=0.5, linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     plt.plot(x_axis_data, Tensor.cpu(y_axis_data_Q), 'r-', alpha=0.5, linewidth=1)  # 'bo-'表示蓝色实线，数据点实心原点标注
        #     ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
        #
        #     # plt.legend()  # 显示上面的label
        #     # plt.xlabel('time')  # x_label
        #     # plt.ylabel('number')  # y_label
        #
        #     # plt.ylim(-1,1)#仅设置y轴坐标范围
        #     # plt.show()
        #
        #     plt.savefig('./SampledImgs/SampledGuidenceSignal_{}_{}.jpg'.format(labels[i], i))
        #     # plt.savefig('FFRCNet_Tsne_{}db.jpg'.format(snr), bbox_inches='tight', dpi=600)
        #     # plt.show()
        #     plt.close()

        save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        return sampledImgs, labels