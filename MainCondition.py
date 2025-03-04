from torch import Tensor

from DiffusionFreeGuidence.TrainCondition import train, eval
import numpy as np

def main(model_config=None):
    modelConfig = {
        # "state": "train", # train or eval
        "state": "eval",  # train or eval
        "epoch": 70,
        "batch_size": 20,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 3,
        "dropout": 0.05,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size1": 2,
        "img_size2": 128,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        # "training_load_weight": None,
        "training_load_weight": "ckpt_31_.pt",
        "test_load_weight": "ckpt_37_.pt",#ckpt_61_.pt
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 1
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        for i in range(250):
            print(i)
            sampledImgs, labels = eval(modelConfig)
            sampledImgs = Tensor.cpu(sampledImgs)
            labels = Tensor.cpu(labels)
            # print(labels)
            # 加载现有数据
            if i==0:
                # print(sampledImgs.shape)
                np.save('train_GS/x_GS_50000.npy', sampledImgs)
                np.save('train_GS/y_GS_50000.npy', labels)
            else:
                x_data = np.load('train_GS/x_GS_50000.npy')
                y_data = np.load('train_GS/y_GS_50000.npy')
                # print(x_data.shape)
                x = np.concatenate((x_data, sampledImgs), 0)
                y = np.concatenate((y_data, labels), 0)
                np.save('train_GS/x_GS_50000.npy', x)
                np.save('train_GS/y_GS_50000.npy', y)

            # numpy.concatenate()
            #
            # np.save('train_GS/x_GS.npy', sampledImgs)
            # np.save('train_GS/y_GS.npy', labels)



if __name__ == '__main__':
    main()
