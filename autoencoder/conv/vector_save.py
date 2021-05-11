import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np
import json
import argparse

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_path", required=True,
                 type=str, help="Please enter data path")
par.add_argument("-f", "--num_features", default=64, choices=[64, 256],
                 type=int, help="Set the feature size. (64/256)")
par.add_argument("-t", "--types", default="semantic", choices=["real", "semantic_annotations"],
                 type=str, help="Choose a data type. (real/semantic_annotations)")
par.add_argument("-i", "--iteration", default=5,
                 type=int, help="number of iteration")
args = par.parse_args()

## Set the parameters and data path
data_path = args.data_path + "/"
num_features = args.num_features
types = args.types

data_path = data_path + ("activity" if types == "real" else types)
test_23_path = data_path + "/image/test_23"
test_34_path = data_path + "/image/test_34"

## Set data loader
dataset_23 = datasets.ImageFolder(root=test_23_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5,0.5, 0.5)),
                           ]))

dataset_34 = datasets.ImageFolder(root=test_34_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5,0.5, 0.5)),
                           ]))

## Convolutional autoencoder
class AutoEncoderConv(nn.Module):
    def __init__(self, num_features):
        super(AutoEncoderConv, self).__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(64*8*4, 2048)
        self.fc2 = nn.Linear(2048, 256)
        if self.num_features == 64:
            self.fc3 = nn.Linear(256, 64)
            self.de_fc1 = nn.Linear(64, 256)
        self.de_fc2 = nn.Linear(256, 2048)
        self.de_fc3 = nn.Linear(2048, 64*8*4)

        self.encoder = nn.Sequential(
            # Input : 3*256*128
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 8*128*64

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 16*64*32

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 16*32*16
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 32*16*8
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 64*8*4
        )

        self.decoder = nn.Sequential(
            # 64*8*4

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(True),
            # 32*16*8

            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(True),
            # 16*32*16
            
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.ReLU(True),
            # 16*64*32
            
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(True),
            # 8*128*64

            nn.ConvTranspose2d(8, 3, 2, stride=2),
            nn.Sigmoid()
            # 3*256*128 
        )
        
    def forward(self, x):
        y = self.encoder(x)
        y = F.relu(self.fc1(y.view(y.size(0), -1)))
        y = F.relu(self.fc2(y))
        if self.num_features == 64:
            y = F.relu(self.fc3(y))
        y_e = y
        
        # -- decoder --
        if self.num_features == 64:
            y = F.relu(self.de_fc1(y))
        y = F.relu(self.de_fc2(y))
        y = F.relu(self.de_fc3(y))
        y_d = self.decoder(y.view(y.size(0), 64, 8, 4))
        return y_e, y_d

def run(dataset_num, _iter):
    log_path = ("log/pth/best_loss_" + types + "_conv_autoencoder_d"
            + str(num_features) + "_" + str(_iter)  + "_save_model.pth")
    data_name = "conv_" + ("re" if types == "real" else "se") + "_" + str(dataset_num)

    model = AutoEncoderConv(num_features).cuda()
    model.load_state_dict(torch.load(log_path))
    model.eval()
    if dataset_num == 23:
        dataset = dataset_23
    elif dataset_num == 34:
        dataset = dataset_34

    name = []
    result = []
    name_dic = {}
    for data, img_path in tqdm(zip(dataset, dataset.imgs)):
        img = Variable(data[0]).cuda().unsqueeze(0)
        label = img_path[0].split('/')[-1].split('.')[0]
        output, predict = model(img)

        name.extend([label])
        result.extend(output.cpu().detach().tolist())

    result = np.array(result)

    save_path = "../../data/" +data_name + "_" + str(_iter) + "/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(save_path + data_name + "_" + str(_iter) + "_data.npy", result)
    name_dic["name"] = name
    name_json = json.dumps(name_dic)
    name_file = open(save_path + data_name + "_" + str(_iter) + "_names.json","w")
    name_file.write(name_json)
    name_file.close()

if __name__ == "__main__":
    for _iter in range(args.iteration):
        run(23, _iter)
        run(34, _iter)
