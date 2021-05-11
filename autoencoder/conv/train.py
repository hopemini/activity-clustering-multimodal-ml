import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os
from tqdm import tqdm
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
par.add_argument("-e", "--number_of_epochs", default=100,
                 type=int, help="number of epochs")
args = par.parse_args()

## Set the parameters and data path
data_path = args.data_path + "/"
num_features = args.num_features
types = args.types
num_epochs = args.number_of_epochs
learning_rate = 0.001
batch_size = 128

## Set each path
log_path = "log/pth/best_loss_" + types + "_conv_autoencoder_d" + str(num_features) + "_save_model.pth"
data_path = data_path + ("activity" if types == "real" else types)
data_all_path = data_path + "/image/all"

if not os.path.exists("./log/"):
    os.mkdir("./log/")
if not os.path.exists("./log/pth/"):
    os.mkdir("./log/pth/")
save_log_path = "./log/check_point/"
save_img_path = "./log/img/"
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)
save_log_path = save_log_path + ("re" if types == "real" else "se") + "_d" + str(num_features) + "/"
save_img_path = save_img_path + ("re" if types == "real" else "se") + "_d" + str(num_features) + "/"
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)

## Set data loader
dataset = datasets.ImageFolder(root=data_all_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=8)

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
        # -- encoder --
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

## Training
class Trainer:
    def __init__(self, n_features, _iter):
        self.n_features = n_features
        self.iter = _iter
        
    def train(self, dataloader, num_epochs):
        model = AutoEncoderConv(self.n_features).cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = 1.0
        best_epoch = 0

        for epoch in range(num_epochs):
            for data in tqdm(dataloader):
                img, _ = data
                img = Variable(img).cuda()
                # ===================forward=====================
                _, output = model(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            if epoch == num_epochs-1:
                save_image(img, save_img_path + "image_{}.png".format(self.iter))
                save_image(output, save_img_path + "g_image_{}.png".format(self.iter))

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch+1
                torch.save(model.state_dict(), "./log/pth/best_loss_"
                           + types + "_conv_autoencoder_d" + str(num_features)
                           + "_" + str(self.iter) + "_save_model.pth")

            print("Iter:{} epoch [{}/{}], loss:{:.4f}, best loss:{:.4f}[{}/{}]"
                  .format(self.iter, epoch+1, num_epochs, loss.item(),
                      best_loss, best_epoch, num_epochs))
            with open(save_log_path + str(self.iter) + "_" + "epoch" + str(epoch+1), 'w') as f:
                f.write("epoch [{}/{}], loss:{:.4f}, best loss:{:.4f}[{}/{}]"
                  .format(epoch+1, num_epochs, loss.item(),
                      best_loss, best_epoch, num_epochs))
            torch.save(model.state_dict(), "./log/pth/"
                       + types + "_conv_autoencoder_d" + str(num_features)
                       + "_" + str(self.iter) + "_save_model.pth")

        torch.save(model.state_dict(), "./log/pth/"
                   + types + "_conv_autoencoder_d" + str(num_features)
                   + "_" + str(self.iter) + "_save_model.pth")
        
if __name__ == "__main__":
    for _iter in range(args.iteration):
        trainer = Trainer(num_features, _iter)
        trainer.train(dataloader, num_epochs)
