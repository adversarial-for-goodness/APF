import torch.nn as nn
import torch
import numpy as np
import os
import sys
import shutil
import argparse
import logging as logger
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

from Attack import DI_MI_Attack

sys.path.append('../../')
from backbone.backbone_def import BackboneFactory



class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.channel_list = [3, 64, 128, 256, 512]
        self.down1 = DownBlock(self.channel_list[0], self.channel_list[1])
        self.down2 = DownBlock(self.channel_list[1], self.channel_list[2])
        self.down3 = DownBlock(self.channel_list[2], self.channel_list[3])
        self.down4 = DownBlock(self.channel_list[3], self.channel_list[4])

        self.channel_list_reverse = [512, 256, 128, 64, 3]
        self.up1 = UpBlock(self.channel_list_reverse[0], self.channel_list_reverse[1])
        self.up2 = UpBlock(self.channel_list_reverse[0], self.channel_list_reverse[2])
        self.up3 = UpBlock(self.channel_list_reverse[1], self.channel_list_reverse[3])
        self.up4 = UpBlock(self.channel_list_reverse[2], self.channel_list_reverse[4])

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x3_ = self.up1(x4)
        x2_ = self.up2(torch.cat([x3_, x3], dim=1))
        x1_ = self.up3(torch.cat([x2_, x2], dim=1))
        x0_ = self.up4(torch.cat([x1_, x1], dim=1))

        return x0_


class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, backbone_factory, head_factory=None):
        """Init face model by backbone factorcy and head factory.

        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        #self.head = head_factory.get_head()

    def forward(self, data):
        feat = self.backbone.forward(data)
        #pred = self.head.forward(feat, label)
        return feat

def MIDI(x, y, model, eps, num_iters=10):
    # x和y是一对图像，返回用MIDI生成的关于model在x上添加的噪声
    return None

def load_model(model, sd):
    try:
        sd = sd['state_dict']
    except:
        pass
    try:
        model.load_state_dict(sd)
    except:
        new_dict = {}
        model_dict = model.state_dict()
        for k in model_dict:
            new_dict[k] = sd['backbone.' + k]
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    return model


def train(conf):
    conf.device = torch.device('cuda:{}'.format(conf.device))
    criterion = torch.nn.MSELoss().cuda(conf.device)

    white = BackboneFactory('MobileFaceNet', conf.backbone_conf_file).get_backbone()

    pt = torch.load('../../black_model/MobileFaceNet.pth', map_location='cpu')
    load_model(white, pt)

    black = BackboneFactory('ResNet', conf.backbone_conf_file)
    black = FaceModel(black)

    pt = torch.load('out_dir/standard_web.pt', map_location='cpu')['state_dict']
    pt = {k: v for k, v in pt.items() if "head" not in k}
    black.load_state_dict(pt)

    unet = U_Net().to(conf.device).train()
    white = white.to(conf.device).eval()
    black = black.to(conf.device).eval()

    optimizer = optim.Adam(unet.parameters(), lr=1e-5)

    train_dataset = np.load(args.train_data, allow_pickle=True)
    #list_img = np.array(train_dataset) / 127.5 - 1.0
    list_img = np.array(train_dataset, dtype=np.float32) / 255.
    list_img = np.reshape(list_img, [-1, 112, 112, 3])
    list_img = list_img.transpose([0, 3, 1, 2])
    list_img_0 = list_img[0::2]
    list_img_1 = list_img[1::2]
    len_train = len(list_img_0)
    n_batch = len_train // conf.batch_size

    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    attacker = DI_MI_Attack(white, criterion, preprocess=normalize, bounding=(0, 1))
    # saver = tf.train.Saver(max_to_keep=1)
    for i in range(50):
        for j, batch in enumerate(range(n_batch)):
            x_batch = torch.from_numpy(list_img_0[batch * conf.batch_size:(batch + 1) * conf.batch_size]).to(conf.device)
            y_batch = torch.from_numpy(list_img_1[batch * conf.batch_size:(batch + 1) * conf.batch_size]).to(conf.device)
            optimizer.zero_grad()

            # 生成对抗噪声
            #x_noise = MIDI(x_batch, y_batch, white, conf.eps)
            with torch.no_grad():
                label = white(normalize(y_batch))

            _, x_noise = attacker.run(x_batch, label, conf.eps, conf.num_iters)

            x_noise = normalize(x_noise)

            output = unet(x_noise)
            s = torch.clamp(output, -conf.eps, conf.eps)
            image_adv = torch.clamp(x_batch + s, 0, 1.0)

            x_adv_embd = F.normalize(black(normalize(image_adv)), p=2, dim=1)
            en_embd = F.normalize(black(normalize(y_batch)), p=2, dim=1)
            loss = 10 - criterion(x_adv_embd, en_embd)
            loss.backward()
            optimizer.step()

            if j % (100) == 0:
                print(loss)

    torch.save({'state_dict': unet.state_dict()}, 'out_dir/apf_unet.pth')



if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root", type=str, default='/data/jiaming/datasets/faces/faces_emore/imgs',
                      help="The root folder of training set.")
    conf.add_argument("--backbone_type", type=str, default='ResNet',
                      help="Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type=str, default='../backbone_conf.yaml',
                      help="the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type=str, default='ArcFace',
                      help="mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type=str, default='../head_conf.yaml',
                      help="the path of head_conf.yaml.")
    conf.add_argument('--lr', type=float, default=0.01,
                      help='The initial learning rate.')
    conf.add_argument('--epoches', type=int, default=15,
                      help='The training epoches.')
    conf.add_argument('--batch_size', type=int, default=64,
                      help='The training batch size over all gpus.')
    conf.add_argument("--train_data", type=str, default='/data/jiaming/datasets/ms1m_pair.pickle')
    conf.add_argument('--device', type=int, default=0)

    conf.add_argument('--eps', type=float, default=16)
    conf.add_argument('--num_iters', type=int, default=10)
    args = conf.parse_args()

    args.eps = args.eps / 255

    train(args)

