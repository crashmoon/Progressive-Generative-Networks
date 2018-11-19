from __future__ import print_function, division
import torch.nn as nn
import numpy as np

import plot

import scipy.misc
from scipy.misc import imsave

import torch
from torch.autograd import Variable
import torch.autograd as autograd

from os.path import join
from glob import glob
from torch.utils.data import Dataset, DataLoader

import os
from skimage import io, transform
from skimage.transform import resize
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import time
import random
import math
import cv2


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Param:
    unet_channel = 64
    cnn_channel = 64
    batch_size = 16
    image_size = 128
    n_critic = 1
    gan_weight = 0.001
    tv_weight = 1.0
    weight_decay = 0.00
    elliptic_a = 45.254833995939045
    elliptic_c = 34.899525263274995                                   
    G_learning_rate = 0.0002
    D_learning_rate = 0.00002
    out_path = '/mnt/zhanghaoran/oval/'


def conv_down(dim_in, dim_out):
    return nn.Sequential(
        nn.LeakyReLU(0.2),
        nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1,bias = False),
        nn.BatchNorm2d(dim_out)
    )


def conv_up(dim_in, dim_out):
    return nn.Sequential(
        nn.ReLU(),
        nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1, bias=False),
        nn.BatchNorm2d(dim_out)
    )


class Unet(nn.Module):
    def __init__(self, unet_input_channel=3, hidden_channel=Param.unet_channel * 8):
        super(Unet, self).__init__()
        self.start = nn.Conv2d(unet_input_channel, Param.unet_channel,3,1,1)  # 128
        self.conv0 = conv_down(Param.unet_channel, Param.unet_channel)  # 64
        self.conv1 = conv_down(Param.unet_channel, Param.unet_channel * 2)  # 32
        self.conv2 = conv_down(Param.unet_channel * 2, Param.unet_channel * 4)  # 16
        self.conv3 = conv_down(Param.unet_channel * 4, Param.unet_channel * 8)  # 8
        self.conv4 = conv_down(Param.unet_channel * 8, Param.unet_channel * 8)  # 4
        self.conv5 = conv_down(Param.unet_channel * 8, Param.unet_channel * 8)  # 2
        self.conv6 = conv_down(Param.unet_channel * 8, Param.unet_channel * 8)  # 1

        self.up5 = conv_up(hidden_channel, Param.unet_channel * 8)  # 2
        self.dp5 = nn.Dropout(p=0.5)
        self.up4 = conv_up(Param.unet_channel * 8 * 2, Param.unet_channel * 8)  # 4
        self.dp4 = nn.Dropout(p=0.5)
        self.up3 = conv_up(Param.unet_channel * 8 * 2, Param.unet_channel * 8)  # 8
        self.dp3 = nn.Dropout(p=0.5)
        self.up2 = conv_up(Param.unet_channel * 8 * 2, Param.unet_channel * 4)  # 16
        self.up1 = conv_up(Param.unet_channel * 4 * 2, Param.unet_channel * 2)  # 32
        self.up0 = conv_up(Param.unet_channel * 2 * 2, Param.unet_channel)  # 64
        self.end = conv_up(Param.unet_channel * 2, 3)  # 128

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data_in, hidden_input=None):
        start_out = self.start(data_in)
        conv0_out = self.conv0(start_out)
        conv1_out = self.conv1(conv0_out)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)

        mid = conv6_out  # Param.batch_size * 256 * 1 * 1

        if hidden_input is None:
            up5_out = self.up5(conv6_out)
        else:
            hidden_input = hidden_input.view(hidden_input.size(0), hidden_input.size(1), 1, 1)
            up5_out = self.up5(torch.cat((hidden_input, conv6_out), 1))

        up4_out = self.up4(torch.cat((up5_out, conv5_out), 1))
        up3_out = self.up3(torch.cat((up4_out, conv4_out), 1))
        up2_out = self.up2(torch.cat((up3_out, conv3_out), 1))
        up1_out = self.up1(torch.cat((up2_out, conv2_out), 1))
        up0_out = self.up0(torch.cat((up1_out, conv1_out), 1))
        out = self.end(torch.cat((up0_out, conv0_out), 1))
        out = F.sigmoid(out)
        return out, mid


def conv_stage(dim_in, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 4, 2, 1,bias=False),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(dim_out)
    )


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = nn.Conv2d(3, Param.cnn_channel, 3, 1, 1,bias=False)
        self.conv1 = conv_stage(Param.cnn_channel, Param.cnn_channel * 2)
        self.conv2 = conv_stage(Param.cnn_channel * 2, Param.cnn_channel * 4)
        self.conv3 = nn.Conv2d(Param.cnn_channel * 4, 1, 4, 1, 1)
        self.bn0 = nn.BatchNorm2d(Param.cnn_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data_in):  # map   channel
        conv0_out = self.conv0(data_in)  # 128   64
        conv0_out = self.bn0(conv0_out)
        conv1_out = self.conv1(conv0_out)  # 64    128
        conv2_out = self.conv2(conv1_out)  # 32    256
        out = self.conv3(conv2_out)  # 31    1
        out = F.sigmoid(out)
        return out

class ImageNetData(object):
    def __init__(self, csv_file, trans=None):
        self.lines = pd.read_csv(csv_file)
        self.trans = trans

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_pos = self.lines.ix[idx, 0]
        image = io.imread(image_pos)
        image = image.astype(np.float)
        h,w = image.shape[:2]
        if(h<w):
            factor = h/350.0
            w = w/factor
            h = 350
        else:
            factor = w/350.0
            h = h/factor
            w = 350
        image = transform.resize(image, (int(h), int(w), 3))
        image_id = self.lines.ix[idx, 1]
        sample = {'image': image, 'id': image_id}
        if self.trans is not None:
            sample = self.trans(sample)
        return sample

class ParisData(object):
    def __init__(self, csv_file, trans=None):
        self.lines = pd.read_csv(csv_file)
        self.trans = trans

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_pos = self.lines.ix[idx, 0]
        image = io.imread(image_pos)
        image = image.astype(np.float)
        h,w = image.shape[:2]
        if(h<w):
            factor = h/350.0
            w = w/factor
            h = 350
        else:
            factor = w/350.0
            h = h/factor
            w = 350
        image = transform.resize(image, (int(h), int(w), 3))
        image_id = self.lines.ix[idx, 1]
        sample = {'image': image, 'id': image_id}
        if self.trans is not None:
            sample = self.trans(sample)
        return sample


class RandCrop(object):
    def __call__(self, sample):
        image = sample['image']
        image_id = sample['id']
        h, w = image.shape[:2]
        sx = random.randint(0, h - Param.image_size)
        sy = random.randint(0, w - Param.image_size)
        image = image[sx:(sx + Param.image_size), sy:(sy + Param.image_size)]
        image = image.transpose((2, 0, 1))
        if(random.randint(0,1)):
            image = image[:,:,::-1]
        image /= 255.0
        return {'image': torch.FloatTensor(image), 'id': torch.Tensor([image_id])}


def inf_get(train):
    while (True):
        for x in train:
            yield x['image']

'''
def destroy(image):
    re = image.clone()
    re[:, :, 32:32 + 64, 32:32 + 64] = torch.zeros(Param.batch_size, 3, 64, 64)
    return re
'''

'''
def destroy(image, crop_size=64):
    re = image.clone().cuda()
    re[:, 0, int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size),
    int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size)] = torch.zeros(
        Param.batch_size, 1, crop_size, crop_size).fill_(0.45703125).cuda()
    re[:, 1, int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size),
    int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size)] = torch.zeros(
        Param.batch_size, 1, crop_size, crop_size).fill_(0.40625).cuda()
    re[:, 2, int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size),
    int((Param.image_size - crop_size) / 2):int((Param.image_size - crop_size) / 2 + crop_size)] = torch.zeros(
        Param.batch_size, 1, crop_size, crop_size).fill_(0.48046875).cuda()

    return re
'''

def destroy(image, crop_percent = 1.0):
    re = image.clone()
    c1 = 39
    c2 = 88
    divide = (Param.elliptic_c + (Param.elliptic_a - Param.elliptic_c) * crop_percent)*2.0 - 1.0

    for channel in range(3):
        for x in range(128):
            for y in range(128):
                f1 = math.sqrt((x-c1)**2 + (y-c1)**2)
                f2 = math.sqrt((x-c2)**2 + (y-c2)**2)
                if ( (f1 + f2) < divide):
                    re[:,0,x,y] = 0.999#0.45703125
                    re[:,1,x,y] = 0.999#0.40625
                    re[:,2,x,y] = 0.999#0.48046875
    re = re.cuda()
    return re
                    
                      
                


class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        self.unet_1 = Unet(3, Param.unet_channel * 8)
        self.unet_2 = Unet(6, Param.unet_channel * 8 * 3)
        self.unet_3 = Unet(6, Param.unet_channel * 8 * 3)
        self.unet_4 = Unet(6, Param.unet_channel * 8 * 3)
        self.rnn = nn.LSTMCell(Param.unet_channel * 8, Param.unet_channel * 8 * 2)

    def forward(self, data_1, data_2, data_3, data_4, h0, c0):
        #print(data_1.size())
        unet_out_1, unet_mid_1 = self.unet_1(data_1)
        h1, c1 = self.rnn(unet_mid_1.view(Param.batch_size, -1), (h0, c0))
        unet_out_2, unet_mid_2 = self.unet_2(torch.cat((data_1, unet_out_1), 1), h1)
        h2, c2 = self.rnn(unet_mid_2.view(Param.batch_size, -1), (h1, c1))
        unet_out_3, unet_mid_3 = self.unet_3(torch.cat((data_1, unet_out_2), 1), h2)
        h3, c3 = self.rnn(unet_mid_3.view(Param.batch_size, -1), (h2, c2))
        unet_out_4, unet_mid_4 = self.unet_4(torch.cat((data_1, unet_out_3), 1), h3)
        return unet_out_1, unet_out_2, unet_out_3, unet_out_4


class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        self.cnn1 = CNN()
        self.cnn2 = CNN()
        self.cnn3 = CNN()
        self.cnn4 = CNN()

    def forward(self, data_48, data_32, data_16, data_0):
        out1 = self.cnn1(data_48)
        out2 = self.cnn2(data_32)
        out3 = self.cnn3(data_16)
        out4 = self.cnn4(data_0)
        return out1, out2, out3, out4


def save_image_plus(x, save_path):
    x = (255.99 * x).astype('uint8')
    x = x.transpose(0, 1, 3, 4, 2)
    nh, nw = x.shape[:2]
    h = x.shape[2]
    w = x.shape[3]
    img = np.zeros((h * nh, w * nw, 3))
    for i in range(nh):
        for j in range(nw):
            img[i * h:i * h + h, j * w:j * w + w] = x[i][j]
    imsave(save_path, img)

def cal_tv(image):
    temp = image.clone()
    temp[:,:,:Param.image_size-1,:] = image[:,:,1:,:]
    re = ((image-temp)**2).mean()
    temp = image.clone()
    temp[:,:,:,:Param.image_size-1] = image[:,:,:,1:]
    re += ((image-temp)**2).mean()
    return re

def main():
    one = torch.FloatTensor([1.0]).cuda()
    mone = torch.FloatTensor([-1.0]).cuda()
    ones_31 = torch.zeros(Param.batch_size, 1, 31, 31).fill_(1.0).type(torch.FloatTensor).cuda()
    mones_31 = torch.zeros(Param.batch_size, 1, 31, 31).fill_(-1.0).type(torch.FloatTensor).cuda()
    zeros_31 = torch.zeros(Param.batch_size, 1, 31, 31).type(torch.FloatTensor).cuda()

    mask = torch.ones(Param.batch_size, 3, 128, 128)
    mask[:, :, 32:32 + 64, 32:32 + 64] = torch.zeros(Param.batch_size, 3, 64, 64)
    mask = Variable(mask.type(torch.FloatTensor).cuda(), requires_grad=False)

    h0 = torch.zeros(Param.batch_size, Param.unet_channel * 8 * 2).cuda()
    c0 = torch.zeros(Param.batch_size, Param.unet_channel * 8 * 2).cuda()

    netG = Net_G().cuda()
    netD = Net_D().cuda()

    #netG.load_state_dict(torch.load('/home/lmc-09/PycharmProjects/unet_wgan/gan_lstm_l2_y/netG_79999.pickle'))
    #netD.load_state_dict(torch.load('/home/lmc-09/PycharmProjects/unet_wgan/gan_lstm_l2_y/netD_79999.pickle'))
    #netG = nn.DataParallel(netG, device_ids=[0, 1])
    #netD = nn.DataParallel(netD, device_ids=[0, 1])

    opt_G = optim.Adam(netG.parameters(), lr=Param.G_learning_rate, betas = (0.5,0.999), weight_decay=Param.weight_decay)
    opt_D = optim.Adam(netD.parameters(), lr=Param.D_learning_rate, betas = (0.5,0.999), weight_decay=Param.weight_decay)

    #trainset = ParisData('paris.csv', RandCrop())
    trainset = ParisData('paris.csv', RandCrop())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=Param.batch_size, shuffle=True, num_workers=2,
                                               drop_last=True)
    train_data = inf_get(train_loader)

    epoch = 0
    maxepoch = 200000
    bce_loss = nn.BCELoss()

    while (epoch < maxepoch):
        start_time = time.time()
        # step D
        for p in netD.parameters():
            p.requires_grad = True
        #for D_step in range(Param.n_critic):
        ###################################
        ######   D  #######################
        ###################################
        real_data = train_data.next()
        # print(real_data)
        real_data = real_data.cuda()
        real_data_64 = destroy(real_data, 1.0)
        real_data_48 = destroy(real_data, 0.75)
        real_data_32 = destroy(real_data, 0.5)
        real_data_16 = destroy(real_data, 0.25)

        real_data_64 = Variable(real_data_64)
        real_data_48 = Variable(real_data_48)
        real_data_32 = Variable(real_data_32)
        real_data_16 = Variable(real_data_16)
        real_data_0 = Variable(real_data)

        netD.zero_grad()
        p_real_48, p_real_32, p_real_16, p_real_0 = netD(real_data_48, real_data_32, real_data_16, real_data_0)
        target = Variable(ones_31)

        #print(p_real_48.size())
        real_loss_48 = bce_loss(p_real_48, target)
        real_loss_32 = bce_loss(p_real_32, target)
        real_loss_16 = bce_loss(p_real_16, target)
        real_loss_0 = bce_loss(p_real_0, target)

        fake_data_48, fake_data_32, fake_data_16, fake_data_0 = netG(real_data_64, real_data_48, real_data_32,
                                                                        real_data_16, Variable(h0), Variable(c0))

        p_fake_48, p_fake_32, p_fake_16, p_fake_0 = netD(Variable(fake_data_48.data), Variable(fake_data_32.data), Variable(fake_data_16.data), Variable(fake_data_0.data))
        target = Variable(zeros_31)
        fake_loss_48 = bce_loss(p_fake_48, target)
        fake_loss_32 = bce_loss(p_fake_32, target)
        fake_loss_16 = bce_loss(p_fake_16, target)
        fake_loss_0 = bce_loss(p_fake_0, target)

        gan_loss = real_loss_48 + real_loss_32 + real_loss_16 + real_loss_0 + fake_loss_48 + fake_loss_32 + fake_loss_16 + fake_loss_0
        '''
        real_loss_48.backward(retain_graph=True)
        real_loss_32.backward(retain_graph=True)
        real_loss_16.backward(retain_graph=True)
        real_loss_0.backward(retain_graph=True)

        fake_loss_48.backward(retain_graph=True)
        fake_loss_32.backward(retain_graph=True)
        fake_loss_16.backward(retain_graph=True)
        fake_loss_0.backward(retain_graph=True)
        '''
        gan_loss.backward(retain_graph=True)

        D_cost = fake_loss_48.data[0] + fake_loss_32.data[0] + fake_loss_16.data[0] + fake_loss_0.data[0]
        D_cost += real_loss_48.data[0] + real_loss_32.data[0] + real_loss_16.data[0] + real_loss_0.data[0]

        opt_D.step()
        ##################
        ## step G ########
        ##################
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        #real_data = train_data.next()
        #real_data = real_data.cuda()
        #real_data_64 = destroy(real_data, 64)
        #real_data_48 = destroy(real_data, 48)
        #real_data_32 = destroy(real_data, 32)
        #real_data_16 = destroy(real_data, 16)

        #real_data_64 = Variable(real_data_64)
        #real_data_48 = Variable(real_data_48)
        #real_data_32 = Variable(real_data_32)
        #real_data_16 = Variable(real_data_16)
        #real_data_0 = Variable(real_data)

        #fake_data_48, fake_data_32, fake_data_16, fake_data_0 = netG(real_data_64, real_data_48, real_data_32, real_data_16, Variable(h0), Variable(c0))

        l1_loss = ((fake_data_48 - real_data_48).abs()).mean() + ((fake_data_32 - real_data_32).abs()).mean() + ((
            fake_data_16 - real_data_16).abs()).mean() + ((fake_data_0 - real_data_0).abs()).mean()

        tv_loss = cal_tv(fake_data_48) + cal_tv(fake_data_32) + cal_tv(fake_data_16) + cal_tv(fake_data_0)
        tv_loss = tv_loss * Param.tv_weight

        p_fake_48, p_fake_32, p_fake_16, p_fake_0 = netD(fake_data_48, fake_data_32, fake_data_16, fake_data_0)
        target = Variable(ones_31)
        fake_loss_48 = bce_loss(p_fake_48, target)
        fake_loss_32 = bce_loss(p_fake_32, target)
        fake_loss_16 = bce_loss(p_fake_16, target)
        fake_loss_0 = bce_loss(p_fake_0, target)

        gan_loss = fake_loss_48 + fake_loss_32 + fake_loss_16 + fake_loss_0
        gan_loss = gan_loss * Param.gan_weight

        gan_loss.backward(retain_graph=True)
        l1_loss.backward(one, retain_graph=True)
        tv_loss.backward(one, retain_graph=True)

        G_cost = fake_loss_48.data[0] + fake_loss_32.data[0] + fake_loss_16.data[0] + fake_loss_0.data[0]
        G_cost += l1_loss.data[0]
        opt_G.step()
        print('epoch: ' + str(epoch) + ' l1_loss: ' + str(l1_loss.data[0]))

        # Write logs and save samples
        #print(D_cost.size())
        #print(G_cost.size())
        os.chdir(Param.out_path)
        plot.plot('train D cost', D_cost)
        plot.plot('time', time.time() - start_time)
        plot.plot('train G cost', G_cost)
        plot.plot('train l1 loss', l1_loss.data.cpu().numpy())

        if epoch % 100 == 99:
            #  real_data = train_data.next()
            out_image = torch.cat(
                (
                    fake_data_48.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    fake_data_32.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    fake_data_16.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    fake_data_0.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    real_data_64.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    real_data_48.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    real_data_32.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    real_data_16.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size),
                    real_data_0.data.view(Param.batch_size, 1, 3, Param.image_size, Param.image_size)
                ),
                1
            )
            # out_image.transpose(0,1,3,4,2)
            save_image_plus(out_image.cpu().numpy(), Param.out_path + 'train_image_{}.jpg'.format(epoch))

        if (epoch < 5) or (epoch % 100 == 99):
            plot.flush()
        plot.tick()

        if epoch % 20000 == 19999:
            torch.save(netD.state_dict(),Param.out_path+ 'netD_{}.pickle'.format(epoch))
            torch.save(netG.state_dict(),Param.out_path+ 'netG_{}.pickle'.format(epoch))
            #opt_D.param_groups[0]['lr'] /= 10.0
            #opt_G.param_groups[0]['lr'] /= 10.0
        epoch += 1


if __name__ == '__main__':
    main()
















