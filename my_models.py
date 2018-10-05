from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import scipy.io as sio
import numpy as np

from tools.densenet import _DenseBlock, _Transition, DenseBlock

class UpsampleByPS(nn.Module):
    def __init__(self, upscale_factor, in_channels=1, is_out_layer=False):
        super(UpsampleByPS, self).__init__()
        self.is_out_layer = is_out_layer

        self.conv1 = nn.Conv2d(in_channels, 64, (5, 5), (1, 1), (2, 2))
        # self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.initParameters()

    def initParameters(self):
        stateDict = self.state_dict()
        nn.init.xavier_normal(stateDict['conv1.weight'])
        nn.init.xavier_normal(stateDict['conv2.weight'])
        # nn.init.xavier_normal(stateDict['conv3.weight'])
        # nn.init.xavier_normal(stateDict['conv4.weight'])

    def forward(self, x):
        # x = F.leaky_relu(self.conv1(x), negative_slope=0.1, inplace=True)
        # x = F.leaky_relu(self.conv2(x), negative_slope=0.1, inplace=True)
        # x = F.leaky_relu(self.conv3(x), negative_slope=0.1, inplace=True)
        # if self.use_sig:
        #     x = F.sigmoid(self.pixel_shuffle(self.conv4(x)))
        # else:
        #     x = F.leaky_relu(self.pixel_shuffle(self.conv4(x)))
        # return x

        # out = F.relu(self.conv1(x[0]), inplace=True)
        # cat_out = torch.cat([out, x[1]], 1)
        # out = F.relu(self.conv2(cat_out), inplace=True)
        # out = F.relu(self.conv3(out), inplace=True)
        out = F.relu(self.conv1(x))

        if self.is_out_layer:
            out = F.relu(self.pixel_shuffle(self.conv2(out)))
        else:
            out = self.pixel_shuffle(self.conv2(out))
        return out


class DFCN_PS_FS(nn.Module):
    """DFCN_PS_FS is short for DFCN with pixelshuffle and scale fusion"""
    def __init__(self, is_Train=True):
        super(DFCN_PS_FS, self).__init__()
        self.isTrain = is_Train

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)

        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.denseblock1 = _DenseBlock(num_layers=6, num_input_features=64, bn_size=4, growth_rate=32, drop_rate=0)
        self.transition1 = _Transition(num_input_features=256, num_output_features=256 // 2)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.denseblock2 = _DenseBlock(num_layers=12, num_input_features=128, bn_size=4, growth_rate=32, drop_rate=0)
        self.transition2 = _Transition(num_input_features=512, num_output_features=512 // 2)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.denseblock3 = _DenseBlock(num_layers=24, num_input_features=256, bn_size=4, growth_rate=32, drop_rate=0)
        self.transition3 = _Transition(num_input_features=1024, num_output_features=1024 // 2)
        
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.denseblock4 = _DenseBlock(num_layers=16, num_input_features=512, bn_size=4, growth_rate=32, drop_rate=0)
        self.norm5 = nn.BatchNorm2d(1024)


        self.smthBlock = DenseBlock(inputDim=64, outputDim=128 ,growthRate=32, blockDepth=6)
        self.smthConv = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=5, padding=2, bias=False)
        self.smthUpsample = nn.PixelShuffle(2)

        self.deconv16_ = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.padding16 = nn.ReplicationPad2d((0, 0, 1, 0))

        self.bx32_dconvx8 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx32_dconvx4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx32_dconvx2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx32_dconvx1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx32_score = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.bx16_dconvx8 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx16_dconvx4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx16_dconvx2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx16_dconvx1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx16_score = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.bx8_dconvx4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx8_dconvx2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx8_dconvx1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1))
        self.bx8_score = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False)

        # self.bx4_dconvx2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1))
        # self.bx4_dconvx1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1))
        # self.bx4_score = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False)
        
        self.subconv_to_8 = UpsampleByPS(2, 512, is_out_layer=True)
        self.subconv_to_4 = UpsampleByPS(2, 1+256+256, is_out_layer=True)
        self.subconv_to_2 = UpsampleByPS(2, 1+128+128+128, is_out_layer=True)
        self.subconv_to_1_ = UpsampleByPS(2, 1+64+64+64, is_out_layer=True)

        self.fs_score_ = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, padding=0, bias=False)

        self.initParameters()
        self.fixLayer()

    def setTrainMode(self, isTrain):
        self.isTrain = isTrain
        
    def fixLayer(self):
        for param in self.parameters():
            if param is not None:
                param.requires_grad = False
        layerList = [self.smthBlock, self.smthConv, self.smthUpsample]
        for layer in layerList:
            for param in layer.parameters():
                if param is not None:
                    param.requires_grad = True

    def parameters(self):
        """
        overload Module.parameters
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

    def initParameters(self):
        stateDict = self.state_dict()
        # nn.init.xavier_normal(stateDict['conv_1.weight'])

        nn.init.xavier_normal(stateDict['deconv16_.weight'])
        nn.init.xavier_normal(stateDict['bx32_dconvx8.weight'])
        nn.init.xavier_normal(stateDict['bx32_dconvx4.weight'])
        nn.init.xavier_normal(stateDict['bx32_dconvx2.weight'])
        nn.init.xavier_normal(stateDict['bx32_dconvx1.weight'])
        nn.init.xavier_normal(stateDict['bx32_score.weight'])
        nn.init.xavier_normal(stateDict['bx16_dconvx8.weight'])
        nn.init.xavier_normal(stateDict['bx16_dconvx4.weight'])
        nn.init.xavier_normal(stateDict['bx16_dconvx2.weight'])
        nn.init.xavier_normal(stateDict['bx16_dconvx1.weight'])
        nn.init.xavier_normal(stateDict['bx16_score.weight'])
        nn.init.xavier_normal(stateDict['bx8_dconvx4.weight'])
        nn.init.xavier_normal(stateDict['bx8_dconvx2.weight'])
        nn.init.xavier_normal(stateDict['bx8_dconvx1.weight'])
        nn.init.xavier_normal(stateDict['bx8_score.weight'])
        # nn.init.xavier_normal(stateDict['bx4_dconvx2.weight'])
        # nn.init.xavier_normal(stateDict['bx4_dconvx1.weight'])
        # nn.init.xavier_normal(stateDict['bx4_score.weight'])
        nn.init.uniform(stateDict['fs_score_.weight'])
        nn.init.xavier_normal(stateDict['smthConv.weight'])

    def alignScale(self, inputData, scaleSize):
        inputShape = inputData.data.shape
        if scaleSize[0] == inputShape[2] and scaleSize[1] == inputShape[3]:
            return inputData
        elif abs(scaleSize[0]-inputShape[2]) <= 2 and abs(scaleSize[1]-inputShape[3]) <= 2:
            return nn.functional.upsample(inputData, size=scaleSize, mode='bilinear')
        else:
            raise ValueError('target size[{}, {}] is far from input size[{}, {}]'
                                .format(scaleSize[0], scaleSize[1], inputShape[2], inputShape[3]))

    def forward(self, x):
        # inputShape = x.data.shape
        # sizex1 = (inputShape[2], inputShape[3])
        # sizex2 = (sizex1[0]//2, sizex1[1]//2)
        # sizex4 = (sizex2[0]//2, sizex2[1]//2)
        # sizex8 = (sizex4[0]//2, sizex4[1]//2)
        # sizex16 = (sizex8[0]//2, sizex8[1]//2)
        # sizex32 = (sizex16[0]//2, sizex16[1]//2)

        # out_2 = F.relu(self.bn_1(self.conv_1(x)))
        # out_4 = self.denseBlock1(self.pooling_1(out_2))
        # out_8 = self.denseBlock2(self.pooling_2(out_4))
        # out_16 = self.denseBlock3(self.pooling_3(out_8))
        # out_32 = F.relu(self.bn_4(self.denseBlock4(self.pooling_4(out_16))))
        # if self.isTrain:
        #     out.volatile = False

        out_2 = self.relu0(self.norm0(self.conv0(x)))
        out_4 = self.transition1(self.denseblock1(self.pool0(out_2)))
        out_8 = self.transition2(self.denseblock2(self.pool1(out_4)))
        out_16 = self.transition3(self.denseblock3(self.pool2(out_8)))
        out_32 = self.norm5(self.denseblock4(self.pool3(out_16)))

        out_up_16 = self.padding16(self.deconv16_(out_32))
        bx32_outx8 = F.relu(self.bx32_dconvx8(out_up_16))
        bx32_outx4 = F.relu(self.bx32_dconvx4(bx32_outx8))
        bx32_outx2 = F.relu(self.bx32_dconvx2(bx32_outx4))
        bx32_outx1 = F.relu(self.bx32_dconvx1(bx32_outx2))
        bx32_score = self.bx32_score(bx32_outx1)

        bx16_outx8 = F.relu(self.bx16_dconvx8(out_16))
        bx16_outx4 = F.relu(self.bx16_dconvx4(bx16_outx8))
        bx16_outx2 = F.relu(self.bx16_dconvx2(bx16_outx4))
        bx16_outx1 = F.relu(self.bx16_dconvx1(bx16_outx2))
        bx16_score = self.bx16_score(bx16_outx1)

        bx8_outx4 = F.relu(self.bx8_dconvx4(out_8))
        bx8_outx2 = F.relu(self.bx8_dconvx2(bx8_outx4))
        bx8_outx1 = F.relu(self.bx8_dconvx1(bx8_outx2))
        bx8_score = self.bx8_score(bx8_outx1)

        # bx4_outx2 = self.alignScale(F.relu(self.bx4_dconvx2(out_4)), sizex2)
        # bx4_outx1 = self.alignScale(F.relu(self.bx4_dconvx1(bx4_outx2)), sizex1)
        # bx4_score = self.alignScale(self.bx4_score(bx4_outx1), sizex1)
        bx4_score = 0


        outx8 = self.subconv_to_8(out_up_16)
        outx4 = self.subconv_to_4(torch.cat([outx8, bx32_outx8, bx16_outx8], 1))
        outx2 = self.subconv_to_2(torch.cat([outx4, bx32_outx4, bx16_outx4, bx8_outx4], 1))
        outx1 = self.subconv_to_1_(torch.cat([outx2, bx32_outx2, bx16_outx2, bx8_outx2], 1))

        out_fs = self.fs_score_(torch.cat([bx32_score, bx16_score, bx8_score, outx1], 1))

        if self.isTrain:
            out_smth = self.smthBlock(out_2)
            out_smth = self.smthConv(out_smth)
            out_smth = self.smthUpsample(out_smth)

            return (outx1, out_fs, bx4_score, bx8_score, bx16_score, bx32_score,
                    outx2, outx4, outx8, out_smth)
        else:
            return (outx1, out_fs, bx4_score, bx8_score, bx16_score, bx32_score,
                    outx2, outx4, outx8)
        # return outx2, outx4, outx8, bx8_score, bx16_score, bx32_score

    def computeLoss(self, targets, predictions, with_mask=False, with_smth=False):
        criterion = nn.MSELoss(size_average=True)

        if with_mask:
            mask, maskx2, maskx4, maskx8 = targets[4], targets[5], targets[6], targets[7]
            lossx1 = criterion(predictions[0]*mask, targets[0]*mask)
            fs_loss = criterion(predictions[1]*mask, targets[0]*mask)
            # bx4_loss = criterion(predictions[2]*mask, targets[0]*mask)
            bx8_loss = criterion(predictions[3]*mask, targets[0]*mask)
            bx16_loss = criterion(predictions[4]*mask, targets[0]*mask)
            bx32_loss = criterion(predictions[5]*mask, targets[0]*mask)

            lossx2 = criterion(predictions[6]*maskx2, targets[1]*maskx2)
            lossx4 = criterion(predictions[7]*maskx4, targets[2]*maskx4)
            lossx8 = criterion(predictions[8]*maskx8, targets[3]*maskx8)
        else:
            lossx1 = criterion(predictions[0], targets[0])
            fs_loss = criterion(predictions[1], targets[0])
            # bx4_loss = criterion(predictions[2], targets[0])
            bx8_loss = criterion(predictions[3], targets[0])
            bx16_loss = criterion(predictions[4], targets[0])
            bx32_loss = criterion(predictions[5], targets[0])

            lossx2 = criterion(predictions[6], targets[1])
            lossx4 = criterion(predictions[7], targets[2])
            lossx8 = criterion(predictions[8], targets[3])

        if with_smth:
            SmthLoss = NerighborSmthLoss(lamda=0.01, t=2)
            smthTerm = SmthLoss(predictions[0], predictions[9])
            mainTerm = fs_loss + 0.5*(bx8_loss+bx16_loss+bx32_loss) + lossx1 + lossx2/2 + lossx4/4 + lossx8/8
            # print('smooth term: %f, main termL: %f'%(smthTerm.data[0], mainTerm.data[0]))
            loss = smthTerm + mainTerm
        else:
            loss = fs_loss + 0.5*(bx8_loss+bx16_loss+bx32_loss) + lossx1 + lossx2/2 + lossx4/4 + lossx8/8
        return loss


class NerighborSmthLoss(_Loss):
    def __init__(self, size_average=True, lamda=0.01, t=2):
        super(NerighborSmthLoss, self).__init__(size_average)
        self.lamda = lamda
        self.t = t

    def forward(self, input, target):
        predict = input
        smthMap = target

        horRelDiffMap = smthMap[:,:,:,0:-1] - smthMap[:,:,:,1:]
        verRelDiffMap = smthMap[:,:,0:-1,:] - smthMap[:,:,1:,:]
        horDpDiffMap = predict[:,:,:,0:-1] - predict[:,:,:,1:]
        verDpDiffMap = predict[:,:,0:-1,:] - predict[:,:,1:,:]

        horSmthLoss = torch.sum((horDpDiffMap**2)* torch.exp(-self.t*horRelDiffMap**2)).mean()
        verSmthLoss = torch.sum((verDpDiffMap**2)* torch.exp(-self.t*verRelDiffMap**2)).mean()

        return self.lamda/2*(horSmthLoss+verSmthLoss)


class DFCN_PS(nn.Module):
    """docstring for DFCN_PS"""
    def __init__(self, is_Train=True):
        super(DFCN_PS, self).__init__()
        self.isTrain = is_Train
        
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.denseBlock1 = DenseBlock(inputDim=64, outputDim=128 ,growthRate=32, blockDepth=6)
        self.pooling_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.denseBlock2 = DenseBlock(inputDim=128, outputDim=256 ,growthRate=32, blockDepth=12)
        self.pooling_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.denseBlock3 = DenseBlock(inputDim=256, outputDim=512 ,growthRate=32, blockDepth=24)
        self.pooling_4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_fc_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=False)
        self.drop_5 = nn.Dropout2d(p=0.2)
        self.conv_fc_5_2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0, bias=False)
        self.drop_6 = nn.Dropout2d(p=0.2)

        self.score_32 = nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=3, padding=1, bias=False)

        self.branch_score_16 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.branch_score_8 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.branch_score_4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False)
        
        self.subconv_to_8_ = UpsampleByPS(2, (64, 32))
        self.subconv_to_4_ = UpsampleByPS(2, (1, 32))
        self.subconv4x_ = UpsampleByPS(4, (1, 32), is_out_layer=True)

        self.initParameters()

    def setTrainMode(self, isTrain):
        self.isTrain = isTrain

    def initParameters(self):
        stateDict = self.state_dict()
        nn.init.xavier_normal(stateDict['conv_1.weight'])
        # nn.init.constant(stateDict['conv_1.bias'], 0)
        nn.init.xavier_normal(stateDict['conv_fc_5_1.weight'])
        # nn.init.constant(stateDict['conv_fc_5_1.bias'], 0)
        nn.init.xavier_normal(stateDict['conv_fc_5_2.weight'])
        # nn.init.constant(stateDict['conv_fc_5_2.bias'], 0)
        nn.init.xavier_normal(stateDict['score_32.weight'])
        # nn.init.constant(stateDict['score_32.bias'], 0)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(self.bn_1(out))

        out_4 = self.pooling_1(out)
        out_8 = self.pooling_2(F.relu(self.denseBlock1(out_4)))
        out_16 = self.pooling_3(F.relu(self.denseBlock2(out_8)))
        out_32 = self.pooling_4(F.relu(self.denseBlock3(out_16)))
        # if self.isTrain:
        #     out.volatile = False

        out_32 = F.relu(self.drop_5(self.conv_fc_5_1(out_32)))
        out_32 = F.relu(self.drop_6(self.conv_fc_5_2(out_32)))

        out_up_16 = nn.functional.upsample(out_32, size=(15,20), mode='bilinear')
        score_16 = F.relu(self.score_32(out_up_16))

        score_b_16 = self.branch_score_16(out_16)
        score_b_8 = self.branch_score_8(out_8)
        score_b_4 = self.branch_score_4(out_4)

        outx8 = self.subconv_to_8_([score_16, score_b_16])
        outx4 = self.subconv_to_4_([outx8, score_b_8])
        outx1 = self.subconv4x_([outx4, score_b_4])

        return outx1, outx4, outx8


class DFCN_32(nn.Module):
    """docstring for DFCN_32"""
    def __init__(self, is_Train=True):
        super(DFCN_32, self).__init__()
        self.isTrain = is_Train
        
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu1 =  nn.ReLU(inplace=True)
        self.pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.denseBlock1 = DenseBlock(inputDim=64, outputDim=128 ,growthRate=32, blockDepth=6)
        self.relu2 =  nn.ReLU(inplace=True)
        self.pooling_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.denseBlock2 = DenseBlock(inputDim=128, outputDim=256 ,growthRate=32, blockDepth=12)
        self.relu3 =  nn.ReLU(inplace=True)
        self.pooling_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.denseBlock3 = DenseBlock(inputDim=256, outputDim=512 ,growthRate=32, blockDepth=24)
        self.relu4 =  nn.ReLU(inplace=True)
        self.pooling_4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_fc_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=False)
        self.relu5 =  nn.ReLU(inplace=True)
        self.drop_5 = nn.Dropout2d(p=0.2)
        self.conv_fc_5_2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0, bias=False)
        self.relu6 =  nn.ReLU(inplace=True)
        self.drop_6 = nn.Dropout2d(p=0.2)

        self.score_32 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.relu7 =  nn.ReLU(inplace=True)
        # self.upsample_to_16 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
        #     nn.PixelShuffle(2),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.upsample_to_8 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample_to_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample4x = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.initParameters()

    def setTrainMode(self, isTrain):
        self.isTrain = isTrain

    def initParameters(self):
        stateDict = self.state_dict()
        nn.init.xavier_normal(stateDict['conv_1.weight'])
        # nn.init.constant(stateDict['conv_1.bias'], 0)
        nn.init.xavier_normal(stateDict['conv_fc_5_1.weight'])
        # nn.init.constant(stateDict['conv_fc_5_1.bias'], 0)
        nn.init.xavier_normal(stateDict['conv_fc_5_2.weight'])
        # nn.init.constant(stateDict['conv_fc_5_2.bias'], 0)
        nn.init.xavier_normal(stateDict['score_32.weight'])
        # nn.init.constant(stateDict['score_32.bias'], 0)

        # nn.init.xavier_normal(stateDict['upsample_to_16.0.weight'])
        nn.init.xavier_normal(stateDict['upsample_to_8.0.weight'])
        nn.init.xavier_normal(stateDict['upsample_to_4.0.weight'])
        nn.init.xavier_normal(stateDict['upsample4x.0.weight'])

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu1(out)
        out = self.pooling_1(out)

        out = self.denseBlock1(out)
        out = self.relu2(out)
        out = self.pooling_2(out)

        out = self.denseBlock2(out)
        out = self.relu3(out)
        out = self.pooling_3(out)
        out = self.denseBlock3(out)
        out = self.relu4(out)
        out = self.pooling_4(out)
        # if self.isTrain:
        #     out.volatile = False

        out = self.conv_fc_5_1(out)
        out = self.drop_5(out)
        out = self.relu5(out)
        out = self.conv_fc_5_2(out)
        out = self.drop_6(out)
        out = self.relu6(out)
        out = nn.functional.upsample(out, size=(15,20), mode='bilinear')

        out = self.score_32(out)
        out = self.relu7(out)

        # out = self.upsample_to_16(out)

        # outSize = out.size()
        # marginLeft = Variable(torch.zeros(outSize[0], outSize[1], 1, outSize[3]))
        # # marginTop = Variable(torch.zeros(outSize[0], outSize[1], outSize[2]+1, 1))
        # if out.is_cuda:
        #     marginLeft = marginLeft.cuda()
        #     # marginTop = marginTop.cuda()
        # out = torch.cat([out, marginLeft], 2)

        outx8 = self.upsample_to_8(out)
        outx4 = self.upsample_to_4(outx8)
        outx1 = self.upsample4x(outx4)

        return outx1, outx4, outx8


class DFCN_16(DFCN_32):
    """docstring for DFCN_16"""
    def __init__(self):
        super(DFCN_16, self).__init__()

        self.score_16 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample_to_8 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = self.pooling_1(out)

        out = F.relu(self.denseBlock1(out))
        out = self.pooling_2(out)
        out = F.relu(self.denseBlock2(out))
        out = self.pooling_3(out)
        out_16 = out
        out_16 = self.score_16(out_16)
        out_16 = self.upsample(out_16)

        out = self.denseBlock3(out)
        out = self.relu4(out)
        out = self.pooling_4(out)

        out = self.conv_fc_5_1(out)
        out = self.relu5(out)
        out = self.conv_fc_5_2(out)
        out = self.relu6(out)
        out = self.score_32(out)
        out = self.relu7(out)

        out = self.upsample_to_16(out)
        out_cat = torch.cat([out, out_16], 1)
        out_cat = self.upsample_to_8(out_cat)
        out_cat = self.upsample_to_4(out_cat)
        out_cat = self.upsample4x(out_cat)

        return out_cat

class RDCN_VGG(nn.Module):
    def __init__(self, rec_num):
        super(RDCN_VGG, self).__init__()

        self.recNum = rec_num
        self.downsample = nn.Sequential(OrderedDict([
            ('data/bn', nn.BatchNorm2d(3)),
            ('conv1_1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)),
            ('conv1_1/bn', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ('conv1_2/bn', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ('conv2_1/bn', nn.BatchNorm2d(128)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)),
            ('conv2_2/bn', nn.BatchNorm2d(128)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_1/bn', nn.BatchNorm2d(256)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_2/bn', nn.BatchNorm2d(256)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_3/bn', nn.BatchNorm2d(256)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('conv3_4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('conv3_4/bn', nn.BatchNorm2d(256)),
            ('relu3_4', nn.ReLU(inplace=True))
        ]))

        self.denseBlock = DenseBlock(inputDim=256, outputDim=256 ,growthRate=32, blockDepth=8)
        self.predictx4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        self.weightedAvg = nn.Conv2d(in_channels=self.recNum, out_channels=1, kernel_size=1, bias=True)

        self.upsample4x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )  
        self.predict = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False)

    def loadConv(self, pretrain_model):
        pretrainModel = sio.loadmat(pretrain_model)
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name = name.split('.')[-1]
                if module.bias is not None:
                    for key, value in pretrainModel.items():
                        if '%s_0'%last_name == key:  # for weight
                            print('load %s'%key)
                            self.copyArrayToTensor(value, module.weight.data)

                        if '%s_1'%last_name == key:  # for weight
                            print('load %s'%key)
                            self.copyArrayToTensor(value, module.bias.data)
                else:
                    for key, value in pretrainModel.items():
                        if '%s_0'%last_name == key:  # for weight
                            print('load %s'%key)
                            self.copyArrayToTensor(value, module.weight.data)


    def copyArrayToTensor(self, array, tensor):
        aShape = array.shape
        tShape = tensor.shape
        
        if len(aShape) == 2 and aShape[0] == 1:
            array = np.squeeze(array)
            aShape = array.shape

        if len(aShape) != len(tShape):
            raise ValueError('array shape:{} mismatches with tensor: {}'.format(aShape, tShape))

        for indx in range(len(aShape)):
            if aShape[indx] != tShape[indx]:
                raise ValueError('array shape:{} mismatches with tensor: {}'.format(aShape, tShape))

        if len(aShape) == 1:
            for n in range(aShape[0]):
                tensor[n] = float(array[n])
        elif len(aShape) == 2:
            for n in range(aShape[0]):
                for c in range(aShape[1]):
                    tensor[n, c] = float(array[n, c])
        elif len(aShape) == 3:
            for n in range(aShape[0]):
                for c in range(aShape[1]):
                    for h in range(aShape[2]):
                        tensor[n, c, h] = float(array[n, c, h])
        elif len(aShape) == 4:
            for n in range(aShape[0]):
                for c in range(aShape[1]):
                    for h in range(aShape[2]):
                        for w in range(aShape[3]):
                            tensor[n, c, h, w] = float(array[n, c, h, w])


    def forward(self, x):
        out = self.downsample(x)
        predictx4s = [None for i in range(self.recNum)]
        catFlag = False
        predictx4Cat = None
        predict_final = None

        # input("RDCN_VGG before loop")
        for indx in range(self.recNum):
            out = self.denseBlock(out)
            predictx4s[indx] = self.predictx4(out)
            if not catFlag:
                catFlag = True
                predictx4Cat = predictx4s[indx]
            else:
                predictx4Cat = torch.cat([predictx4Cat, predictx4s[indx]], 1)
            # print(predictx4s[indx])

        predictx4_avg = self.weightedAvg(predictx4Cat)
        # print('-- avg\n', predictx4_avg)

        out = self.upsample4x(out)
        predict_final = self.predict(out)

        return predictx4s, predictx4_avg, predict_final

class InvLoss(nn.Module):
    def __init__(self, lamda=0.5):
        super(InvLoss, self).__init__()
        self.lamda = lamda

    def forward(self, _input, _target):
        dArr = _input - _target
        nVal = _input.data.shape[2]*_input.data.shape[3]

        mseLoss = torch.sum(torch.sum(dArr*dArr, 2), 3)/nVal
        dArrSum = torch.sum(torch.sum(dArr, 2), 3)
        mssLoss = -self.lamda*(dArrSum*dArrSum)/(nVal**2)

        loss = mseLoss + mssLoss
        loss = torch.sum(loss)
        return loss


def copyArrayToTensor(array, tensor):
    aShape = array.shape
    tShape = tensor.shape
    
    if len(aShape) == 2 and aShape[0] == 1:
        array = np.squeeze(array)
        aShape = array.shape

    if len(aShape) != len(tShape):
        raise ValueError('array shape:{} mismatches with tensor: {}'.format(aShape, tShape))

    for indx in range(len(aShape)):
        if aShape[indx] != tShape[indx]:
            raise ValueError('array shape:{} mismatches with tensor: {}'.format(aShape, tShape))

    if len(aShape) == 1:
        for n in range(aShape[0]):
            tensor[n] = float(array[n])
    elif len(aShape) == 2:
        for n in range(aShape[0]):
            for c in range(aShape[1]):
                tensor[n, c] = float(array[n, c])
    elif len(aShape) == 3:
        for n in range(aShape[0]):
            for c in range(aShape[1]):
                for h in range(aShape[2]):
                    tensor[n, c, h] = float(array[n, c, h])
    elif len(aShape) == 4:
        for n in range(aShape[0]):
            for c in range(aShape[1]):
                for h in range(aShape[2]):
                    for w in range(aShape[3]):
                        tensor[n, c, h, w] = float(array[n, c, h, w])


def copyParametersToModel(params, modules, rule_file):
    ruleDict = dict()
    ruleFile = open(rule_file, 'r')
    line = ruleFile.readline()
    while line != '' and line != '\n':
        contents = line.split(' ')
        currSrcLayer = contents[0]
        if contents[1][-1] == '\n':
            currTargetLayer = contents[1][:-1]
        else:
            currTargetLayer = contents[1]

        if currSrcLayer in params.keys():
            ruleDict[currSrcLayer] = currTargetLayer
        else:
            raise ValueError('pretrainModel has no key: %s'%currSrcLayer)
        line = ruleFile.readline()

    ruleFile.close()

    # load parameters
    for key, item in ruleDict.items():
        copyArrayToTensor(params[key], modules[item])

