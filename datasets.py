import torch.utils.data as data
import torch

import glob
import scipy.io as sio
import numpy as np

# TODO: data argumentation
class NYUv2DataSet(data.Dataset):
    def __init__(self, data_root, is_train=True):
        super(NYUv2DataSet, self).__init__()

        self.dataRoot = data_root
        self.dataFiles = glob.glob('%s/*.mat'%self.dataRoot)
        self.dataNum = len(self.dataFiles)
        self.requiredSize = [240, 320]
        self.reqSizex4 = [60, 80]
        self.reqSizex8 = [30, 40]
        self.isTrain = is_train
        self.leastScale = 8

    def __getitem__(self, index):
        currFile = self.dataFiles[index]
        data = sio.loadmat(currFile)
        data = data['data']

        rgb = data['rgb'][0,0].transpose((2, 0, 1))
        depth = data['depth'][0,0]
        depthx4 = data['depthx4'][0,0]
        depthx8 = data['depthx8'][0,0]
        imageSize = data['imageSize'][0,0][0]

        if imageSize[0] < self.requiredSize[0] or imageSize[1] < self.requiredSize[1]:
            raise ValueError('input image size is smaller than [240, 320]')

        if self.isTrain:
            import random
            offset_x = random.randint(0, imageSize[0] - self.requiredSize[0]) // self.leastScale
            offset_y = random.randint(0, imageSize[1] - self.requiredSize[1]) // self.leastScale
        else:
            offset_x = int((imageSize[0] - self.requiredSize[0])/2) // self.leastScale
            offset_y = int((imageSize[1] - self.requiredSize[1])/2) // self.leastScale

        rgb = rgb[:, self.leastScale*offset_x:self.leastScale*offset_x+self.requiredSize[0],
                        self.leastScale*offset_y:self.leastScale*offset_y+self.requiredSize[1]]
                        
        depth = depth[np.newaxis, self.leastScale*offset_x:self.leastScale*offset_x+self.requiredSize[0],
                                        self.leastScale*offset_y:self.leastScale*offset_y+self.requiredSize[1]]

        depthx4 = depthx4[np.newaxis, 2*offset_x:2*offset_x+self.reqSizex4[0],
                                        2*offset_y:2*offset_y+self.reqSizex4[1]]
        depthx8 = depthx8[np.newaxis, offset_x:offset_x+self.reqSizex8[0],
                                        offset_y:offset_y+self.reqSizex8[1]]

        return torch.from_numpy(rgb).float(), torch.from_numpy(depth).float(), \
                torch.from_numpy(depthx4).float(), torch.from_numpy(depthx8).float(), currFile

    def __len__(self):
        return self.dataNum


class NYUv2FusionSet(data.Dataset):
    def __init__(self, data_root, is_train=True, rgb_norm=False):
        super(NYUv2FusionSet, self).__init__()

        self.rgb_norm = rgb_norm
        self.dataRoot = data_root
        self.dataFiles = glob.glob('%s/*.mat'%self.dataRoot)
        self.dataNum = len(self.dataFiles)
        self.requiredSize = [240, 320]
        self.reqSizex2 = [120, 160]
        self.reqSizex4 = [60, 80]
        self.reqSizex8 = [30, 40]
        ## for make3d
        # self.requiredSize = [230, 172]
        # self.reqSizex2 = [115, 86]
        # self.reqSizex4 = [57, 43]
        # self.reqSizex8 = [28, 21]
        self.isTrain = is_train
        self.leastScale = 8

    def __getitem__(self, index):
        currFile = self.dataFiles[index]
        data = sio.loadmat(currFile)
        data = data['data']

        rgb = data['rgb'][0,0].transpose((2, 0, 1))
        if self.rgb_norm:
            rgb = rgb/255.
        depth = data['depth'][0,0]
        depthx2 = data['depthx2'][0,0]
        depthx4 = data['depthx4'][0,0]
        depthx8 = data['depthx8'][0,0]
        imageSize = data['imageSize'][0,0][0]
 
        if imageSize[0] < self.requiredSize[0] or imageSize[1] < self.requiredSize[1]:
            raise ValueError('input image size is smaller than [240, 320]')

        if self.isTrain:
            import random
            offset_x = random.randint(0, imageSize[0] - self.requiredSize[0]) // self.leastScale
            offset_y = random.randint(0, imageSize[1] - self.requiredSize[1]) // self.leastScale
        else:
            offset_x = int((imageSize[0] - self.requiredSize[0])/2) // self.leastScale
            offset_y = int((imageSize[1] - self.requiredSize[1])/2) // self.leastScale

        rgb = rgb[:, self.leastScale*offset_x:self.leastScale*offset_x+self.requiredSize[0],
                        self.leastScale*offset_y:self.leastScale*offset_y+self.requiredSize[1]]
                        
        depth = depth[np.newaxis, self.leastScale*offset_x:self.leastScale*offset_x+self.requiredSize[0],
                                        self.leastScale*offset_y:self.leastScale*offset_y+self.requiredSize[1]]

        depthx2 = depthx2[np.newaxis, 4*offset_x:4*offset_x+self.reqSizex2[0],
                                        4*offset_y:4*offset_y+self.reqSizex2[1]]

        depthx4 = depthx4[np.newaxis, 2*offset_x:2*offset_x+self.reqSizex4[0],
                                        2*offset_y:2*offset_y+self.reqSizex4[1]]
        depthx8 = depthx8[np.newaxis, offset_x:offset_x+self.reqSizex8[0],
                                        offset_y:offset_y+self.reqSizex8[1]]

        return torch.from_numpy(rgb).float(), torch.from_numpy(depth).float(), \
                torch.from_numpy(depthx2).float(), torch.from_numpy(depthx4).float(), torch.from_numpy(depthx8).float(), \
                currFile

    def __len__(self):
        return self.dataNum


class NYUv2MaskSet(data.Dataset):
    def __init__(self, data_root, is_train=True, rgb_norm=False):
        super(NYUv2MaskSet, self).__init__()

        self.rgb_norm = rgb_norm
        self.dataRoot = data_root
        self.dataFiles = glob.glob('%s/*.mat'%self.dataRoot)
        self.dataNum = len(self.dataFiles)
        self.requiredSize = [240, 320]
        self.reqSizex2 = [120, 160]
        self.reqSizex4 = [60, 80]
        self.reqSizex8 = [30, 40]
        self.isTrain = is_train
        self.leastScale = 8

    def __getitem__(self, index):
        currFile = self.dataFiles[index]

        # print('load %s'%currFile)
        data = sio.loadmat(currFile)
        data = data['data']

        rgb = data['rgb'][0,0].transpose((2, 0, 1))
        if self.rgb_norm:
            rgb = rgb/255.
        depth = data['depth'][0,0]
        depthx2 = data['depthx2'][0,0]
        depthx4 = data['depthx4'][0,0]
        depthx8 = data['depthx8'][0,0]
        mask = data['dpMask'][0,0]
        maskx2 = data['dpMaskx2'][0,0]
        maskx4 = data['dpMaskx4'][0,0]
        maskx8 = data['dpMaskx8'][0,0]
        imageSize = data['imageSize'][0,0][0]
 
        if imageSize[0] < self.requiredSize[0] or imageSize[1] < self.requiredSize[1]:
            raise ValueError('input image size is smaller than [240, 320]')

        if self.isTrain:
            import random
            offset_x = random.randint(0, imageSize[0] - self.requiredSize[0]) // self.leastScale
            offset_y = random.randint(0, imageSize[1] - self.requiredSize[1]) // self.leastScale
        else:
            offset_x = int((imageSize[0] - self.requiredSize[0])/2) // self.leastScale
            offset_y = int((imageSize[1] - self.requiredSize[1])/2) // self.leastScale

        rgb = rgb[:, self.leastScale*offset_x:self.leastScale*offset_x+self.requiredSize[0],
                        self.leastScale*offset_y:self.leastScale*offset_y+self.requiredSize[1]]
                        
        depth = depth[np.newaxis, self.leastScale*offset_x:self.leastScale*offset_x+self.requiredSize[0],
                                        self.leastScale*offset_y:self.leastScale*offset_y+self.requiredSize[1]]

        depthx2 = depthx2[np.newaxis, 4*offset_x:4*offset_x+self.reqSizex2[0],
                                        4*offset_y:4*offset_y+self.reqSizex2[1]]

        depthx4 = depthx4[np.newaxis, 2*offset_x:2*offset_x+self.reqSizex4[0],
                                        2*offset_y:2*offset_y+self.reqSizex4[1]]
        depthx8 = depthx8[np.newaxis, offset_x:offset_x+self.reqSizex8[0],
                                        offset_y:offset_y+self.reqSizex8[1]]

        mask = mask[np.newaxis, self.leastScale*offset_x:self.leastScale*offset_x+self.requiredSize[0],
                                        self.leastScale*offset_y:self.leastScale*offset_y+self.requiredSize[1]]

        maskx2 = maskx2[np.newaxis, 4*offset_x:4*offset_x+self.reqSizex2[0],
                                        4*offset_y:4*offset_y+self.reqSizex2[1]]

        maskx4 = maskx4[np.newaxis, 2*offset_x:2*offset_x+self.reqSizex4[0],
                                        2*offset_y:2*offset_y+self.reqSizex4[1]]
        maskx8 = maskx8[np.newaxis, offset_x:offset_x+self.reqSizex8[0],
                                        offset_y:offset_y+self.reqSizex8[1]]

        return torch.from_numpy(rgb).float(), torch.from_numpy(depth).float(), \
                torch.from_numpy(depthx2).float(), torch.from_numpy(depthx4).float(), torch.from_numpy(depthx8).float(), \
                currFile, torch.from_numpy(mask).float(), torch.from_numpy(maskx2).float(), \
                torch.from_numpy(maskx4).float(), torch.from_numpy(maskx8).float()

    def __len__(self):
        return self.dataNum