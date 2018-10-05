import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tools.EvalutateMetrics import myMetrics
from datasets import NYUv2DataSet, NYUv2FusionSet

import scipy.io as sio
import numpy as np
from PIL import Image
import glob

import time

def loadImage(test_file, in_size=[240, 320]):
    data = sio.loadmat(test_file)
    data = data['data']

    rgb = data['rgb'][0,0]
    depth = data['depth'][0,0]
    depthx2 = data['depthx2'][0,0]
    depthx4 = data['depthx4'][0,0]
    depthx8 = data['depthx8'][0,0]
    imageSize = data['imageSize'][0,0][0]
    offset_x = int((imageSize[0] - in_size[0])/2)
    offset_y = int((imageSize[1] - in_size[1])/2)

    rgb_new = rgb.transpose((2, 0, 1))
    rgb_new = torch.from_numpy(rgb_new[np.newaxis,:,offset_x:in_size[0]+offset_x, offset_y:in_size[1]+offset_y]).float()
    depth_new = depth[offset_x:in_size[0]+offset_x, offset_y:in_size[1]+offset_y]
    depthx2_new = depthx2[int(offset_x/2):120+int(offset_x/2), int(offset_y/2):160+int(offset_y/2)]
    depthx4_new = depthx4[int(offset_x/4):60+int(offset_x/4), int(offset_y/4):80+int(offset_y/4)]
    depthx8_new = depthx8[int(offset_x/8):30+int(offset_x/8), int(offset_y/8):40+int(offset_y/8)]

    depth_target = (np.exp(depth_new), np.exp(depthx2_new), np.exp(depthx4_new), np.exp(depthx8_new))

    depth_new = np.exp(depth_new)
    rgb_new = rgb_new.cuda()

    inputData = Variable(rgb_new)
    inputData.volatile = True

    return inputData, depth_target


def covert2Array(inData):
    cpuData = inData.cpu()
    return np.exp( cpuData.data[0].numpy()[0,...].astype(np.float32) )


parser = argparse.ArgumentParser(description="pythorch recusive densely-connected nerual network Test")
parser.add_argument("--model", default=None, type=str, help="model path")
parser.add_argument("--image", default=None, type=str, help="image name")
# parser.add_argument("--cpu", action="store_true", help="Use cpu only")
parser.add_argument("--data", default='', type=str, help='assign dataset for test. when assinged, --image become useless')

opt = parser.parse_args()
print(opt)

print('build model...')
model = torch.load(opt.model)["model"]
model.setTrainMode(False)
model.eval()

model = model.cuda()
model.is_train = False

# print(model)

metrics = myMetrics()
metrics.resetMetrics()

if opt.data:
    dataFiles = glob.glob('%s/*.mat'%opt.data)
    dataNum = len(dataFiles)

    for indx in range(min(dataNum,700)):
        inputData, target = loadImage(dataFiles[indx])
        predictions = model(inputData)

        begin = time.time()
        predictions = model(inputData)
        end = time.time()
        # print(end-begin)

        if indx <= 1:
            detectTime = end-begin
        else:
            detectTime = detectTime + end-begin

        predictedx1 = predictions[0].cpu()
        predictedx1_np = covert2Array(predictedx1)

        metrics.computeMetrics(predictedx1_np, target[0], disp=True, image_name=dataFiles[indx])

    metricsVals = metrics.getMetrics()

    print('-- [average metrics] -------')
    print('rel: %f, log10: %f, rms: %f, thr1: %f, thr2: %f, thr3: %f'%(metricsVals[0],metricsVals[1], 
            metricsVals[2], metricsVals[3], metricsVals[4], metricsVals[5]))
    print('average time: %f'%(detectTime/float(dataNum-1)) ) 

else:
    test_file = opt.image
    data = sio.loadmat(test_file)
    data = data['data']

    rgb = data['rgb'][0,0]

    inputData, targets = loadImage(test_file)

    # out of the network
    predictions = model(inputData)
    predictedx1 = covert2Array(predictions[0])
    O_8x, O_16x, O_32x = predictions[3:6]
    pred2x, pred4x, pred8x = predictions[6:]
    O_4x = 0 # No O_4x for this version
    O_8x = covert2Array(O_8x)
    O_16x = covert2Array(O_16x)
    O_32x = covert2Array(O_32x)
    pred2x = covert2Array(pred2x)
    pred4x = covert2Array(pred4x)
    pred8x = covert2Array(pred8x)

    currRel = metrics.computeRel(predictedx1, targets[0])
    currRMS = metrics.computeRMS(predictedx1, targets[0])
    currL10 = metrics.computeLog10(predictedx1, targets[0])

    print('rel: %f, rms: %f, log10: %f'%(currRel, currRMS, currL10))

    sio.savemat('results.mat', {'rgb': rgb, 'depth':targets[0],
        'pred1x': predictedx1, 'pred2x': pred2x, 'pred4x': pred4x,
        'pred8x': pred8x, 'bx4': O_4x,'bx8': O_8x, 'bx16': O_16x, 'bx32': O_32x})

print('Done!')