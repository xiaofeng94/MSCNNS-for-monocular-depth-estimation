### used to parse parameters in .caffemodel to a dict
import sys
caffe_root = '/media/xiaofeng/codes/LinuxFiles/caffe'
sys.path.insert(0, caffe_root + '/python')

import caffe

import os
import numpy as np

caffe_cfg = './model/DenseNet_121.prototxt'
caffemodel_path = './model/DenseNet_121.caffemodel'

target_root = './model/pretrain'
target_path = '%s/densenet_121'%(target_root)
if not os.path.exists(target_root):
   os.makedirs(target_root)


caffe.set_mode_cpu() 
net = caffe.Net(caffe_cfg, caffemodel_path, caffe.TEST)

indxStr = ['weight', 'bias', '3']
model_dict = dict()
for key in net.params.keys():
    if 'conv' in key:
        print('-- {}, len: {}'.format(key, len(net.params[key])))
        for indx in range(len(net.params[key])):
            currKey = '%s.%s'%(key, indxStr[indx])
            shape = net.params[key][indx].data.shape
            data = net.params[key][indx].data

            model_dict[currKey] = data

    # if key == 'conv1_1':
    #     print(net.params[key][0].data)
    #     print(net.params[key][1].data)

import scipy.io as sio

# sio.savemat('conv1_1', {'conv1_1': model_dict['conv1_1'][0]})
print('save model to %s'%target_path)
sio.savemat(target_path, model_dict)