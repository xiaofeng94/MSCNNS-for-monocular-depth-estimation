# Introduction for MSCNNS
MSCN<sub>NS</sub> (Multi-scale Sub-pixel Convolutional Network with a Neighborhood Smoothness constraint) is a CNN-based approach for monocular depth estimation.

For technical details, please see this [paper](https://ieeexplore.ieee.org/document/8624409) (comming soon).

## Prerequisites
* Matlab R2017a (or other proper version)
* python v3.5.x
* pytorch v0.3.0 (or later version)
* numpy
* scipy

## How to test

### Quick test

You may use the provided model (see the BaiduYun link below) and test samples to test this apporach as follows,

`python3 test.py --model <the pytorch model> --image ./test_samples/nyu_v2_175.mat`

### Test on the whole NYU Depth v2 dataset.

1. Download [The Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [The Train/Test Split file](https://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html).
2. Suppose you have saved the dataset in <path_to_data> and the split file in <path_to_split>. Open ```matlab/gen_test_data_for_mscn.m``` and assign <path_to_data> to 'NYUv2_data' and <path_to_split> to 'split_file'.
3. run ```matlab/gen_test_data_for_mscn.m``` and the test data will be generated in '../Dataset/test'. You may change the save root 'test_root' to anywhere you like.
4. Download the model in the `BaiduYun disk (Link: https://pan.baidu.com/s/1U0hw58K2M0y5QE4c3hbNng password: qnv3)`
5. Test the model as follows,

`python3 test.py --model <the pytorch model> --data <folder for the generated test data>`

## Results

Note that you may find the references and more comparisons in the aforementioned paper.

### Quantitative results
<img src="https://github.com/xiaofeng94/MSCNNS-for-monocular-depth-estimation/blob/master/results/table_nyu.png" width="65%" height="65%" />

### Qualitative results
<img src="https://github.com/xiaofeng94/MSCNNS-for-monocular-depth-estimation/blob/master/results/figure_nyu.png" width="80%" height="80%" />

### Citation 
Please consider citing the following paper if the code is helpful in your research work:
<pre>
@ARTICLE{8624409, 
author={Shiyu Zhao and Lin Zhang and Ying Shen and Shengjie Zhao and Huijuan Zhang}, 
journal={IEEE Access}, 
title={Super-Resolution for Monocular Depth Estimation With Multi-Scale Sub-Pixel Convolutions and a Smoothness Constraint}, 
year={2019}, 
volume={7}, 
pages={16323-16335}
}
</pre>
