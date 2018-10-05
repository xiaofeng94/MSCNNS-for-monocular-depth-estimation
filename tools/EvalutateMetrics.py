import numpy as np

class myMetrics(object):
    """docstring for myMetrics"""
    def __init__(self):
        super(myMetrics, self).__init__()

        self.rel = 0 # average relative error
        self.rms = 0 # root mean squared error
        self.log10Err = 0 # average log10 error
        self.thrAcc1 = 0 # accuracy with threshold
        self.thrAcc2 = 0 # accuracy with threshold
        self.thrAcc3 = 0 # accuracy with threshold
        self.thrCount1 = 0
        self.thrValue1 = 1.25
        self.thrCount2 = 0
        self.thrValue2 = 1.25**2
        self.thrCount3 = 0
        self.thrValue3 = 1.25**3

        self.pointsNum = 0 # valid point number
        self.exclude_list = []
        self.exclude_thr = 10 #0.7 for make3d #0.4 for nyn2
        self.rms_avg_by_image = True
        self.min_depth = 0.7
        self.max_depth = 99999
        self.test_count = 0

        
    def fineModification(self, depth_predicted, depth_gt, max_depth = 70, clip_value = 80):
        # fine modification
        for indx_x in np.arange(depth_gt.shape[0]):                                                                                   
            for indx_y in np.arange(depth_gt.shape[1]):
                if depth_predicted[indx_x, indx_y] > max_depth: #70
                    depth_predicted[indx_x, indx_y] = clip_value #80

    def computeRel(self, depth_predicted, depth_gt):
        number = depth_gt.shape[0]*depth_gt.shape[1]
        curr_rel_arr = np.abs(depth_predicted-depth_gt)/depth_gt
        curr_rel = np.sum(curr_rel_arr)/number

        return curr_rel

    def computeRMS(self, depth_predicted, depth_gt):
        number = depth_gt.shape[0]*depth_gt.shape[1]
        curr_rms_arr = (depth_predicted-depth_gt)**2
        curr_rms = np.sum(curr_rms_arr)/number

        return np.sqrt(curr_rms)

    def computeLog10(self, depth_predicted, depth_gt):
        number = depth_gt.shape[0]*depth_gt.shape[1]
        curr_log10Err_arr = np.abs(np.log10(depth_predicted)-np.log10(depth_gt))
        curr_log10Err = np.sum(curr_log10Err_arr)/number

        return curr_log10Err

    def setMetricsType(self, typeStr = 'c1'):
        if typeStr == 'c1':
            self.max_depth = 70
        else:
            self.max_depth = 99999

    def resetMetrics(self):
        self.test_count = 0

        self.rel = list() # average relative error
        self.rms2 = list() # root mean squared error
        self.log10Err = list() # average log10 error
        self.thrAcc1 = list() # accuracy with threshold
        self.thrAcc2 = list() # accuracy with threshold
        self.thrAcc3 = list() # accuracy with threshold

        self.thrCount1 = 0
        self.thrCount2 = 0
        self.thrCount3 = 0

        self.pointsNum = 0 # valid point number
        self.exclude_list = []
        self.test_count = 0

    def fastCompute(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25   ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse2 = np.mean((gt - pred) ** 2)

        log10_err = np.mean(np.absolute(np.log10(gt) - np.log10(pred)))
                        
        abs_rel = np.mean(np.abs(gt - pred) / gt)
                                           
        return abs_rel, rmse2, log10_err, a1, a2, a3


    def computeMetrics(self, pred_depth_real, depth_real, disp=False, image_name=''):
        self.test_count += 1

        mask = np.logical_and(depth_real>self.min_depth, depth_real<self.max_depth)

        currMetrics = self.fastCompute(depth_real[mask], pred_depth_real[mask])
        self.rel.append(currMetrics[0])
        self.rms2.append(currMetrics[1])
        self.log10Err.append(currMetrics[2])
        self.thrAcc1.append(currMetrics[3])
        self.thrAcc2.append(currMetrics[4])
        self.thrAcc3.append(currMetrics[5])

        if currMetrics[0] > self.exclude_thr:
            self.exclude_list.append(self.test_count)

        if disp:
            print('({}){}:'.format(self.test_count, image_name))
            print('rel: {}, rms: {}, log10: {}'.format(currMetrics[0], np.sqrt(currMetrics[1]), currMetrics[2]))
            print('---- file end ----')

        
    def getMetrics(self):

        rel = np.array(self.rel).mean()
        log10Err = np.array(self.log10Err).mean()
        rms = np.sqrt(self.rms2).mean()
        a1 = np.array(self.thrAcc1).mean()
        a2 = np.array(self.thrAcc2).mean()
        a3 = np.array(self.thrAcc3).mean()

        return rel,log10Err,rms,a1,a2,a3, self.exclude_list