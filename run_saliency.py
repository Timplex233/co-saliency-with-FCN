from __future__ import division
import numpy as np
import sys
caffe_root = '/home/timplex/caffe_dss/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

base_weights = 'ours_TRI_iter_36000.caffemodel'  # the vgg16 model

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

#print solver.net.blobs['data1'].data.shape
#print solver.net.blobs['gt1'].data.shape
#print solver.net.blobs['label1'].data.shape
#print solver.net.blobs['upscore_1'].data.shape
#print solver.net.blobs['conv5_3_f'].data.shape
# copy base weights for fine-tuning
#solver.restore('./snapshot/ours_iter_4000.solverstate')
'''print solver.net.blobs['data1'].data.shape
#print solver.net.blobs['gt1'].data.shape
print solver.net.blobs['label1'].data.shape
print solver.net.blobs['conv1_1'].data.shape
print solver.net.blobs['conv1_2'].data.shape
print solver.net.blobs['pool1'].data.shape
print solver.net.blobs['conv2_1'].data.shape
print solver.net.blobs['conv2_2'].data.shape
print solver.net.blobs['pool2'].data.shape
print solver.net.blobs['conv3_1'].data.shape
print solver.net.blobs['conv4_1'].data.shape
print solver.net.blobs['conv4_3'].data.shape
print solver.net.blobs['pool4'].data.shape
#print solver.net.blobs['conv_1'].data.shape
#print solver.net.blobs['conv_2'].data.shape
#print solver.net.blobs['conv_3'].data.shape
#print solver.net.blobs['concat_g1'].data.shape
print 'conv_f1_1',solver.net.blobs['conv_f1_1'].data.shape
print 'conv_f2_1',solver.net.blobs['conv_f2_1'].data.shape
print 'upsample32_1',solver.net.blobs['score_1'].data.shape
print 'upscore_1',solver.net.blobs['upscore_1'].data.shape'''
#exit()
solver.net.copy_from(base_weights)
solver.step(36000)
