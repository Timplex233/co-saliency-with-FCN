import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import matplotlib
import scipy.misc
from PIL import Image
import scipy.io
import os


# Make sure that caffe is on the python path:
caffe_root = '/home/timplex/caffe_dss/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

EPSILON = 1e-8

#data_root = '/home/timplex/dataset/saliency/MSRA-100/'
with open('lists/imgs1.lst') as f:
    test_lst = f.readlines()

test_lst = [x.split(' ')[0] for x in test_lst]

with open('lists/gt1.lst') as f:
    output_lst = f.readlines()
output_lst = [x.split(' ')[0] for x in output_lst]


#remove the following two lines if testing with cpu
caffe.set_mode_cpu()
# choose which GPU you want to use
#caffe.set_device(0)
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('DSS_deploy.prototxt', 'dss_model_released.caffemodel', caffe.TEST)

def plot_saveimg(id,out):
   # plt.figure()
   # plt.imshow(out)
   # plt.savefig('output_img/test2.jpg')
  #  matplotlib.image.imsave(output_lst[id], out, cmap = cm.Greys_r)
    new_im = Image.fromarray(out*255).convert('L')
    new_im.save(output_lst[id])
    print id, out.shape

#for id in range(38): os.mkdir('/home/timplex/dataset/cosaliency/iCoseg/singlegt/'+str(id+1)+'/')
for idx in range(0,len(test_lst)):
    # load image
    img = Image.open(test_lst[idx])
    img = np.array(img, dtype=np.uint8)
    im = np.array(img, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    # run net and take argmax for prediction
    net.forward()

    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    out6 = net.blobs['sigmoid-dsn6'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    res = (out3 + out4 + out5 + fuse) / 4
    res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
    #plot_single_scale(out_lst, name_lst, 10)
    plot_saveimg(idx,res)
