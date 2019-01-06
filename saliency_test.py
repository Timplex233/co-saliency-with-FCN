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

with open('lists/test1_1.lst') as f:
    test_lst = f.readlines()

gt_lst = [x.strip().split(' ')[1] for x in test_lst]
test_lst = [x.split(' ')[0] for x in test_lst]

with open('lists/test1_2.lst') as f:
    test_lst2 = f.readlines()

gt_lst2 = [x.strip().split(' ')[1] for x in test_lst2]
test_lst2 = [x.split(' ')[0] for x in test_lst2]

with open('lists/test1_3.lst') as f:
    test_lst3 = f.readlines()

gt_lst3 = [x.strip().split(' ')[1] for x in test_lst3]
test_lst3 = [x.split(' ')[0] for x in test_lst3]

'''test_lst=['/home/timplex/dataset/cosaliency/MSRC/images/4/4_8_s.bmp']
test_lst2=['/home/timplex/dataset/cosaliency/MSRC/images/4/4_23_s.bmp']
test_lst3=['/home/timplex/dataset/cosaliency/MSRC/images/4/4_21_s.bmp']
gt_lst=['/home/timplex/dataset/cosaliency/MSRC/groundtruth/4/4_8_s.bmp']
gt_lst2=['/home/timplex/dataset/cosaliency/MSRC/groundtruth/4/4_23_s.bmp']
gt_lst3=['/home/timplex/dataset/cosaliency/MSRC/groundtruth/4/4_21_s.bmp']'''

'''
with open('lists/gt1.lst') as f:
    single_lst = f.readlines()
single_lst = [x.split(' ')[0] for x in single_lst]
data_root = '/home/timplex/dataset/saliency/MSRA-100/'
with open('../DSS/lists/test.lst') as f:
    test_lst = f.readlines()

test_lst = [data_root + x.strip() for x in test_lst]'''


#remove the following two lines if testing with cpu
caffe.set_mode_cpu()
# choose which GPU you want to use
#caffe.set_device(0)
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('deploy.prototxt', 'ours_TRI_iter_36000.caffemodel', caffe.TEST)

#Visualization
def plot_single_scale(scale_lst, name_lst, size):
    pylab.rcParams['figure.figsize'] = size, size
    plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(3,size/2,i+1)
        s.set_xlabel(name_lst[i], fontsize=10)
        if name_lst[i][0:2] == 'So':
            plt.imshow(scale_lst[i])
        else:
            plt.imshow(scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()
    plt.show()
def plot_saveimg(id,out,src):
   # plt.figure()
   # plt.imshow(out)
   # plt.savefig('output_img/test2.jpg')
   # matplotlib.image.imsave('output_img/'+str(id)+'_.jpg', src, cmap = cm.Greys_r)
   # matplotlib.image.imsave('output_img/'+str(id)+'_.png', out, cmap = cm.Greys_r)
    new_im = Image.fromarray(out*255).convert('L')
    new_im.save('output_img/'+id[:-4]+'.png')
    print id

for idx in range(len(test_lst)):
	print test_lst[idx],test_lst2[idx],test_lst3[idx]
	# load image
	img = Image.open(test_lst[idx])
	img = img.resize((80,80),Image.BILINEAR)
	img = np.array(img, dtype=np.uint8)
	im = np.array(img, dtype=np.float32)
	if im.ndim!=3: continue
	im = im[:,:,::-1]
	im -= np.array((104.00698793,116.66876762,122.67891434))
	im = im.transpose((2,0,1))
	# load gt
	gt = Image.open(gt_lst[idx])#
	gt = gt.resize((80,80),Image.BILINEAR)
	gt = np.array(gt, dtype=np.uint8)

	img2 = Image.open(test_lst2[idx])
	img2 = img2.resize((80,80),Image.BILINEAR)
	img2 = np.array(img2, dtype=np.uint8)
	im2 = np.array(img2, dtype=np.float32)
	if im2.ndim!=3: continue
	im2 = im2[:,:,::-1]
	im2 -= np.array((104.00698793,116.66876762,122.67891434))
	im2 = im2.transpose((2,0,1))
	# load gt
	gt2 = Image.open(gt_lst2[idx])#
	gt2 = gt2.resize((80,80),Image.BILINEAR)
	gt2 = np.array(gt2, dtype=np.uint8)

	img3 = Image.open(test_lst3[idx])
	img3 = img3.resize((80,80),Image.BILINEAR)
	img3 = np.array(img3, dtype=np.uint8)
	im3 = np.array(img3, dtype=np.float32)
	if im3.ndim!=3: continue
	im3 = im3[:,:,::-1]
	im3 -= np.array((104.00698793,116.66876762,122.67891434))
	im3 = im3.transpose((2,0,1))
	# load gt
	gt3 = Image.open(gt_lst3[idx])#
	gt3 = gt3.resize((80,80),Image.BILINEAR)
	gt3 = np.array(gt3, dtype=np.uint8)

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data1'].reshape(1, *im.shape)
	net.blobs['data1'].data[...] = im
	net.blobs['data2'].reshape(1, *im2.shape)
	net.blobs['data2'].data[...] = im2
	net.blobs['data3'].reshape(1, *im3.shape)
	net.blobs['data3'].data[...] = im3
	#net.blobs['gt1'].reshape(1, *im_single.shape)
	#net.blobs['gt1'].data[...] = im_single
	# run net and take argmax for prediction
	net.forward()
	out1 = net.blobs['sigmoid-1'].data[0][0,:,:]
	out2 = net.blobs['sigmoid-2'].data[0][0,:,:]
	out3 = net.blobs['sigmoid-3'].data[0][0,:,:]

	out_lst = [out1, img, gt, out2, img2, gt2, out3, img3, gt3]
	name_lst = ['Result1', 'Source1', 'GT1','Result2', 'Source2', 'GT2','Result3', 'Source3', 'GT3']
	plot_single_scale(out_lst, name_lst, 6)

	print test_lst[idx].split('/')
	dirr = test_lst[idx].split('/')
#	plot_saveimg(test_lst[idx].split('/')[-1],out1,img)
#	plot_saveimg(test_lst2[idx].split('/')[-1],out2,img)
#	plot_saveimg(test_lst3[idx].split('/')[-1],out3,img)