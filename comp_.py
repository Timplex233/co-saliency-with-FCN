import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import matplotlib
import scipy.misc
from PIL import Image
import scipy.io
import os


EPSILON = .1

with open('lists/imgs_cosal.lst') as f:
    test_lst = f.readlines()
test_lst = [x.strip().split(' ')[0] for x in test_lst]

def plot_single_scale(scale_lst, name_lst, size):
    pylab.rcParams['figure.figsize'] = size, 2
    plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(1,size/2,i+1)
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
    plt.subplots_adjust(wspace =0, hspace =0)
    plt.show()
def plot_saveimg(id,out):
   # plt.figure()
   # plt.imshow(out)
   # plt.savefig('output_img/test2.jpg')
   # matplotlib.image.imsave('output_img/'+str(id)+'_.jpg', src, cmap = cm.Greys_r)
   # matplotlib.image.imsave('output_img/'+str(id)+'_.png', out, cmap = cm.Greys_r)
    new_im = Image.fromarray(out).convert('L')
    new_im.save('output_111/'+id+'.png')
    #print id

for idx in range(len(test_lst)):
#	print test_lst[idx],test_lst2[idx],test_lst3[idx]
	# load image
	iname = test_lst[idx].split('/')[-1][:-4]
	print idx,iname

	if not os.path.exists('output_img/' + iname + '.png'):continue
	img1 = Image.open('output_img/' + iname + '.png')
	img1 = img1.resize((80,80),Image.BILINEAR)
	img1 = np.array(img1, dtype=np.uint8)

	if not os.path.exists('output_mul/' + iname + '.png'):continue
	img2 = Image.open('output_mul/' + iname + '.png')
	img2 = img2.resize((80,80),Image.BILINEAR)
	img2 = np.array(img2, dtype=np.uint8)

	if not os.path.exists('output_CRF/' + iname + '.png'):continue
	img3 = Image.open('output_CRF/' + iname + '.png')
	img3 = img3.resize((80,80),Image.BILINEAR)
	img3 = np.array(img3, dtype=np.uint8)

	if not os.path.exists('output_f/' + iname + '.png'):continue
	img4 = Image.open('output_f/' + iname + '.png')
	img4 = img4.resize((80,80),Image.BILINEAR)
	img4 = np.array(img4, dtype=np.uint8)


	#print img1
	#print img2

	res = (np.array(img1, dtype=np.float32)/255.+EPSILON)*np.sqrt(np.array(img2, dtype=np.float32)/255.+EPSILON)
	#print res,np.min(res),np.max(res)
	res = np.array((res-np.min(res))/(np.max(res)-np.min(res))*255., dtype=np.uint8)

	# load gt
	gt = Image.open(test_lst[idx].replace('images','groundtruth')[:-4]+'.png')#
	gt = gt.resize((80,80),Image.BILINEAR)
	gt = np.array(gt, dtype=np.uint8)
	# load So
	so = Image.open(test_lst[idx])#
	so = so.resize((80,80),Image.BILINEAR)
	so = np.array(so, dtype=np.uint8)

	out_lst = [so, img1, img2, img3, img4, gt]
	name_lst = ['Source', 'Origin', 'F A', 'F B', 'F C', 'GT']
	plot_single_scale(out_lst, name_lst, 12)

#	print test_lst[idx].split('/')
#	dirr = test_lst[idx].split('/')
#	plot_saveimg(iname,res)
#	plot_saveimg(test_lst2[idx].split('/')[-1],out2,img)
#	plot_saveimg(test_lst3[idx].split('/')[-1],out3,img)