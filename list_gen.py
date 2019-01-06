import os

data_root = '/home/timplex/caffe_dss/examples/COSS/output_img/'
print os.listdir('output_img')

img_lst = os.listdir('output_img')

with open('lists/mask_MSRC.lst') as f1:
    lst = f1.readlines()
lst = [x.split(' ')[0] for x in lst]
fout1 = open('test_lst.txt','w')

for x in lst:
	nm = x.split('/')[-1][:-4]+'.png'
	if nm in img_lst:
		fout1.write(x + ' ' + data_root+nm+'\n')
fout1.close()