import os

f1 = open('imgs_MSRC.lst','r')
fout = open('train_MSRC.lst','w')
f2 = open('mask_MSRC.lst','r')
#data_root = '/home/timplex/dataset/cosaliency/iCoseg/images/'
while 1:
	l1 = f1.readline()
	l2 = f2.readline()
	if not l1: break
	fout.write(l1.split(' ')[0]+' '+l2.split(' ')[0]+'\n')
f1.close()
f2.close()
fout.close()