import os

data_root = '/home/timplex/dataset/cosaliency/MSRC/images/'
print os.listdir(data_root)

fout1 = open('imgs_MSRC.lst','w')
fout3 = open('mask_MSRC.lst','w')

for fs in os.listdir(data_root):
	path = data_root + fs + '/'
	print path
	for fname in os.listdir(path):
	#	if (fname[-1]=='p'): continue
		fout1.write(path+fname+' 0\n')
		fout3.write('/home/timplex/dataset/cosaliency/MSRC/groundtruth/' + str(fs) + '/'+fname[:-4]+'.bmp 0\n')
		print fname[:-4]
fout1.close()
fout3.close()