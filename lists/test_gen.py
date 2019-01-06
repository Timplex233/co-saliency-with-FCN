import os
import random

fout1 = open('test1_1.lst','w')
fout2 = open('test1_2.lst','w')
fout3 = open('test1_3.lst','w')
#f2 = open('mask1.lst','r')
#data_root = '/home/timplex/dataset/cosaliency/iCoseg/images/'
'''with open('train1.lst') as f1:
    lst = f1.readlines()
with open('train_cosal.lst') as f2:
    lst += f2.readlines()'''
with open('train_MSRC.lst') as f2:
    lst = f2.readlines()
lst = [x.strip() for x in lst]
idx = [x.split(' ')[0].split('/')[-2] for x in lst]
#print idx
i=0
out=[]
while i+2<len(lst):
	print i
	if (idx[i]==idx[i+1]) and (idx[i]==idx[i+2]):
		out.append([lst[i],lst[i+1],lst[i+2]])
		i+=2
	i+=1
print out
random.shuffle(out)
for x in out:
	fout1.write(x[0]+'\n')
	fout2.write(x[1]+'\n')
	fout3.write(x[2]+'\n')
'''while 1:
	l1 = f1.readline()
	l2 = f2.readline()
	if not l1: break
	fout.write(l2.split(' ')[0]+' '+l2.split(' ')[0]+'\n')'''
#f1.close()
fout1.close()
fout2.close()
fout3.close()