from __future__ import division
import numpy as np
import sys
caffe_root = '/home/timplex/caffe_dss/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')
solver.restore('snapshot/ours_TRI_iter_12000.solverstate')
solver.step(12000)
