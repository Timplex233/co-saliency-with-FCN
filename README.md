Our caffemodel file can be downloaded [here](敌法敌法链接)

Our Environment: Ubuntu 18.04, CUDA9.0

We use Image Label Data Layer from HED Caffe, so  if you are using official Caffe please add the layer manually and re-compile the Caffe. Or you can directly install [this](https://github.com/Andrew-Qibin/caffe_dss) version.

The usage of some files are here below

- run_saliency.py: Train the model, please alter the location of base model before running
- saliency_test.py: Test the model. You can either print the images on the screen or directly save the output images. 
- mul.py & CRF.py: Further processing the output images from FCN (saliency_test) with method A and B in the paper respectively. Notice: mul.py requires user generate the single-image saliency map before (we use method [DSS](https://github.com/Andrew-Qibin/DSS]) ). CRF.py we use [pyDense](https://github.com/lucasb-eyer/pydensecrf), please install it first.
- comp_.py: Compare results with different further improvement like CRF, mul
- some other python files are used to generate image lists, you can read them if you need them.

## Notice, please check the data path in the code before using.
