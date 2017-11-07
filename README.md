# MX Mask R-CNN
An MXNet implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870).

This repository is based largely on the mx-rcnn implementation of Faster RCNN available [here](https://github.com/precedenceguo/mx-rcnn).


<div align="center">
<img src="https://github.com/TuSimple/mx-maskrcnn/blob/master/figures/maskrcnn_result.png"><br><br>
</div>


## Main Results


### Cityscapes 

| Method |Training data| Test data| Average | person | rider | car | truck | bus  | train| motorcycle| bicycle|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ours| fine-only |test|26.9|33.0|25.7|47.7|21.6|27.4|23.0|19.9|16.9|
| Reference[5]| fine-only |test|26.2|30.5|23.8|46.9|22.8|32.2|18.6|19.1|16.0|
| Ours | fine-only |val|31.3|32.6|26.6|49.5|26.5|45.4|32.1|17.6|20.4|
| Reference[5]| fine-only |val|31.5| -| -| -| -| -| -| -| -| -| -|

- Backbone: Resnet-50-FPN

### COCO
Coming soon, please stay tuned.

## Requirement

We tested our code on:

Ubuntu 16.04, Python 2.7 with

numpy(1.12.1), cv2(2.4.9), PIL(4.3), matplotlib(2.1.0), cython(0.26.1), easydict

## Preparation for Training

1. Download Cityscapes data (gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip). Extract them into 'data/cityscape/'.
 The folder structure would then look as shown below:

```
data/cityscape/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
├── gtFine/
│   ├── train/
│   ├── val/
│   └── test/
└── imglists/
    ├── train.lst
    ├── val.lst
    └── test.lst
```


2. Download Resnet-50 pretrained model.
```
bash scripts/download_res50.sh

```

3. Build MXNet with ROIAlign operator.

```
cp rcnn/CXX_OP/* incubator-mxnet/src/operator/
```

To build MXNet from source, please refer to the [tutorial](https://mxnet.incubator.apache.org/get_started/build_from_source.html).

4. Build related cython code.

```
make
```

5. Kick off training

```
bash scripts/train_alternate.sh
```

## Preparation for Evaluation
1. Prepare Cityscapes evaluation scripts.

```
bash scripts/download_cityscapescripts.sh
```
2. Eval
```
bash scripts/eval.sh
```

## Demo
1. Download model, available at [Dropbox](https://www.dropbox.com/s/zidcbbt7apwg3z6/final-0000.params?dl=0)/[BaiduYun](https://pan.baidu.com/s/1o8n4VMU), and place it in the model folder. 
2. Make sure that you have the cityscapes data in 'data/cityscapes' folder.
```
bash scripts/demo.sh
```

## Test single image
1. Download model, available at [Dropbox](https://www.dropbox.com/s/zidcbbt7apwg3z6/final-0000.params?dl=0)/[BaiduYun](https://pan.baidu.com/s/1o8n4VMU), and place it in the model folder. 
2. Follow `Preparation for Training` (step1-step4)
3. run `bash scripts/demo_single_image.sh`, you can change the image path in script demo_single_image.sh.

## FAQ
Q: It says **`AttributeError: 'module' object has no attribute 'ROIAlign'`**.

A: This is because either
 - you forget to copy the operators to your MXNet folder
 - or you forget to re-compile MXNet and re-install MXNet python interface
 - or you install the wrong MXNet
 
     Please print `mxnet.__path__` to make sure you use correct MXNet
     
Q: I encounter **`incubator-mxnet/mshadow/mshadow/././././cuda/tensor_gpu-inl.cuh:110: Check failed: err == cudaSuccess (7 vs. 0) Name: MapPlanKernel ErrStr:too many resources requested for launch`** at the begining.

A: Please try adding `MSHADOW_CFLAGS += -DMSHADOW_OLD_CUDA=1` in `mxnet/mshadow/make/mshadow.mk` and re-compile MXNet.

## References
1. Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
3. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.
4. Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. "Feature Pyramid Networks for Object Detection." In Computer Vision and Pattern Recognition, IEEE Conference on, 2017.
5. Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. "Mask R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2017.
4. Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the ACM International Conference on Multimedia, 2014.
5. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, IEEE Conference on, 2009.
6. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". In Computer Vision and Pattern Recognition, IEEE Conference on, 2016.
7. Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. "The Cityscapes Dataset for Semantic Urban Scene Understanding." In Computer Vision and Pattern Recognition, IEEE Conference on, 2016.
