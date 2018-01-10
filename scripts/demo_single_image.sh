export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=${PYTHONPATH}:incubator-mxnet/python/

MODEL_PATH=model/
PREFIX=${MODEL_PATH}final
DATASET=Cityscape

python2 -m rcnn.tools.demo_single_image \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --prefix ${PREFIX} \
    --epoch 0 \
    --gpu 0 \
    --image_name figures/test.jpg \
    --thresh 0.3 \
    --vis


