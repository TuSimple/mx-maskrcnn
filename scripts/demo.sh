export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=${PYTHONPATH}:incubator-mxnet/python/

MODEL_PATH=model/
RESULT_PATH=data/cityscape/results/

PREFIX=${MODEL_PATH}final
DATASET=Cityscape
SET=train
TEST_SET=val

mkdir -p ${RESULT_PATH}

python demo_mask.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${PREFIX} \
    --result_path ${RESULT_PATH} \
    --has_rpn \
    --epoch 0 \
    --gpu 0
