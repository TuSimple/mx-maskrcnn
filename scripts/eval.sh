export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=${PYTHONPATH}:incubator-mxnet/python/

TRAIN_DIR=model/res50-fpn/cityscape/alternate/
DATASET=Cityscape
SET=train
TEST_SET=val

# Test
python eval_maskrcnn.py \
    --network resnet_fpn \
    --has_rpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${TRAIN_DIR}final \
    --result_path data/cityscape/results/pred/ \
    --epoch 0 \
    --gpu 0

python data/cityscape/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py
