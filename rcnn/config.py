import numpy as np
from easydict import EasyDict as edict

config = edict()

# network related params
config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.ROIALIGN = True

config.RPN_FEAT_STRIDE = [64, 32, 16, 8, 4]
config.RCNN_FEAT_STRIDE = [32, 16, 8, 4]

config.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
config.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'stage4',
                              'P5', 'P4', 'P3', 'P2',
                              'gamma', 'beta']

# dataset related params
config.NUM_CLASSES = 9
config.SCALES = [(1024, 2048)]  # first is scale (the shorter side); second is max size
config.ANCHOR_SCALES = (8,)
config.ANCHOR_RATIOS = (0.5, 1, 2)
config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)
config.CLASS_ID = [0, 24, 25, 26, 27, 28, 31, 32, 33]

config.TRAIN = edict()

# R-CNN and RPN
config.TRAIN.BATCH_IMAGES = 1
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# scale
config.TRAIN.SCALE = True
config.TRAIN.SCALE_RANGE = (0.8, 1)

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 256

# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# RPN proposal
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000

config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = True
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.RPN_NMS_THRESH = 0.7

config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 1000

config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000

config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3


# default settings
default = edict()

# default network
default.network = 'resnet_fpn'
default.pretrained = 'model/resnet-50'
default.pretrained_epoch = 0
default.base_lr = 0.004
# default dataset
default.dataset = 'Cityscape'
default.image_set = 'train'
default.test_image_set = 'val'
default.root_path = 'data'
default.dataset_path = 'data/cityscape'
# default training
default.frequent = 20
default.kvstore = 'device'
# default rpn
default.rpn_prefix = 'model/rpn'
default.rpn_epoch = 8
default.rpn_lr = default.base_lr
default.rpn_lr_step = '6'
# default rcnn
default.rcnn_prefix = 'model/rcnn'
default.rcnn_epoch = 24
default.rcnn_lr = default.base_lr
default.rcnn_lr_step = '20'
# default alternate
default.alternate_prefix = 'model/alternate'

# network settings
network = edict()

network.vgg = edict()

network.resnet_fpn = edict()
network.resnet_fpn.pretrained = 'model/resnet-50'
network.resnet_fpn.pretrained_epoch = 0
network.resnet_fpn.PIXEL_MEANS = np.array([0, 0, 0])
network.resnet_fpn.RPN_FEAT_STRIDE = [64, 32, 16, 8, 4]
network.resnet_fpn.RCNN_FEAT_STRIDE = [32, 16, 8, 4]
network.resnet_fpn.RPN_MIN_SIZE = network.resnet_fpn.RPN_FEAT_STRIDE
network.resnet_fpn.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
network.resnet_fpn.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'stage4',
                                          'P5', 'P4', 'P3', 'P2',
                                          'gamma', 'beta']

# dataset settings
dataset = edict()

dataset.Cityscape = edict()
dataset.Cityscape.image_set = 'train'
dataset.Cityscape.test_image_set = 'val'
dataset.Cityscape.root_path = 'data'
dataset.Cityscape.dataset_path = 'data/cityscape'
dataset.Cityscape.NUM_CLASSES = 9
dataset.Cityscape.SCALES = [(1024, 2048)]
dataset.Cityscape.ANCHOR_SCALES = (8,)
dataset.Cityscape.ANCHOR_RATIOS = (0.5, 1, 2)
dataset.Cityscape.NUM_ANCHORS = len(dataset.Cityscape.ANCHOR_SCALES) * len(dataset.Cityscape.ANCHOR_RATIOS)
dataset.Cityscape.CLASS_ID = [0, 24, 25, 26, 27, 28, 31, 32, 33]

def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v

