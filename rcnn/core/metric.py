import mxnet as mx
import numpy as np
from rcnn.config import config

def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names():
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    return pred, label


def get_rcnn_fpn_names():
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = []
    for s in config.RCNN_FEAT_STRIDE:
        label.append('rcnn_label_stride%s' % s)
        label.append('rcnn_bbox_target_stride%s' % s)
        label.append('rcnn_bbox_weight_stride%s' % s)
    return pred, label

def get_maskrcnn_fpn_name():
    rcnn_pred, rcnn_label = get_rcnn_fpn_names()
    pred = rcnn_pred + ['mask_prob']
    label = []
    for s in config.RCNN_FEAT_STRIDE:
        label.append('mask_target_stride%s' % s)
        label.append('mask_weight_stride%s' % s)
    label = rcnn_label + label
    return pred, label

class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = []
        for s in config.RCNN_FEAT_STRIDE:
            label.append(labels[self.label.index('rcnn_label_stride%s' % s)].asnumpy().reshape((-1,)).astype('Int32'))
        label = np.concatenate(label, axis=0)

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')

        index = np.where(label!=-1)
        label = label[index]
        pred_label = pred_label[index]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class MaskLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskLogLossMetric, self).__init__('MaskLogLoss')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        # reshape and concat
        label = []
        mask_target = []
        mask_weight = []
        mask_prob = preds[self.pred.index('mask_prob')].asnumpy()  # (n_rois, c, h, w)
        for s in config.RCNN_FEAT_STRIDE:
            label.append(labels[self.label.index('rcnn_label_stride%s' % s)].asnumpy().reshape((-1,)).astype('Int32'))
            mask_target.append(labels[self.label.index('mask_target_stride%s' % s)].asnumpy().reshape((-1, config.NUM_CLASSES, 28,28)))
            mask_weight.append(labels[self.label.index('mask_weight_stride%s' % s)].asnumpy().reshape((-1, config.NUM_CLASSES, 1,1)))
        label = np.concatenate(label, axis=0)
        mask_target = np.concatenate(mask_target, axis=0)
        mask_weight = np.concatenate(mask_weight, axis=0)

        real_inds   = np.where(label != -1)[0]
        n_rois      = real_inds.shape[0]
        mask_prob   = mask_prob[real_inds, label[real_inds]]
        mask_target = mask_target[real_inds, label[real_inds]]
        mask_weight = mask_weight[real_inds, label[real_inds]]
        l = mask_weight*mask_target * np.log(mask_prob + 1e-14) + mask_weight * (1 - mask_target) * np.log(1 - mask_prob + 1e-14)
        self.sum_metric += -np.sum(l)
        self.num_inst += mask_prob.shape[-1] * mask_prob.shape[-2] * n_rois

class MaskAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskAccMetric, self).__init__('MaskACC')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        # reshape and concat
        label = []
        mask_target = []
        mask_weight = []
        mask_prob = preds[self.pred.index('mask_prob')].asnumpy()  # (n_rois, c, h, w)
        for s in config.RCNN_FEAT_STRIDE:
            label.append(labels[self.label.index('rcnn_label_stride%s' % s)].asnumpy().reshape((-1,)).astype('Int32'))
            mask_target.append(labels[self.label.index('mask_target_stride%s' % s)].asnumpy().reshape((-1, config.NUM_CLASSES, 28,28)))
            mask_weight.append(labels[self.label.index('mask_weight_stride%s' % s)].asnumpy().reshape((-1, config.NUM_CLASSES, 1,1)))
        label = np.concatenate(label, axis=0)
        mask_target = np.concatenate(mask_target, axis=0)
        mask_weight = np.concatenate(mask_weight, axis=0)

        real_inds = np.where(label != -1)[0]
        n_rois = real_inds.shape[0]
        mask_prob   = mask_prob[real_inds, label[real_inds]]
        mask_target = mask_target[real_inds, label[real_inds]]
        mask_weight = mask_weight[real_inds, label[real_inds]]
        idx = np.where(np.logical_and(mask_prob > 0.5, mask_weight == 1))
        mask_pred = np.zeros_like(mask_prob)
        mask_pred[idx] = 1
        self.sum_metric += np.sum(mask_target == mask_pred)
        self.num_inst += mask_prob.shape[-1] * mask_prob.shape[-2] * n_rois


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = []
        for s in config.RCNN_FEAT_STRIDE:
            label.append(labels[self.label.index('rcnn_label_stride%s' % s)].asnumpy().reshape((-1,)).astype('Int32'))
        label = np.concatenate(label, axis=0)

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)

        cls = pred[np.arange(label.shape[0]), label]

        index = np.where(label!=-1)
        cls = cls[index]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += index[0].shape[0]


class RPNRegLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        name = 'RPNL1Loss'
        super(RPNRegLossMetric, self).__init__(name)
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNRegLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        name = 'RCNNL1Loss'
        super(RCNNRegLossMetric, self).__init__(name)
        self.pred, self.label = get_maskrcnn_fpn_name()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        label = []
        for s in config.RCNN_FEAT_STRIDE:
            label.append(labels[self.label.index('rcnn_label_stride%s' % s)].asnumpy().reshape((-1,)).astype('Int32'))
        label = np.concatenate(label, axis=0)

        last_dim = bbox_loss.shape[-1]
        bbox_loss = bbox_loss.reshape((-1, last_dim))
        label = label.reshape(-1)

        # calculate num_inst
        keep_inds = np.where((label != 0) & (label != -1))[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

