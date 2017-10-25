"""
Generate proposals for feature pyramid networks.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper

DEBUG = False

class ProposalFPNOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride_fpn, scales, ratios, output_score,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size_fpn):
        super(ProposalFPNOperator, self).__init__()
        self._feat_stride_fpn = np.fromstring(feat_stride_fpn[1:-1], dtype=int, sep=',')
        self.fpn_keys = []
        fpn_stride = []

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)
            fpn_stride.append(int(s))

        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_stride, scales=self._scales, ratios=self._ratios)))
        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        self._output_score = output_score
        self._rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self._rpn_post_nms_top_n = rpn_post_nms_top_n
        self._threshold = threshold
        self._rpn_min_size_fpn = dict(zip(self.fpn_keys, np.fromstring(rpn_min_size_fpn[1:-1], dtype=int, sep=',')))
        self._bbox_pred = nonlinear_pred

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride_fpn)
            print 'anchors: {}'.format(self._anchors_fpn)
            print 'num_anchors: {}'.format(self._num_anchors)

    def forward(self, is_train, req, in_data, out_data, aux):
        nms = gpu_nms_wrapper(self._threshold, in_data[0][0].context.device_id)

        cls_prob_dict = dict(zip(self.fpn_keys, in_data[0:len(self.fpn_keys)]))
        bbox_pred_dict = dict(zip(self.fpn_keys, in_data[len(self.fpn_keys):2*len(self.fpn_keys)]))
        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        pre_nms_topN = self._rpn_pre_nms_top_n
        post_nms_topN = self._rpn_post_nms_top_n
        min_size_dict = self._rpn_min_size_fpn

        proposals_list = []
        scores_list = []
        for s in self._feat_stride_fpn:
            stride = int(s)
            scores = cls_prob_dict['stride%s'%s].asnumpy()[:, self._num_anchors['stride%s'%s]:, :, :]
            bbox_deltas = bbox_pred_dict['stride%s'%s].asnumpy()
            im_info = in_data[-1].asnumpy()[0, :]

            if DEBUG:
                print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                print 'scale: {}'.format(im_info[2])

            height, width = int(im_info[0] / stride), int(im_info[1] / stride)

            A = self._num_anchors['stride%s'%s]
            K = height * width

            anchors = anchors_plane(height, width, stride, self._anchors_fpn['stride%s'%s].astype(np.float32))
            anchors = anchors.reshape((K * A, 4))

            bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

            scores = self._clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            proposals = self._bbox_pred(anchors, bbox_deltas)

            proposals = clip_boxes(proposals, im_info[:2])

            keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
            proposals = proposals[keep, :]
            scores = scores[keep]
            proposals_list.append(proposals)
            scores_list.append(scores)

        proposals = np.vstack(proposals_list)
        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        det = np.hstack((proposals, scores)).astype(np.float32)

        if np.shape(det)[0] == 0:
            print "Something wrong with the input image(resolution is too low?), generate fake proposals for it."
            proposals = np.array([[1.0, 1.0, 2.0, 2.0]]*post_nms_topN, dtype=np.float32)
            scores = np.array([[0.9]]*post_nms_topN, dtype=np.float32)
            det = np.array([[1.0, 1.0, 2.0, 2.0, 0.9]]*post_nms_topN, dtype=np.float32)

        keep = nms(det)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]

        if len(keep) < post_nms_topN:
            pad = npr.choice(keep, size=post_nms_topN - len(keep))
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]
        scores = scores[keep]

        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], blob)

        if self._output_score:
            self.assign(out_data[1], req[1], scores.astype(np.float32, copy=False))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # forward only currently
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor


@mx.operator.register("proposal_fpn")
class ProposalFPNProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='(64,32,16,8,4)', scales='(8)', ratios='(0.5, 1, 2)', output_score='False',
                 rpn_pre_nms_top_n='6000', rpn_post_nms_top_n='300', threshold='0.3', rpn_min_size='(64,32,16,8,4)'):
        super(ProposalFPNProp, self).__init__(need_top_grad=False)
        self._feat_stride_fpn = feat_stride
        self._scales = scales
        self._ratios = ratios
        self._output_score = strtobool(output_score)
        self._rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        self._threshold = float(threshold)
        self._rpn_min_size_fpn = rpn_min_size

    def list_arguments(self):
        args_list = []
        for s in np.fromstring(self._feat_stride_fpn[1:-1], dtype=int, sep=','):
            args_list.append('cls_prob_stride%s'%s)
        for s in np.fromstring(self._feat_stride_fpn[1:-1], dtype=int, sep=','):
            args_list.append('bbox_pred_stride%s' % s)
        args_list.append('im_info')

        return args_list

    def list_outputs(self):
        if self._output_score:
            return ['output', 'score']
        else:
            return ['output']

    def infer_shape(self, in_shape):
        output_shape = (self._rpn_post_nms_top_n, 5)
        score_shape = (self._rpn_post_nms_top_n, 1)

        if self._output_score:
            return in_shape, [output_shape, score_shape]
        else:
            return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalFPNOperator(self._feat_stride_fpn, self._scales, self._ratios, self._output_score,
                                self._rpn_pre_nms_top_n, self._rpn_post_nms_top_n, self._threshold, self._rpn_min_size_fpn)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
