"""
Make rois for mask branch.
"""

import mxnet as mx
from rcnn.processing.bbox_transform import *

bbox_pred = nonlinear_pred

class MaskROIOperator(mx.operator.CustomOp):
    def __init__(self, num_classes):
        super(MaskROIOperator, self).__init__()
        self._num_classes = int(num_classes)


    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0]
        bbox_deltas = in_data[1]
        data = in_data[2]
        if is_train:
            label = in_data[3].asnumpy().astype('Int32')[0]
            pred_boxes = bbox_pred(rois.asnumpy()[:, 1:], bbox_deltas.asnumpy())
        else:
            label = np.argmax(in_data[3].asnumpy()[0], axis=1)
            pred_boxes = bbox_pred(rois.asnumpy()[:, 1:], bbox_deltas.asnumpy()[0])
        output_box = np.zeros([rois.shape[0], 4])
        for i in range(label.shape[0]):
            cls = int(label[i])
            output_box[i, :] = pred_boxes[i, 4 * cls : 4 * (cls + 1)]
        output_box = clip_boxes(output_box, data.shape[-2:])
        batch_inds = np.zeros((output_box.shape[0], 1), dtype=np.float32)
        output_box = np.hstack((batch_inds, output_box.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], output_box)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)


@mx.operator.register("mask_roi")
class MaskROIProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes):
        super(MaskROIProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)

    def list_arguments(self):
        return ['rois', 'bbox_deltas', 'data', 'label']

    def list_outputs(self):
        return ['mask_rois']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        bbox_deltas_shape = in_shape[1]
        data_shape = in_shape[2]
        label_shape = in_shape[3]

        return [rois_shape, bbox_deltas_shape, data_shape, label_shape], [rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MaskROIOperator(num_classes=self._num_classes)