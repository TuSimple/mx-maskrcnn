"""
Compute the instance segmentation output using the class-specific masks.
"""

import mxnet as mx
import numpy as np

DEBUG = False

class MaskOutputOperator(mx.operator.CustomOp):
    def __init__(self):
        super(MaskOutputOperator, self).__init__()
        self.factor = 0.03125

    def forward(self, is_train, req, in_data, out_data, aux):
        if DEBUG:
            print is_train
            print len(in_data)

        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert len(in_data) == 4
        assert len(out_data) == 1
        mask_prob, mask_target, mask_weight, label = in_data
        n_rois_fg = np.where(label.asnumpy()>0)[0].shape[0]
        grad = self.factor*mask_weight*(mask_prob - mask_target)/float(n_rois_fg) # only fg rois contribute to grad
        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("MaskOutput")
class MaskOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MaskOutputProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['mask_prob', 'mask_target', 'mask_weight', 'label']

    def list_outputs(self):
        return ['output_prob']

    def infer_shape(self, in_shape):
        mask_prob_shape = in_shape[0]
        mask_target_shape = in_shape[1]
        mask_weight_shape = in_shape[2]
        label_shape = in_shape[3]

        assert mask_prob_shape[0] == mask_target_shape[0], \
            'mask_prob_shape[0] != mask_target_shape[0], {} vs {}'.format(mask_prob_shape[0], mask_target_shape[0])
        assert mask_prob_shape[0] == mask_weight_shape[0]
        assert mask_prob_shape[0] == label_shape[0]
        assert mask_target_shape[2] == mask_target_shape[3]

        output_mask_shape = mask_prob_shape
        return in_shape, [output_mask_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MaskOutputOperator()

