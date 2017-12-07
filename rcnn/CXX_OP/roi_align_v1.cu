/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align.cu
 * \brief roi align operator
 * \author Yuchen Guo, Zehao Shi
*/
#include "./roi_align_v1-inl.h"


namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_ROIAlign_v1)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignForward_v1<gpu>);

NNVM_REGISTER_OP(_backward_ROIAlign_v1)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignBackward_v1<gpu>);

}  // namespace op
}  // namespace mxnet
