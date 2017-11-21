/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align-inl.h
 * \brief roi align operator and symbol
 * \author Yuchen Guo, Zehao Shi
*/
#ifndef MXNET_OPERATOR_ROI_ALIGN_INL_H_
#define MXNET_OPERATOR_ROI_ALIGN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace roialign {
enum ROIAlignOpInputs {kData, kBox};
enum ROIAlignOpOutputs {kOut, kMaxIdx_x, kMaxIdx_y};
}  // roialign

struct ROIAlignParam : public dmlc::Parameter<ROIAlignParam> {
  TShape pooled_size;
  float spatial_scale;
  DMLC_DECLARE_PARAMETER(ROIAlignParam) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("fix pooled size: (h, w)");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
  }
};

template<typename xpu, typename DType>
class ROIAlignOp : public Operator {
 public:
  explicit ROIAlignOp(ROIAlignParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected_in = 2;
    size_t expected_out = 3;
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);
    CHECK_EQ(out_data[roialign::kOut].shape_[0], in_data[roialign::kBox].shape_[0]);
    CHECK_EQ(out_data[roialign::kMaxIdx_x].shape_[0], in_data[roialign::kBox].shape_[0]);
    CHECK_EQ(out_data[roialign::kMaxIdx_y].shape_[0], in_data[roialign::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[roialign::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[roialign::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[roialign::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> max_idx_x = out_data[roialign::kMaxIdx_x].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> max_idx_y = out_data[roialign::kMaxIdx_y].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(max_idx_x.CheckContiguous(), true);
    CHECK_EQ(max_idx_y.CheckContiguous(), true);
    out = -FLT_MAX;
    max_idx_x = -1.0f;
    max_idx_y = -1.0f;
    ROIAlignForward(out, data, bbox, max_idx_x, max_idx_y, param_.spatial_scale);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected_in = 2;
    size_t expected_out = 3;
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);
    CHECK_EQ(out_grad[roialign::kOut].shape_[0], in_data[roialign::kBox].shape_[0]);
    CHECK_EQ(out_data[roialign::kMaxIdx_x].shape_[0], in_data[roialign::kBox].shape_[0]);
    CHECK_EQ(out_data[roialign::kMaxIdx_y].shape_[0], in_data[roialign::kBox].shape_[0]);
    CHECK_NE(req[roialign::kData], kWriteInplace) <<
      "ROIAlign: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[roialign::kBox], kWriteInplace) <<
      "ROIAlign: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[roialign::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[roialign::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> max_idx_x = out_data[roialign::kMaxIdx_x].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> max_idx_y = out_data[roialign::kMaxIdx_y].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[roialign::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grad_roi = in_grad[roialign::kBox].get<xpu, 2, DType>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(max_idx_x.CheckContiguous(), true);
    CHECK_EQ(max_idx_y.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    if (kAddTo == req[roialign::kData] || kWriteTo == req[roialign::kData]) {
      if (kWriteTo == req[roialign::kData]) {
        grad_in = 0.0f;
      }
      ROIAlignBackwardAcc(grad_in, grad_out, bbox, max_idx_x, max_idx_y, param_.spatial_scale);
    }
    if (kWriteTo == req[roialign::kBox]) {
      grad_roi = 0.0f;
    }
  }

 private:
  ROIAlignParam param_;
};  // class ROIAlignOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ROIAlignParam param, int dtype);

#if DMLC_USE_CXX11
class ROIAlignProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "rois"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "maxidx_x", "maxidx_y"};
  }

  int NumOutputs() const override {
    return 3;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(roialign::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(roialign::kBox);
    CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
    CHECK_EQ(bshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";

    // out: [num_rois, c, pooled_h, pooled_w]
    // max_idx_x: [num_rois, c, pooled_h, pooled_w]
    // max_idx_y: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    ROIAlignProp* roi_align_sym = new ROIAlignProp();
    roi_align_sym->param_ = this->param_;
    return roi_align_sym;
  }

  std::string TypeString() const override {
    return "ROIAlign";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[roialign::kOut], in_data[roialign::kBox], out_data[roialign::kMaxIdx_x], out_data[roialign::kMaxIdx_y]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ROIAlignParam param_;
};  // class ROIAlignProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ROI_Align_INL_H_
