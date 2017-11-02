/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align.cu
 * \brief roi align operator
 * \author Yuchen Guo, Zehao Shi
*/
#include "./roi_align-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void ROIAlignForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale,
                                     const int channels, const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois, Dtype* top_data,
                                     Dtype* argmax_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_data[index] = 0;
      continue;
    }

    Dtype roi_start_w = (bottom_rois[1]) * spatial_scale;
    Dtype roi_start_h = (bottom_rois[2]) * spatial_scale;
    Dtype roi_end_w = (bottom_rois[3]) * spatial_scale;
    Dtype roi_end_h = (bottom_rois[4]) * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(1));
    Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(1));
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>((ph) * bin_size_h);
    Dtype wstart = static_cast<Dtype>((pw) * bin_size_w);
    Dtype hend = static_cast<Dtype>((ph + 1) * bin_size_h);
    Dtype wend = static_cast<Dtype>((pw + 1) * bin_size_w);

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height));
    hend = min(max(hend + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height));
    wstart = min(max(wstart + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    wend = min(max(wend + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    int bottom_index = 0;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    Dtype h_stride = (hend - hstart)/3.0;
    Dtype w_stride = (wend - wstart)/3.0;
    for (Dtype h = hstart+h_stride; h <= hend-h_stride+0.01; h += max(h_stride, 0.01)) {
      for (Dtype w = wstart+w_stride; w <= wend-w_stride+0.01; w += max(w_stride, 0.01)) {
        bottom_index ++;
        int hlow = min(max(static_cast<int>(floor(h)), 0), height-1);
        int hhigh = hlow + 1;
        int wleft = min(max(static_cast<int>(floor(w)), 0), width-1);
        int wright = wleft + 1;
        int topleft = hlow * width + wleft;
        int topright = hlow * width + wright;
        int bottomleft = hhigh * width + wleft;
        int bottomright = hhigh * width + wright;
        
        Dtype alpha = (hlow == hhigh) ? static_cast<Dtype>(0.5) : (h - hlow) / (hhigh - hlow);
        Dtype beta = (wleft == wright) ? static_cast<Dtype>(0.5) : (w - wleft) / (wright - wleft);
        Dtype value = (1 - alpha) * (1 - beta) * bottom_data[topleft] + alpha * (1 - beta) * bottom_data[bottomleft]
                            + (1 - alpha) * beta * bottom_data[topright] + alpha * beta * bottom_data[bottomright];
        
        if (value > maxval) {
          maxval = value;
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = (Dtype)maxidx;
  }
}

template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIAlignForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data, argmax_data);
}

template<typename Dtype>
__global__ void ROIAlignBackwardAccKernel(const int count, const Dtype* top_diff,
                                         const Dtype* argmax_data, const int num_rois,
                                         const float spatial_scale,
                                         const int channels, const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         Dtype* bottom_diff, const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      Dtype roi_start_w = (offset_bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h = (offset_bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w = (offset_bottom_rois[3]) * spatial_scale;
      Dtype roi_end_h = (offset_bottom_rois[4]) * spatial_scale;

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w > roi_start_w - 1.0 && w < roi_end_w + 1.0 &&
                           h > roi_start_h - 1.0 && h < roi_end_h + 1.0);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(1));
      Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(1));

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          Dtype hstart = static_cast<Dtype>((ph) * bin_size_h);
          Dtype wstart = static_cast<Dtype>((pw) * bin_size_w);
          Dtype hend = static_cast<Dtype>((ph + 1) * bin_size_h);
          Dtype wend = static_cast<Dtype>((pw + 1) * bin_size_w);

          hstart = min(max(hstart + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height));
          hend = min(max(hend + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height));
          wstart = min(max(wstart + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width));
          wend = min(max(wend + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width));

          bool in_bin = (w > wstart - 1.0 && w < wend + 1.0 &&
                      h > hstart - 1.0 && h < hend + 1.0);
          if (!in_bin) {
            continue;
          }

          const int pool_index = ph * pooled_width + pw;
          int bottom_index = 0;
          Dtype h_stride = (hend - hstart)/3.0;
          Dtype w_stride = (wend - wstart)/3.0;
          for (Dtype rh = hstart+h_stride; rh <= hend-h_stride+0.01; rh += max(h_stride, 0.01)) {
            for (Dtype rw = wstart+w_stride; rw <= wend-w_stride+0.01; rw += max(w_stride, 0.01)) {
              bottom_index ++;
              if (offset_argmax_data[pool_index] != bottom_index) continue;
              // compute the integer coordinates around (h, w) for bilinear interpolation
              int hlow = min(max(static_cast<int>(floor(rh)), 0), height-1);
              int hhigh = hlow + 1;
              int wleft = min(max(static_cast<int>(floor(rw)), 0), width-1);
              int wright = wleft + 1;
              if (h != hlow && h != hhigh && w != wleft && w != wright) // (w, h) is not around (rw, rh)
                  continue;
              
              Dtype alpha = (hlow == hhigh) ? static_cast<Dtype>(0.5) : (rh - hlow) / (hhigh - hlow);
              Dtype beta = (wleft == wright) ? static_cast<Dtype>(0.5) : (rw - wleft) / (wright - wleft);
              if (h == hlow && w == wleft) gradient += offset_top_diff[pool_index] * (1 - alpha) * (1 - beta);
              else if (h == hlow && w == wright) gradient += offset_top_diff[pool_index] * (1 - alpha) * beta;
              else if (h == hhigh && w == wleft) gradient += offset_top_diff[pool_index] * alpha * (1 - beta);
              else if (h == hhigh && w == wright) gradient += offset_top_diff[pool_index] * alpha * beta;
            }
          }
        }
      }
    }
    bottom_diff[index] += gradient;
  }
}

template<typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIPooling Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ROIAlignBackwardAccKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, argmax_data, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois);
}

}  // namespace cuda

template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  cuda::ROIAlignForward(out, data, bbox, max_idx, spatial_scale);
}

template<typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  cuda::ROIAlignBackwardAcc(in_grad, out_grad, bbox, max_idx, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIAlignParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
