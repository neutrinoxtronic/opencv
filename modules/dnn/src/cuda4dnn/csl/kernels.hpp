// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_KERNELS_HPP
#define OPENCV_DNN_CUDA4DNN_KERNELS_HPP

#include "stream.hpp"
#include "memory.hpp"
#include "span.hpp"
#include "tensor.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace kernels {

    template <class T>
    void abs(const Stream& stream, span<T> output, view<T> input);

    template <class T>
    void tanh(const Stream& stream, span<T> output, view<T> input);

    template <class T>
    void sigmoid(const Stream& stream, span<T> output, view<T> input);

    template <class T>
    void bnll(const Stream& stream, span<T> output, view<T> input);

    template <class T>
    void elu(const Stream& stream, span<T> output, view<T> input);

    template <class T>
    void relu(const Stream& stream, span<T> output, view<T> input, T slope);

    template <class T>
    void clipped_relu(const Stream& stream, span<T> output, view<T> input, T floor, T ceiling);

    template <class T>
    void axiswise_relu(const Stream& stream, span<T> output, view<T> input, view<T> slope, std::size_t inner_size);

    template <class T>
    void power(const Stream& stream, span<T> output, view<T> input, T exp, T scale, T shift);

    template <class T>
    void concat(
        const Stream& stream,
        TensorSpan<T> output, std::size_t output_axis_offset,
        TensorView<T> input, std::size_t axis);

    template <class T>
    void concat_with_offsets(const Stream& stream, TensorSpan<T> output, TensorView<T> input, const std::vector<std::size_t>& axis_offsets);

    template <class T>
    void bias1(const Stream& stream, TensorSpan<T> output, TensorView<T> input, T alpha);

    template <class T>
    void biasN(const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> bias);

    template <class T>
    void scale1(const Stream& stream, TensorSpan<T> output, TensorView<T> input, T alpha);

    template <class T>
    void scaleN(const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights);

    template <class T>
    void scale1_with_bias1(const Stream& stream, span<T> output, view<T> input, T alpha, T beta);

    template <class T>
    void scaleN_with_biasN(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights, TensorView<T> bias);

    template <class T>
    void eltwise_max_2(const Stream& stream, span<T> output, view<T> x, view<T> y);

    template <class T>
    void eltwise_sum_2(const Stream& stream, span<T> output, view<T> x, view<T> y);

    template <class T>
    void eltwise_sum_coeff_2(const Stream& stream, span<T> output, T coeff_x, view<T> x, T coeff_y, view<T> y);

    template <class T>
    void eltwise_prod_2(const Stream& stream, span<T> output, view<T> x, view<T> y);

    template <class T>
    void permute(const Stream& stream, TensorSpan<T> output, TensorView<T> input, std::vector<std::size_t> order);

    template <class T>
    void normalize(
        const Stream& stream,
        span<T> output, view<T> input,
        std::size_t outer_size, std::size_t mid_size, std::size_t inner_size, std::size_t norm, T epsilon,
        span<T> workspace);

    template <class T>
    void generate_prior_boxes(
        const Stream& stream,
        span<T> output,
        view<float> boxWidth, view<float> boxHeight, view<float> offsetX, view<float> offsetY, float stepX, float stepY,
        std::vector<float> variance,
        std::size_t numPriors,
        std::size_t layerWidth, std::size_t layerHeight,
        std::size_t imageWidth, std::size_t imageHeight,
        bool normalize, bool clip);

    template <class T>
    void resize_nn(const Stream& stream, TensorSpan<T> output, TensorView<T> input);

    template <class T>
    void resize_bilinear(const Stream& stream, TensorSpan<T> output, TensorView<T> input, float scale_y, float scale_x);

    template <class T>
    void slice(const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::size_t>& offsets);

    template <class T>
    void fill(const Stream& stream, span<T> output, T value);

    template <class T>
    void copy_with_reflection101(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges);

    template <class T>
    void sigmoid_strided(const Stream& stream, span<T> output, view<T> input, std::size_t n, std::size_t stride, std::size_t offset);

    template <class T>
    void softmax_strided(const Stream& stream, span<T> output, view<T> input, std::size_t n, std::size_t stride, std::size_t offset);

    template <class T>
    void region_finalize(const Stream& stream, span<T> output, view<T> input, view<T> bias,
        T object_prob_cutoff, T class_prob_cutoff,
        std::size_t height_norm, std::size_t width_norm,
        std::size_t rows, std::size_t cols,
        std::size_t boxes_per_cell,
        std::size_t box_size,
        std::size_t classes);

}}}}} /* namespace cv::dnn::cuda4dnn::csl::kernels */

#endif /* OPENCV_DNN_CUDA4DNN_KERNELS_HPP */
