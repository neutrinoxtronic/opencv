// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Batch Normalization layer.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "../op_webnn.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/batch_norm.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class BatchNormLayerImpl CV_FINAL : public BatchNormLayer
{
public:
    Mat origin_weights, origin_bias;
    Mat weights_, bias_;
    UMat umat_weight, umat_bias;
    mutable int dims;

    BatchNormLayerImpl(const LayerParams& params)
        : dims(-1)
    {
        setParamsFrom(params);

        hasWeights = params.get<bool>("has_weight", false);
        hasBias = params.get<bool>("has_bias", false);
        useGlobalStats = params.get<bool>("use_global_stats", true);
        if(params.get<bool>("scale_bias", false))
            hasWeights = hasBias = true;
        epsilon = params.get<float>("eps", 1E-5);
        CV_Assert(epsilon >= 0);

        if (blobs.size() >= 2)
            initWeightsBias();
    }

    virtual void serialize(LayerParams& params) const CV_OVERRIDE
    {
        Layer::serialize(params);
        params.set("epsilon", epsilon);
    }

    void initWeightsBias()
    {
        size_t nblobs = blobs.size();
        CV_Assert(nblobs >= 2);
        for (auto& blob: blobs) {
            CV_Assert(blob.isContinuous() && blob.type() == CV_32F);
        }
        if (nblobs == 4) {
            std::vector<Mat> inputs = {
                Mat(), blobs[2], blobs[3], blobs[0], blobs[1]
            };
            calcScaleShift(inputs, origin_weights, origin_bias, epsilon);
        } else {
            size_t n = blobs[0].total();
            CV_Assert(blobs[1].total() == n);
            float varMeanScale = 1.f;
            if (!hasWeights && !hasBias && blobs.size() > 2 && useGlobalStats) {
                CV_Assert(nblobs == 3);
                varMeanScale = blobs[2].at<float>(0);
                if (varMeanScale != 0)
                    varMeanScale = 1/varMeanScale;
            }

            const size_t biasBlobIndex = blobs.size() - 1;
            const size_t weightsBlobIndex = biasBlobIndex - hasBias;

            if( hasWeights )
            {
                CV_Assert((size_t)weightsBlobIndex < blobs.size());
                const Mat& w = blobs[weightsBlobIndex];
                CV_Assert(w.isContinuous() && w.type() == CV_32F && w.total() == (size_t)n);
            }

            if( hasBias )
            {
                CV_Assert((size_t)biasBlobIndex < blobs.size());
                const Mat& b = blobs[weightsBlobIndex];
                CV_Assert(b.isContinuous() && b.type() == CV_32F && b.total() == (size_t)n);
            }

            const float* meanData = blobs[0].ptr<float>();
            const float* stdData = blobs[1].ptr<float>();
            const float* weightsData = hasWeights ? blobs[weightsBlobIndex].ptr<float>() : 0;
            const float* biasData = hasBias ? blobs[biasBlobIndex].ptr<float>() : 0;

            origin_weights.create(1, (int)n, CV_32F);
            origin_bias.create(1, (int)n, CV_32F);

            float* dstWeightsData = origin_weights.ptr<float>();
            float* dstBiasData = origin_bias.ptr<float>();

            for (size_t i = 0; i < n; ++i)
            {
                float w = (hasWeights ? weightsData[i] : 1.0f) / sqrt(stdData[i] * varMeanScale + epsilon);
                dstWeightsData[i] = w;
                dstBiasData[i] = (hasBias ? biasData[i] : 0.0f) - w * meanData[i] * varMeanScale;
            }
        }
        origin_weights.copyTo(weights_);
        origin_bias.copyTo(bias_);
    }

    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
        origin_weights.reshape(1, 1).copyTo(weights_);
        origin_bias.reshape(1, 1).copyTo(bias_);
    }

    void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        scale =  weights_;
        shift = bias_;
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Mat w, b;
        top->getScaleShift(w, b);
        if (w.empty() && b.empty())
            return false;

        const int numChannels = (int)weights_.total();
        const int numFusedWeights = (int)w.total();
        const int numFusedBias = (int)b.total();

        if ((numFusedWeights != numChannels && numFusedWeights != 1 && !w.empty()) ||
            (numFusedBias != numChannels && numFusedBias != 1 && !b.empty()))
            return false;

        if (!w.empty())
        {
            w = w.reshape(1, 1);
            if (numFusedWeights == 1)
            {
                multiply(weights_, w.at<float>(0), weights_);
                multiply(bias_, w.at<float>(0), bias_);
            }
            else
            {
                multiply(weights_, w, weights_);
                multiply(bias_, w, bias_);
            }
        }
        if (!b.empty())
        {
            b = b.reshape(1, 1);
            if (numFusedBias == 1)
                add(bias_, b.at<float>(0), bias_);
            else
                add(bias_, b.reshape(1, 1), bias_);
        }
        return true;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        dims = inputs[0].size();
        if (!useGlobalStats && inputs[0][0] != 1)
            CV_Error(Error::StsNotImplemented, "Batch normalization in training mode with batch size > 1");
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return preferableTarget == DNN_TARGET_CPU || dims == 4;
#endif
        return (backendId == DNN_BACKEND_OPENCV) ||
               backendId == DNN_BACKEND_CUDA ||
               (backendId == DNN_BACKEND_HALIDE && haveHalide()) ||
               backendId == DNN_BACKEND_WEBNN;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inputs_.depth() == CV_16S);
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        CV_Assert(blobs.size() >= 2);
        CV_Assert(inputs.size() == 1);

        if (use_half && inputs[0].dims == 2)
            return false;

        if (umat_weight.empty())
        {
            weights_.copyTo(umat_weight);
            bias_.copyTo(umat_bias);
        }

        UMat &inpBlob = inputs[0];
        int groups = inpBlob.size[0];
        int channels = inpBlob.size[1];
        int planeSize = 1;
        for (size_t i = 2; i < inpBlob.dims; i++) {
            planeSize *= inpBlob.size[i];
        }

        String opts = (use_half) ? " -DDtype=half" : " -DDtype=float";
        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            if (inpBlob.dims == 2)
            {
                UMat& src = inputs[ii];
                UMat& dst = outputs[ii];
                multiply(src, weights_, dst);
                add(dst, bias_, dst);
            }
            else
            {
                MatShape s = shape(groups * channels, planeSize);
                UMat src = inputs[ii].reshape(1, s.size(), &s[0]);
                UMat dst = outputs[ii].reshape(1, s.size(), &s[0]);
                int number = (s[1] % 8 == 0) ? 8 : ((s[1] % 4 == 0) ? 4 : 1);
                String buildopt = format("-DNUM=%d", number) + opts;
                String kname = format("batch_norm%d", number);
                if (number == 1)
                    buildopt += format(" -Dconvert_T=convert_%s", use_half ? "half" : "float");
                else
                    buildopt += format(" -Dconvert_T=convert_%s%d", use_half ? "half" : "float", number);
                ocl::Kernel kernel(kname.c_str(), ocl::dnn::batchnorm_oclsrc, buildopt);
                if (kernel.empty())
                    return false;
                size_t global[] = { (size_t)s[0], (size_t)(s[1] / number) };
                kernel.set(0, ocl::KernelArg::PtrReadOnly(src));
                kernel.set(1, (int)s[0]);
                kernel.set(2, (int)s[1]);
                kernel.set(3, (int)channels);
                kernel.set(4, ocl::KernelArg::PtrReadOnly(umat_weight));
                kernel.set(5, ocl::KernelArg::PtrReadOnly(umat_bias));
                kernel.set(6, ocl::KernelArg::PtrWriteOnly(dst));
                bool ret = kernel.run_(2, global, NULL, false);
                if (!ret)
                    return false;
            }
        }
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert((inputs.size() == 1 && blobs.size() >= 2) ||
                  (inputs.size() == 5 && blobs.empty()));

        Mat &inpBlob = inputs[0];
        int planeSize = 1;
        for (int i = 2; i < inpBlob.dims; i++) {
            planeSize *= inpBlob.size[i];
        }
        if (blobs.empty())
            calcScaleShift(inputs, weights_, bias_, epsilon);

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &outBlob = outputs[ii];

            for(int num = 0; num < outBlob.size[0]; num++)
            {
                for (int n = 0; n < outBlob.size[1]; n++)
                {
                    float w = weights_.at<float>(n);
                    float b = bias_.at<float>(n);
                    Mat inpBlobPlane(1, planeSize, CV_32F, inpBlob.ptr<float>(num, n));
                    Mat outBlobPlane(1, planeSize, CV_32F, outBlob.ptr<float>(num, n));
                    inpBlobPlane.convertTo(outBlobPlane, CV_32F, w, b);
                }
            }
        }
    }

    void forwardSlice(const float* srcptr, float* dstptr, int len, size_t planeSize, int cn0, int cn1) const CV_OVERRIDE
    {
        for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize )
        {
            int i = 0;
            float w = weights_.at<float>(cn);
            float b = bias_.at<float>(cn);
#if CV_SIMD128
            v_float32x4 wV = v_setall_f32(w), bV = v_setall_f32(b);
            for( ; i <= len - 16; i += 16 )
            {
                v_float32x4 x0 = v_load(srcptr + i);
                v_float32x4 x1 = v_load(srcptr + i + 4);
                v_float32x4 x2 = v_load(srcptr + i + 8);
                v_float32x4 x3 = v_load(srcptr + i + 12);
                x0 = v_muladd(x0, wV, bV);
                x1 = v_muladd(x1, wV, bV);
                x2 = v_muladd(x2, wV, bV);
                x3 = v_muladd(x3, wV, bV);
                v_store(dstptr + i, x0);
                v_store(dstptr + i + 4, x1);
                v_store(dstptr + i + 8, x2);
                v_store(dstptr + i + 12, x3);
            }
#endif
            for( ; i < len; i++ )
                dstptr[i] = w * srcptr[i] + b;
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        return make_cuda_node<cuda4dnn::BatchNormOp>(preferableTarget, std::move(context->stream), weights_, bias_);
    }
#endif

    virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node) CV_OVERRIDE
    {
        switch (node->backendId)
        {
            case DNN_BACKEND_HALIDE:
            {
#ifdef HAVE_HALIDE
                auto base = node.dynamicCast<HalideBackendNode>();
                Halide::Func& input = base->funcs.back();
                Halide::Var x("x"), y("y"), c("c"), n("n");
                Halide::Func top = attachHalide(input(x, y, c, n));
                return Ptr<BackendNode>(new HalideBackendNode(base, top));
#endif  // HAVE_HALIDE
                break;
            }
        }
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> input = halideBuffer(inputs[0]);
        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = attachHalide(input(x, y, c, n));
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_HALIDE
    // attachHalide can work both with Halide::Buffer and Halide::Func. In the
    // second case it will be a fusion.
    Halide::Func attachHalide(const Halide::Expr& input)
    {
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Var x("x"), y("y"), c("c"), n("n");

        const int numChannels = weights_.total();
        auto weights = wrapToHalideBuffer(weights_, {numChannels});
        auto bias = wrapToHalideBuffer(bias_, {numChannels});
        top(x, y, c, n) = input * weights(c) + bias(c);
        return top;
    }
#endif  // HAVE_HALIDE


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<size_t> shape(ieInpNode->get_shape().size(), 1);
        shape[1] = weights_.total();
        auto weight = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape(shape), weights_.data);
        auto bias = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape(shape), bias_.data);
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2021_2)
        auto scale_node = std::make_shared<ngraph::op::v1::Multiply>(ieInpNode, weight, ngraph::op::AutoBroadcastType::NUMPY);
#else
        auto scale_node = std::make_shared<ngraph::op::v0::Multiply>(ieInpNode, weight, ngraph::op::AutoBroadcastType::NUMPY);
#endif
        auto scale_shift = std::make_shared<ngraph::op::v1::Add>(scale_node, bias, ngraph::op::AutoBroadcastType::NUMPY);
        return Ptr<BackendNode>(new InfEngineNgraphNode(scale_shift));
    }
#endif  // HAVE_DNN_NGRAPH

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        params.set("input_scale", scales[0][0]);
        params.set("input_zeropoint", zeropoints[0][0]);
        params.set("eps", epsilon);

        params.blobs.clear();
        params.blobs.push_back(origin_weights);
        params.blobs.push_back(origin_bias);
        return true;
    }

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        std::vector<int32_t> weights_shape = webnn::getShape(weights_);
        ml::Operand weights = webnn::BuildConstant(webnnGraphBuilder, weights_shape, weights_.data, weights_.total()*weights_.elemSize(), ml::OperandType::Float32);
        std::vector<int32_t> shape(dims, 1);
        shape[1] = weights_shape[1];
        ml::Operand weights_reshaped = webnnGraphBuilder.Reshape(weights, shape.data(), shape.size());
        ml::Operand mul_res = webnnGraphBuilder.Mul(webnnInpOperand, weights_reshaped);
        std::vector<int32_t> bias_shape = webnn::getShape(bias_);
        ml::Operand bias = webnn::BuildConstant(webnnGraphBuilder, bias_shape, bias_.data, bias_.total()*bias_.elemSize(), ml::OperandType::Float32);
        shape[1] = bias_shape[1];
        ml::Operand bias_reshaped = webnnGraphBuilder.Reshape(bias, shape.data(), shape.size());
        ml::Operand add_res = webnnGraphBuilder.Add(mul_res, bias_reshaped);
        return Ptr<BackendNode>(new WebnnBackendNode(add_res));
    }
#endif

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 3*total(inputs[i]);
        }
        return flops;
    }

private:
    bool useGlobalStats;
};

void BatchNormLayer::calcScaleShift(InputArrayOfArrays inputs_,
                                    OutputArray scale_, OutputArray shift_,
                                    float epsilon)
{
    std::vector<Mat> inputs;
    inputs_.getMatVector(inputs);
    CV_Assert(inputs.size() == 5);
    int i, C;
    C = (int)inputs[1].total();
    C = std::max(C, (int)inputs[2].total());
    C = std::max(C, (int)inputs[3].total());
    C = std::max(C, (int)inputs[4].total());
    for (i = 1; i < 5; i++) {
        if (inputs[i].empty())
            continue;
        if (C < 0)
            C = (int)inputs[i].total();
        CV_Assert(inputs[i].total() == C || inputs[i].total() == 1);
        CV_Assert(inputs[i].type() == CV_32F);
        CV_Assert(inputs[i].isContinuous());
    }
    CV_Assert(C > 0);
    scale_.create(1, C, CV_32F);
    shift_.create(1, C, CV_32F);
    Mat scale = scale_.getMat();
    Mat shift = shift_.getMat();
    const float* inpscale = inputs[1].ptr<float>();
    const float* inpbias = inputs[2].ptr<float>();
    const float* inpmean = inputs[3].ptr<float>();
    const float* inpvar = inputs[4].ptr<float>();
    int vector_scale = inputs[1].total() > 1;
    int vector_bias = inputs[2].total() > 1;
    int vector_mean = inputs[3].total() > 1;
    int vector_var = inputs[4].total() > 1;
    float* wptr = scale.ptr<float>();
    float* bptr = shift.ptr<float>();
    for (i = 0; i < C; i++) {
        float w = (inpscale ? inpscale[i*vector_scale] : 1.f)/sqrtf((inpvar ? inpvar[i*vector_var] : 1.f) + epsilon);
        float b = (inpbias ? inpbias[i*vector_bias] : 0.f) - (inpmean ? inpmean[i*vector_mean] : 0.f)*w;
        wptr[i] = w;
        bptr[i] = b;
    }
}



Ptr<BatchNormLayer> BatchNormLayer::create(const LayerParams& params)
{
    return Ptr<BatchNormLayer>(new BatchNormLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
