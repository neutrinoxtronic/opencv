// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024, OpenCV Team, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

namespace opencv_test { namespace {

class Layer_Test_01D: public testing::TestWithParam<tuple<int>>
{
public:
    int dims;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    float inp_value;
    Mat input;
    LayerParams lp;

    void SetUp()
    {
        dims = get<0>(GetParam());
        input_shape = {dims};
        output_shape = {dims};

        // generate random positeve value from 1 to 10
        RNG& rng = TS::ptr()->get_rng();
        inp_value = rng.uniform(1.0, 10.0); // random uniform value
        input = Mat(dims, input_shape.data(), CV_32F, inp_value);
    }

    void TestLayer(Ptr<Layer> layer, std::vector<Mat> &inputs, const Mat& output_ref){
        std::vector<Mat> outputs;
        runLayer(layer, inputs, outputs);
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    }

};

TEST_P(Layer_Test_01D, Scale)
{

    lp.type = "Scale";
    lp.name = "scaleLayer";
    lp.set("axis", 0);
    lp.set("mode", "scale");
    lp.set("bias_term", false);
    Ptr<ScaleLayer> layer = ScaleLayer::create(lp);

    Mat weight = Mat(dims, output_shape.data(), CV_32F, 2.0);
    std::vector<Mat> inputs{input, weight};
    Mat output_ref = input.mul(weight);

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ReLU6)
{

    lp.type = "ReLU6";
    lp.name = "ReLU6Layer";
    lp.set("min_value", 0.0);
    lp.set("max_value", 1.0);
    Ptr<ReLU6Layer> layer = ReLU6Layer::create(lp);

    Mat output_ref(dims, output_shape.data(), CV_32F, 1.0);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Clip)
{

    lp.type = "Clip";
    lp.name = "clipLayer";
    lp.set("min_value", 0.0);
    lp.set("max_value", 1.0);
    Ptr<ReLU6Layer> layer = ReLU6Layer::create(lp);

    Mat output_ref(dims, output_shape.data(), CV_32F, 1.0);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ReLU)
{

    lp.type = "ReLU";
    lp.name = "reluLayer";
    lp.set("negative_slope", 0.0);
    Ptr<ReLULayer> layer = ReLULayer::create(lp);

    Mat output_ref(dims, output_shape.data(), CV_32F, inp_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Gelu)
{

    lp.type = "Gelu";
    lp.name = "geluLayer";
    Ptr<GeluLayer> layer = GeluLayer::create(lp);

    float value = inp_value * 0.5 * (std::erf(inp_value * 1 / std::sqrt(2.0)) + 1.0);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, GeluApprox)
{

    lp.type = "GeluApprox";
    lp.name = "geluApproxLayer";
    Ptr<GeluApproximationLayer> layer = GeluApproximationLayer::create(lp);

    float value = inp_value * 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (inp_value + 0.044715 * std::pow(inp_value, 3))));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sigmoid)
{

    lp.type = "Sigmoid";
    lp.name = "sigmoidLayer";
    Ptr<SigmoidLayer> layer = SigmoidLayer::create(lp);

    float value = 1.0 / (1.0 + std::exp(-inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Tanh)
{

    lp.type = "TanH";
    lp.name = "TanHLayer";
    Ptr<Layer> layer = TanHLayer::create(lp);


    float value = std::tanh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Swish)
{

    lp.type = "Swish";
    lp.name = "SwishLayer";
    Ptr<Layer> layer = SwishLayer::create(lp);

    float value = inp_value / (1 + std::exp(-inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Mish)
{

    lp.type = "Mish";
    lp.name = "MishLayer";
    Ptr<Layer> layer = MishLayer::create(lp);

    float value = inp_value * std::tanh(std::log(1 + std::exp(inp_value)));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ELU)
{

    lp.type = "ELU";
    lp.name = "eluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ELULayer::create(lp);

    float value = inp_value > 0 ? inp_value : std::exp(inp_value) - 1;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Abs)
{

    lp.type = "Abs";
    lp.name = "absLayer";
    Ptr<Layer> layer = AbsLayer::create(lp);

    float value = std::abs(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, BNLL)
{

    lp.type = "BNLL";
    lp.name = "bnllLayer";
    Ptr<Layer> layer = BNLLLayer::create(lp);

    float value = std::log(1 + std::exp(inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Ceil)
{

    lp.type = "Ceil";
    lp.name = "ceilLayer";
    Ptr<Layer> layer = CeilLayer::create(lp);

    float value = std::ceil(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Floor)
{

    lp.type = "Floor";
    lp.name = "floorLayer";
    Ptr<Layer> layer = FloorLayer::create(lp);

    float value = std::floor(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Log)
{

    lp.type = "Log";
    lp.name = "logLayer";
    Ptr<Layer> layer = LogLayer::create(lp);

    float value = std::log(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Round)
{

    lp.type = "Round";
    lp.name = "roundLayer";
    Ptr<Layer> layer = RoundLayer::create(lp);

    float value = std::round(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sqrt)
{

    lp.type = "Sqrt";
    lp.name = "sqrtLayer";
    Ptr<Layer> layer = SqrtLayer::create(lp);

    float value = std::sqrt(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Acos)
{

    lp.type = "Acos";
    lp.name = "acosLayer";
    Ptr<Layer> layer = AcosLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (inp_value) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(dims, input_shape.data(), CV_32F, inp_value);

    float value = std::acos(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Acosh)
{

    lp.type = "Acosh";
    lp.name = "acoshLayer";
    Ptr<Layer> layer = AcoshLayer::create(lp);

    float value = std::acosh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Asin)
{

    lp.type = "Asin";
    lp.name = "asinLayer";
    Ptr<Layer> layer = AsinLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (inp_value) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(dims, input_shape.data(), CV_32F, inp_value);

    float value = std::asin(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Asinh)
{

    lp.type = "Asinh";
    lp.name = "asinhLayer";
    Ptr<Layer> layer = AsinhLayer::create(lp);

    float value = std::asinh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Atan)
{

    lp.type = "Atan";
    lp.name = "atanLayer";
    Ptr<Layer> layer = AtanLayer::create(lp);

    float value = std::atan(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Cos)
{

    lp.type = "Cos";
    lp.name = "cosLayer";
    Ptr<Layer> layer = CosLayer::create(lp);

    float value = std::cos(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Cosh)
{

    lp.type = "Cosh";
    lp.name = "coshLayer";
    Ptr<Layer> layer = CoshLayer::create(lp);

    float value = std::cosh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sin)
{

    lp.type = "Sin";
    lp.name = "sinLayer";
    Ptr<Layer> layer = SinLayer::create(lp);

    float value = std::sin(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sinh)
{

    lp.type = "Sinh";
    lp.name = "sinhLayer";
    Ptr<Layer> layer = SinhLayer::create(lp);

    float value = std::sinh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Tan)
{

    lp.type = "Tan";
    lp.name = "tanLayer";
    Ptr<Layer> layer = TanLayer::create(lp);

    float value = std::tan(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Erf)
{

    lp.type = "Erf";
    lp.name = "erfLayer";
    Ptr<Layer> layer = ErfLayer::create(lp);

    float out_value = std::erf(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Reciprocal)
{

    lp.type = "Reciprocal";
    lp.name = "reciprocalLayer";
    Ptr<Layer> layer = ReciprocalLayer::create(lp);

    float out_value = 1/inp_value;
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, HardSwish)
{

    lp.type = "HardSwish";
    lp.name = "hardSwishLayer";
    Ptr<Layer> layer = HardSwishLayer::create(lp);

    float out_value = inp_value * std::max(0.0f, std::min(6.0f, inp_value + 3.0f)) / 6.0f;
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Softplus)
{

    lp.type = "Softplus";
    lp.name = "softplusLayer";
    Ptr<Layer> layer = SoftplusLayer::create(lp);

    float out_value = std::log(1 + std::exp(inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, SoftSign)
{

    lp.type = "Softsign";
    lp.name = "softsignLayer";
    Ptr<Layer> layer = SoftsignLayer::create(lp);

    float out_value = inp_value / (1 + std::abs(inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, CELU)
{

    lp.type = "CELU";
    lp.name = "celuLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = CeluLayer::create(lp);

    float out_value = inp_value < 0 ? std::exp(inp_value) - 1 : inp_value;
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, HardSigmoid)
{

    lp.type = "HardSigmoid";
    lp.name = "hardSigmoidLayer";
    Ptr<Layer> layer = HardSigmoidLayer::create(lp);

    float out_value = std::max(0.0f, std::min(1.0f, 0.2f * inp_value + 0.5f));
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, SELU)
{

    lp.type = "SELU";
    lp.name = "seluLayer";
    lp.set("alpha", 1.6732631921768188);
    lp.set("gamma", 1.0507009873554805);
    Ptr<Layer> layer = SeluLayer::create(lp);


    double inp_value_double = static_cast<double>(inp_value); // Ensure the input is treated as double for the computation

    double value_double = 1.0507009873554805 * (inp_value_double > 0 ? inp_value_double : 1.6732631921768188 * (std::exp(inp_value_double / 1.0) - 1));

    float value = static_cast<float>(value_double);

    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ThresholdedReLU)
{

    lp.type = "ThresholdedReLU";
    lp.name = "thresholdedReluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ThresholdedReluLayer::create(lp);

    float value = inp_value > 1.0 ? inp_value : 0.0;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Power)
{

    lp.type = "Power";
    lp.name = "powerLayer";
    lp.set("power", 2.0);
    lp.set("scale", 1.0);
    lp.set("shift", 0.0);
    Ptr<Layer> layer = PowerLayer::create(lp);

    float value = std::pow(inp_value, 2.0);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Exp)
{

    lp.type = "Exp";
    lp.name = "expLayer";
    Ptr<Layer> layer = ExpLayer::create(lp);

    float out_value = std::exp(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sign)
{

    lp.type = "Sign";
    lp.name = "signLayer";
    Ptr<Layer> layer = SignLayer::create(lp);

    float value = inp_value > 0 ? 1.0 : 0.0;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Shrink)
{

    lp.type = "Shrink";
    lp.name = "shrinkLayer";
    lp.set("lambda", 0.5);
    lp.set("bias", 0.5);
    Ptr<Layer> layer = ShrinkLayer::create(lp);

    float value = inp_value > 0.5 ? inp_value - 0.5 : 0.0;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ChannelsPReLU)
{

    lp.type = "ChannelsPReLU";
    lp.name = "ChannelsPReLULayer";
    Mat alpha = Mat(1, 3, CV_32F, 0.5);
    lp.blobs.push_back(alpha);
    Ptr<Layer> layer = ChannelsPReLULayer::create(lp);

    float value = inp_value > 0 ? inp_value : 0.5 * inp_value;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Test_01D, Values(0, 1));

typedef testing::TestWithParam<tuple<int, int>> Layer_Gather_Test;
TEST_P(Layer_Gather_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    int axis = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Gather";
    lp.name = "GatherLayer";
    lp.set("axis", axis);
    lp.set("real_ndims", 1);

    Ptr<GatherLayer> layer = GatherLayer::create(lp);

    std::vector<int> input_shape = {dims};
    std::vector<int> indices_shape = {1};
    std::vector<int> output_shape = {dims};

    Mat input(dims, input_shape.data(), CV_32F, 1.0);
    cv::randu(input, 0.0, 1.0);

    Mat indices(indices_shape, CV_32SC1, 0.0);
    Mat output_ref(dims, output_shape.data(), CV_32F, input.at<float>(0, 0));

    std::vector<Mat> inputs{input, indices};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(output_ref.size, outputs[0].size);
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Gather_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values(0)
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_Arg_Test;
TEST_P(Layer_Arg_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Arg";
    lp.name = "arg" + operation + "_Layer";
    lp.set("op", operation);
    lp.set("axis", 0);
    lp.set("keepdims", 0);
    lp.set("select_last_index", 0);

    Ptr<ArgLayer> layer = ArgLayer::create(lp);
    std::vector<int> input_shape = {dims};
    std::vector<int> output_shape = {1};

    Mat input(dims, input_shape.data(), CV_32F, 1);
    Mat output_ref(dims, output_shape.data(),  CV_32F, 0);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(output_ref.size , outputs[0].size);
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Arg_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values( "max", "min")
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_NaryElemwise_Test;
TEST_P(Layer_NaryElemwise_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<NaryEltwiseLayer> layer = NaryEltwiseLayer::create(lp);

    std::vector<int> input_shape = {dims};
    Mat input1(dims, input_shape.data(), CV_32F, 0.0);
    Mat input2(dims, input_shape.data(), CV_32F, 0.0);
    cv::randu(input1, 0.0, 1.0);
    cv::randu(input2, 0.0, 1.0);

    Mat output_ref;
    if (operation == "sum") {
        output_ref = input1 + input2;
    } else if (operation == "mul") {
        output_ref = input1.mul(input2);
    } else if (operation == "div") {
        output_ref = input1 / input2;
    } else if (operation == "sub") {
        output_ref = input1 - input2;
    } else {
        output_ref = Mat();
    }
    std::vector<Mat> inputs{input1, input2};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    if (!output_ref.empty()) {
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_NaryElemwise_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values("div", "mul", "sum", "sub")
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_Elemwise_Test;
TEST_P(Layer_Elemwise_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<EltwiseLayer> layer = EltwiseLayer::create(lp);

    std::vector<int> input_shape = {dims};
    Mat input1(dims, input_shape.data(), CV_32F);
    Mat input2(dims, input_shape.data(), CV_32F);
    cv::randu(input1, 0.0, 1.0);
    cv::randu(input2, 0.0, 1.0);

    // Dynamically select the operation
    Mat output_ref;
    if (operation == "sum") {
        output_ref = input1 + input2;
    } else if (operation == "max") {
        output_ref = cv::max(input1, input2);
    } else if (operation == "min") {
        output_ref = cv::min(input1, input2);
    } else if (operation == "prod") {
        output_ref = input1.mul(input2);
    } else if (operation == "div") {
        output_ref = input1 / input2;
    } else {
        output_ref = Mat();
    }

    std::vector<Mat> inputs{input1, input2};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);

    if (!output_ref.empty()) {
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Elemwise_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values("div", "prod", "max", "min", "sum")
));

TEST(Layer_Reshape_Test, Accuracy)
{
    LayerParams lp;
    lp.type = "Reshape";
    lp.name = "ReshapeLayer";
    lp.set("axis", 0); // Set axis to 0 to start reshaping from the first dimension
    lp.set("num_axes", -1); // Set num_axes to -1 to indicate all following axes are included in the reshape
    int newShape[] = {1};
    lp.set("dim", DictValue::arrayInt(newShape, 1));

    Ptr<ReshapeLayer> layer = ReshapeLayer::create(lp);

    std::vector<int> input_shape = {0};

    Mat input(0, input_shape.data(), CV_32F);
    randn(input, 0.0, 1.0);
    Mat output_ref(1, newShape, CV_32F, input.data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Split_Test;
TEST_P(Layer_Split_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "Split";
    lp.name = "SplitLayer";
    int top_count = 2; // 2 is for simplicity
    lp.set("top_count", top_count);
    Ptr<SplitLayer> layer = SplitLayer::create(lp);

    std::vector<int> input_shape = std::get<0>(GetParam());

    Mat input(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    Mat output_ref = Mat(input_shape.size(), input_shape.data(), CV_32F, input.data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    for (int i = 0; i < top_count; i++)
    {
        ASSERT_EQ(shape(output_ref), shape(outputs[i]));
        normAssert(output_ref, outputs[i]);
    }
}
INSTANTIATE_TEST_CASE_P(/*nothting*/, Layer_Split_Test,
                        testing::Values(
                            std::vector<int>({}),
                            std::vector<int>({1}),
                            std::vector<int>({1, 4}),
                            std::vector<int>({1, 5}),
                            std::vector<int>({4, 1}),
                            std::vector<int>({4, 5})
));

typedef testing::TestWithParam<tuple<std::vector<int>, std::vector<int>>> Layer_Expand_Test;
TEST_P(Layer_Expand_Test, Accuracy_ND) {

    std::vector<int> input_shape = get<0>(GetParam());
    std::vector<int> target_shape = get<1>(GetParam());
    if (input_shape.size() >= target_shape.size()) // Skip if input shape is already larger than target shape
        return;

    LayerParams lp;
    lp.type = "Expand";
    lp.name = "ExpandLayer";
    lp.set("shape", DictValue::arrayInt(&target_shape[0], target_shape.size()));

    Ptr<ExpandLayer> layer = ExpandLayer::create(lp);
    Mat input(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    cv::Mat output_ref(target_shape, CV_32F, input.data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Expand_Test, Combine(
/*input blob shape*/ testing::Values(
        std::vector<int>({}),
        std::vector<int>({1}),
        std::vector<int>({1, 1}),
        std::vector<int>({1, 1, 1})
    ),
/*output blob shape*/ testing::Values(
        std::vector<int>({1}),
        std::vector<int>({1, 1}),
        std::vector<int>({1, 1, 1}),
        std::vector<int>({1, 1, 1, 1})
    )
));

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Concat_Test;
TEST_P(Layer_Concat_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "Concat";
    lp.name = "ConcatLayer";
    lp.set("axis", 0);

    Ptr<ConcatLayer> layer = ConcatLayer::create(lp);

    std::vector<int> input_shape = get<0>(GetParam());
    std::vector<int> output_shape = {3};

    Mat input1(input_shape.size(), input_shape.data(), CV_32F, 1.0);
    Mat input2(input_shape.size(), input_shape.data(), CV_32F, 2.0);
    Mat input3(input_shape.size(), input_shape.data(), CV_32F, 3.0);

    float data[] = {1.0, 2.0, 3.0};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input1, input2, input3};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Concat_Test,
/*input blob shape*/    testing::Values(
    std::vector<int>({}),
    std::vector<int>({1})
));

typedef testing::TestWithParam<tuple<std::vector<int>, int>> Layer_Softmax_Test;
TEST_P(Layer_Softmax_Test, Accuracy_01D) {

    int axis = get<1>(GetParam());
    std::vector<int> input_shape = get<0>(GetParam());
    if ((input_shape.size() == 0 && axis == 1) ||
        (!input_shape.empty() && input_shape.size() == 2 && input_shape[0] > 1 && axis == 1) ||
        (!input_shape.empty() && input_shape[0] > 1 && axis == 0)) // skip since not valid case
        return;

    LayerParams lp;
    lp.type = "Softmax";
    lp.name = "softmaxLayer";
    lp.set("axis", axis);
    Ptr<SoftmaxLayer> layer = SoftmaxLayer::create(lp);

    Mat input = Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    Mat output_ref;
    cv::exp(input, output_ref);
    if (axis == 1){
        cv::divide(output_ref, cv::sum(output_ref), output_ref);
    } else {
        cv::divide(output_ref, output_ref, output_ref);
    }

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Softmax_Test, Combine(
    /*input blob shape*/
    testing::Values(
        std::vector<int>({}),
        std::vector<int>({1}),
        std::vector<int>({4}),
        std::vector<int>({1, 4}),
        std::vector<int>({4, 1})
        ),
    /*Axis */
    testing::Values(0, 1)
));

typedef testing::TestWithParam<tuple<std::vector<int>, std::string>> Layer_Scatter_Test;
TEST_P(Layer_Scatter_Test, Accuracy1D) {

    std::vector<int> input_shape = get<0>(GetParam());
    std::string opr = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Scatter";
    lp.name = "addLayer";
    lp.set("axis", 0);
    lp.set("reduction", opr);
    Ptr<ScatterLayer> layer = ScatterLayer::create(lp);

    cv::Mat input = cv::Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    int indices[] = {3, 2, 1, 0};
    cv::Mat indices_mat(input_shape.size(), input_shape.data(), CV_32S, indices);
    cv::Mat output(input_shape.size(), input_shape.data(), CV_32F, 0.0);

    // create reference output
    cv::Mat output_ref(input_shape, CV_32F, 0.0);
    for (int i = 0; i < input_shape[0]; i++){
        output_ref.at<float>(indices[i]) = input.at<float>(i);
    }

    if (opr == "add"){
        output_ref += output;
    } else if (opr == "mul"){
        output_ref = output.mul(output_ref);
    } else if (opr == "max"){
        cv::max(output_ref, output, output_ref);
    } else if (opr == "min"){
        cv::min(output_ref, output, output_ref);
    }

    std::vector<Mat> inputs{output, indices_mat, input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Scatter_Test, Combine(
/*input blob shape*/    testing::Values(std::vector<int>{4},
                                        std::vector<int>{1, 4}),
/*reduce*/              Values("none", "add", "mul", "max", "min")
));



typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Permute_Test;
TEST_P(Layer_Permute_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "Permute";
    lp.name = "PermuteLayer";

    int order[] = {0}; // Since it's a 0D tensor, the order remains [0]
    lp.set("order", DictValue::arrayInt(order, 1));
    Ptr<PermuteLayer> layer = PermuteLayer::create(lp);

    std::vector<int> input_shape = get<0>(GetParam());

    Mat input = Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);
    Mat output_ref = input.clone();

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/,  Layer_Permute_Test,
/*input blob shape*/ testing::Values(
            std::vector<int>{},
            std::vector<int>{1},
            std::vector<int>{1, 4},
            std::vector<int>{4, 1}
));

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Slice_Test;
TEST_P(Layer_Slice_Test, Accuracy_1D){

    LayerParams lp;
    lp.type = "Slice";
    lp.name = "SliceLayer";

    std::vector<int> input_shape = get<0>(GetParam());

    int splits = 2;
    int axis = (input_shape.size() > 1 ) ? 1 : 0;

    lp.set("axis", axis);
    lp.set("num_split", splits);

    Ptr<SliceLayer> layer = SliceLayer::create(lp);
    std::vector<int> output_shape;
    if (input_shape.size() > 1)
        output_shape = {1, input_shape[1] / splits};
    else
        output_shape = {input_shape[0] / splits};

    cv::Mat input = cv::Mat(input_shape, CV_32F);
    cv::randu(input, 0.0, 1.0);

    std::vector<cv::Mat> output_refs;
    for (int i = 0; i < splits; ++i){
        output_refs.push_back(cv::Mat(output_shape, CV_32F));
        if (input_shape.size() > 1 ) {
            for (int j = 0; j < output_shape[1]; ++j){
                output_refs[i].at<float>(j) = input.at<float>(i * output_shape[1] + j);
            }
        } else {
            for (int j = 0; j < output_shape[0]; ++j){
                output_refs[i].at<float>(j) = input.at<float>(i * output_shape[0] + j);
            }
        }
    }

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    for (int i = 0; i < splits; ++i){
        ASSERT_EQ(shape(output_refs[i]), shape(outputs[i]));
        normAssert(output_refs[i], outputs[i]);
    }
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Slice_Test,
/*input blob shape*/    testing::Values(
                std::vector<int>({4}),
                std::vector<int>({1, 4})
));

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_FullyConnected_Test;
TEST_P(Layer_FullyConnected_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "InnerProduct";
    lp.name = "InnerProductLayer";
    lp.set("num_output", 1);
    lp.set("bias_term", false);
    lp.set("axis", 0);

    std::vector<int> input_shape = get<0>(GetParam());

    RNG& rng = TS::ptr()->get_rng();
    float inp_value = rng.uniform(0.0, 10.0);
    Mat weights(std::vector<int>{total(input_shape), 1}, CV_32F, inp_value);
    lp.blobs.push_back(weights);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("InnerProduct", lp);

    Mat input(input_shape.size(), input_shape.data(), CV_32F);
    randn(input, 0, 1);
    Mat output_ref = input.reshape(1, 1) * weights;
    output_ref.dims = 1;

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothting*/, Layer_FullyConnected_Test,
                        testing::Values(
                            std::vector<int>({}),
                            std::vector<int>({1}),
                            std::vector<int>({4})
));

typedef testing::TestWithParam<std::vector<int>> Layer_BatchNorm_Test;
TEST_P(Layer_BatchNorm_Test, Accuracy_01D)
{
    std::vector<int> input_shape = GetParam();

    // Layer parameters
    LayerParams lp;
    lp.type = "BatchNorm";
    lp.name = "BatchNormLayer";
    lp.set("has_weight", false);
    lp.set("has_bias", false);

    RNG& rng = TS::ptr()->get_rng();
    float inp_value = rng.uniform(0.0, 10.0);

    Mat meanMat(input_shape.size(), input_shape.data(), CV_32F, inp_value);
    Mat varMat(input_shape.size(), input_shape.data(), CV_32F, inp_value);
    vector<Mat> blobs = {meanMat, varMat};
    lp.blobs = blobs;

    // Create the layer
    Ptr<Layer> layer = BatchNormLayer::create(lp);

    Mat input(input_shape.size(), input_shape.data(), CV_32F, 1.0);
    cv::randn(input, 0, 1);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    //create output_ref to compare with outputs
    Mat output_ref = input.clone();
    cv::sqrt(varMat + 1e-5, varMat);
    output_ref = (output_ref - meanMat) / varMat;

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);

}
INSTANTIATE_TEST_CASE_P(/*nothting*/, Layer_BatchNorm_Test,
                        testing::Values(
                            std::vector<int>({}),
                            std::vector<int>({4}),
                            std::vector<int>({1, 4}),
                            std::vector<int>({4, 1})
));


typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Const_Test;
TEST_P(Layer_Const_Test, Accuracy_01D)
{
    std::vector<int> input_shape = get<0>(GetParam());

    LayerParams lp;
    lp.type = "Const";
    lp.name = "ConstLayer";

    Mat constBlob = Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(constBlob, 0.0, 1.0);
    Mat output_ref = constBlob.clone();

    lp.blobs.push_back(constBlob);
    Ptr<Layer> layer = ConstLayer::create(lp);

    std::vector<Mat> inputs; // No inputs are needed for a ConstLayer
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Const_Test, testing::Values(
    std::vector<int>({}),
    std::vector<int>({1}),
    std::vector<int>({1, 4}),
    std::vector<int>({4, 1})
    ));

}}
