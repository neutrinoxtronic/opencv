/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#ifdef HAVE_OPENCL

using namespace testing;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cvtColor

PARAM_TEST_CASE(CvtColor, MatDepth, bool)
{
    int depth;
    bool use_roi;

    // src mat
    Mat src;
    Mat dst;

    // src mat with roi
    Mat src_roi;
    Mat dst_roi;

    // ocl dst mat for testing
    ocl::oclMat gsrc_whole;
    ocl::oclMat gdst_whole;

    // ocl mat with roi
    ocl::oclMat gsrc_roi;
    ocl::oclMat gdst_roi;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        use_roi = GET_PARAM(1);
    }

    virtual void random_roi(int channelsIn, int channelsOut)
    {
        const int srcType = CV_MAKE_TYPE(depth, channelsIn);
        const int dstType = CV_MAKE_TYPE(depth, channelsOut);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, srcType, 2, 100);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dstType, 5, 16);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst, roiSize, dstBorder);
    }

    void Near(double threshold = 1e-3)
    {
        Mat whole, roi;
        gdst_whole.download(whole);
        gdst_roi.download(roi);

        EXPECT_MAT_NEAR(dst, whole, threshold);
        EXPECT_MAT_NEAR(dst_roi, roi, threshold);
    }

    void doTest(int channelsIn, int channelsOut, int code)
    {
        for (int j = 0; j < LOOP_TIMES; j++)
        {
            random_roi(channelsIn, channelsOut);

            cvtColor(src_roi, dst_roi, code, channelsOut);
            ocl::cvtColor(gsrc_roi, gdst_roi, code, channelsOut);

            Near();
        }
    }
};

#define CVTCODE(name) COLOR_ ## name

// RGB[A] <-> BGR[A]

OCL_TEST_P(CvtColor, BGR2BGRA) { doTest(3, 4, CVTCODE(BGR2BGRA)); }
OCL_TEST_P(CvtColor, RGB2RGBA) { doTest(3, 4, CVTCODE(BGR2BGRA)); }
OCL_TEST_P(CvtColor, BGRA2BGR) { doTest(4, 3, CVTCODE(BGRA2BGR)); }
OCL_TEST_P(CvtColor, RGBA2RGB) { doTest(4, 3, CVTCODE(BGRA2BGR)); }
OCL_TEST_P(CvtColor, BGR2RGBA) { doTest(3, 4, CVTCODE(BGR2RGBA)); }
OCL_TEST_P(CvtColor, RGB2BGRA) { doTest(3, 4, CVTCODE(BGR2RGBA)); }
OCL_TEST_P(CvtColor, RGBA2BGR) { doTest(4, 3, CVTCODE(RGBA2BGR)); }
OCL_TEST_P(CvtColor, BGRA2RGB) { doTest(4, 3, CVTCODE(RGBA2BGR)); }
OCL_TEST_P(CvtColor, BGR2RGB) { doTest(3, 3, CVTCODE(BGR2RGB)); }
OCL_TEST_P(CvtColor, RGB2BGR) { doTest(3, 3, CVTCODE(BGR2RGB)); }
OCL_TEST_P(CvtColor, BGRA2RGBA) { doTest(4, 4, CVTCODE(BGRA2RGBA)); }
OCL_TEST_P(CvtColor, RGBA2BGRA) { doTest(4, 4, CVTCODE(BGRA2RGBA)); }

// RGB <-> Gray

OCL_TEST_P(CvtColor, RGB2GRAY) { doTest(3, 1, CVTCODE(RGB2GRAY)); }
OCL_TEST_P(CvtColor, GRAY2RGB) { doTest(1, 3, CVTCODE(GRAY2RGB)); }
OCL_TEST_P(CvtColor, BGR2GRAY) { doTest(3, 1, CVTCODE(BGR2GRAY)); }
OCL_TEST_P(CvtColor, GRAY2BGR) { doTest(1, 3, CVTCODE(GRAY2BGR)); }
OCL_TEST_P(CvtColor, RGBA2GRAY) { doTest(4, 1, CVTCODE(RGBA2GRAY)); }
OCL_TEST_P(CvtColor, GRAY2RGBA) { doTest(1, 4, CVTCODE(GRAY2RGBA)); }
OCL_TEST_P(CvtColor, BGRA2GRAY) { doTest(4, 1, CVTCODE(BGRA2GRAY)); }
OCL_TEST_P(CvtColor, GRAY2BGRA) { doTest(1, 4, CVTCODE(GRAY2BGRA)); }

// RGB <-> YUV

OCL_TEST_P(CvtColor, RGB2YUV) { doTest(3, 3, CVTCODE(RGB2YUV)); }
OCL_TEST_P(CvtColor, BGR2YUV) { doTest(3, 3, CVTCODE(BGR2YUV)); }
OCL_TEST_P(CvtColor, RGBA2YUV) { doTest(4, 3, CVTCODE(RGB2YUV)); }
OCL_TEST_P(CvtColor, BGRA2YUV) { doTest(4, 3, CVTCODE(BGR2YUV)); }
OCL_TEST_P(CvtColor, YUV2RGB) { doTest(3, 3, CVTCODE(YUV2RGB)); }
OCL_TEST_P(CvtColor, YUV2BGR) { doTest(3, 3, CVTCODE(YUV2BGR)); }
OCL_TEST_P(CvtColor, YUV2RGBA) { doTest(3, 4, CVTCODE(YUV2RGB)); }
OCL_TEST_P(CvtColor, YUV2BGRA) { doTest(3, 4, CVTCODE(YUV2BGR)); }

// RGB <-> YCrCb

OCL_TEST_P(CvtColor, RGB2YCrCb) { doTest(3, 3, CVTCODE(RGB2YCrCb)); }
OCL_TEST_P(CvtColor, BGR2YCrCb) { doTest(3, 3, CVTCODE(BGR2YCrCb)); }
OCL_TEST_P(CvtColor, RGBA2YCrCb) { doTest(4, 3, CVTCODE(RGB2YCrCb)); }
OCL_TEST_P(CvtColor, BGRA2YCrCb) { doTest(4, 3, CVTCODE(BGR2YCrCb)); }
OCL_TEST_P(CvtColor, YCrCb2RGB) { doTest(3, 3, CVTCODE(YCrCb2RGB)); }
OCL_TEST_P(CvtColor, YCrCb2BGR) { doTest(3, 3, CVTCODE(YCrCb2BGR)); }
OCL_TEST_P(CvtColor, YCrCb2RGBA) { doTest(3, 4, CVTCODE(YCrCb2RGB)); }
OCL_TEST_P(CvtColor, YCrCb2BGRA) { doTest(3, 4, CVTCODE(YCrCb2BGR)); }

// RGB <-> XYZ

OCL_TEST_P(CvtColor, RGB2XYZ) { doTest(3, 3, CVTCODE(RGB2XYZ)); }
OCL_TEST_P(CvtColor, BGR2XYZ) { doTest(3, 3, CVTCODE(BGR2XYZ)); }
OCL_TEST_P(CvtColor, RGBA2XYZ) { doTest(4, 3, CVTCODE(RGB2XYZ)); }
OCL_TEST_P(CvtColor, BGRA2XYZ) { doTest(4, 3, CVTCODE(BGR2XYZ)); }

OCL_TEST_P(CvtColor, XYZ2RGB) { doTest(3, 3, CVTCODE(XYZ2RGB)); }
OCL_TEST_P(CvtColor, XYZ2BGR) { doTest(3, 3, CVTCODE(XYZ2BGR)); }
OCL_TEST_P(CvtColor, XYZ2RGBA) { doTest(3, 4, CVTCODE(XYZ2RGB)); }
OCL_TEST_P(CvtColor, XYZ2BGRA) { doTest(3, 4, CVTCODE(XYZ2BGR)); }

// YUV -> RGBA_NV12

struct CvtColor_YUV420 :
        public CvtColor
{
    void random_roi(int channelsIn, int channelsOut)
    {
        const int srcType = CV_MAKE_TYPE(depth, channelsIn);
        const int dstType = CV_MAKE_TYPE(depth, channelsOut);

        Size roiSize = randomSize(1, MAX_VALUE);
        roiSize.width *= 2;
        roiSize.height *= 3;
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, srcType, 2, 100);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dstType, 5, 16);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst, roiSize, dstBorder);
    }
};

OCL_TEST_P(CvtColor_YUV420, YUV2RGBA_NV12) { doTest(1, 4, CV_YUV2RGBA_NV12); }
OCL_TEST_P(CvtColor_YUV420, YUV2BGRA_NV12) { doTest(1, 4, CV_YUV2BGRA_NV12); }
OCL_TEST_P(CvtColor_YUV420, YUV2RGB_NV12) { doTest(1, 3, CV_YUV2RGB_NV12); }
OCL_TEST_P(CvtColor_YUV420, YUV2BGR_NV12) { doTest(1, 3, CV_YUV2BGR_NV12); }

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, CvtColor,
                            testing::Combine(
                                testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32F)),
                                Bool()));

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, CvtColor_YUV420,
                            testing::Combine(
                                testing::Values(MatDepth(CV_8U)),
                                Bool()));

#endif
