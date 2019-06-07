// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include <tuple>

#include "test_precomp.hpp"
#include "opencv2/gapi/gtransform.hpp"

namespace opencv_test
{

namespace
{
using GMat2 = std::tuple<GMat, GMat>;
using GMat3 = std::tuple<GMat, GMat, GMat>;
using GMat = cv::GMat;

GAPI_TRANSFORM(my_transform, <GMat(GMat, GMat)>, "does nothing")
{
    static GMat pattern(GMat, GMat)
    {
        return {};
    };

    static GMat substitute(GMat, GMat)
    {
        return {};
    }
};

GAPI_TRANSFORM(another_transform, <GMat3(GMat, GMat)>, "does nothing")
{
    static GMat3 pattern(GMat, GMat)
    {
        return {};
    };

    static GMat3 substitute(GMat, GMat)
    {
        return {};
    }
};

GAPI_TRANSFORM(copy_transform, <GMat3(GMat, GMat)>, "does nothing")
{
    static GMat3 pattern(GMat, GMat)
    {
        return {};
    };

    static GMat3 substitute(GMat, GMat)
    {
        return {};
    }
};

GAPI_TRANSFORM(simple_transform, <GMat(GMat)>, "does nothing")
{
    static GMat pattern(GMat)
    {
        return {};
    };

    static GMat substitute(GMat)
    {
        return {};
    }
};

GAPI_TRANSFORM(another_simple_transform, <GMat2(GMat)>, "does nothing too")
{
    static GMat2 pattern(GMat)
    {
        return {};
    };

    static GMat2 substitute(GMat)
    {
        return {};
    }
};
} // namespace

TEST(KernelPackageTransform, SimpleInclude)
{
    cv::gapi::GKernelPackage pkg = cv::gapi::kernels<simple_transform,
                                                     another_simple_transform>();
    auto tr = pkg.get_transformations();
    EXPECT_EQ(2u, tr.size());
}

TEST(KernelPackageTransform, SingleOutInclude)
{
    cv::gapi::GKernelPackage pkg;
    pkg.include<my_transform>();
    auto tr = pkg.get_transformations();
    EXPECT_EQ(1u, tr.size());
}

TEST(KernelPackageTransform, MultiOutInclude)
{
    cv::gapi::GKernelPackage pkg;
    pkg.include<my_transform>();
    pkg.include<another_transform>();
    auto tr = pkg.get_transformations();
    EXPECT_EQ(2u, tr.size());
}

TEST(KernelPackageTransform, MultiOutConstructor)
{
    cv::gapi::GKernelPackage pkg = cv::gapi::kernels<my_transform,
                                                     another_transform>();
    auto tr = pkg.get_transformations();
    EXPECT_EQ(2u, tr.size());
}

TEST(KernelPackageTransform, CopyClass)
{
    cv::gapi::GKernelPackage pkg = cv::gapi::kernels<copy_transform,
                                                     another_transform>();
    auto tr = pkg.get_transformations();
    EXPECT_EQ(2u, tr.size());
}

TEST(KernelPackageTransform, Combine)
{
    cv::gapi::GKernelPackage pkg1 = cv::gapi::kernels<my_transform>();
    cv::gapi::GKernelPackage pkg2 = cv::gapi::kernels<another_transform>();
    cv::gapi::GKernelPackage pkg_comb =
        cv::gapi::combine(pkg1, pkg2);
    auto tr = pkg_comb.get_transformations();
    EXPECT_EQ(2u, tr.size());
}

TEST(KernelPackageTransform, GArgsSize)
{
    auto tr = copy_transform::transformation();
    GMat a, b;
    auto subst = tr.substitute({cv::GArg(a), cv::GArg(b)});

    // return type of 'copy_transform' is GMat3
    EXPECT_EQ(3u, subst.size());
}

} // namespace opencv_test
