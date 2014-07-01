//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Rock Li, Rock.li@amd.com
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
//

#if lcn == 1
    #if dcn == 4
        #define LUT_OP(num)\
            int idx = *(__global const int *)(srcptr + mad24(num, src_step, src_index));\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx & 0xff];\
            dst[1] = lut_l[(idx >> 8) & 0xff];\
            dst[2] = lut_l[(idx >> 16) & 0xff];\
            dst[3] = lut_l[(idx >> 24) & 0xff];
    #elif dcn == 3
        #define LUT_OP(num)\
            uchar3 idx = vload3(0, srcptr + mad24(num, src_step, src_index));\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx.x];\
            dst[1] = lut_l[idx.y];\
            dst[2] = lut_l[idx.z];
    #elif dcn == 2
        #define LUT_OP(num)\
            short idx = *(__global const short *)(srcptr + mad24(num, src_step, src_index));\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx & 0xff];\
            dst[1] = lut_l[(idx >> 8) & 0xff];
    #elif dcn == 1
        #define LUT_OP(num)\
            uchar idx = (srcptr + mad24(num, src_step, src_index))[0];\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx];
    #else
        #define LUT_OP(num)\
            __global const srcT * src = (__global const srcT *)(srcptr + mad24(num, src_step, src_index));\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            for (int cn = 0; cn < dcn; ++cn)\
                dst[cn] = lut_l[src[cn]];
    #endif
#else
    #if dcn == 4
        #define LUT_OP(num)\
            __global const uchar4 *src_pixel = (__global const uchar4 *)(srcptr + mad24(num, src_step, src_index));\
            int4 idx = convert_int4(src_pixel[0]) * lcn + (int4)(0, 1, 2, 3);\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx.x];\
            dst[1] = lut_l[idx.y];\
            dst[2] = lut_l[idx.z];\
            dst[3] = lut_l[idx.w];
    #elif dcn == 3
        #define LUT_OP(num)\
            uchar3 src_pixel = vload3(0, srcptr + mad24(num, src_step, src_index));\
            int3 idx = convert_int3(src_pixel) * lcn + (int3)(0, 1, 2);\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx.x];\
            dst[1] = lut_l[idx.y];\
            dst[2] = lut_l[idx.z];
    #elif dcn == 2
        #define LUT_OP(num)\
            __global const uchar2 *src_pixel = (__global const uchar2 *)(srcptr + mad24(num, src_step, src_index));\
            int2 idx = convert_int2(src_pixel[0]) * lcn + (int2)(0, 1);\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx.x];\
            dst[1] = lut_l[idx.y];
    #elif dcn == 1 //error case (1 < lcn) ==> lcn == scn == dcn
        #define LUT_OP(num)\
            uchar idx = (srcptr + mad24(num, src_step, src_index))[0];\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            dst[0] = lut_l[idx];
    #else
        #define LUT_OP(num)\
            __global const srcT *src = (__global const srcT *)(srcptr + mad24(num, src_step, src_index));\
            dst = (__global dstT *)(dstptr + mad24(num, dst_step, dst_index));\
            for (int cn = 0; cn < dcn; ++cn)\
                dst[cn] = lut_l[mad24(src[cn], lcn, cn)];
    #endif
#endif

#define LOCAL_LUT_INIT\
    {\
        __global const dstT * lut = (__global const dstT *)(lutptr + lut_offset);\
        int init = mad24((int)get_local_id(1), (int)get_local_size(0), (int)get_local_id(0));\
        int step = get_local_size(0) * get_local_size(1);\
        for (int i = init; i < 256 * lcn; i += step)\
        {\
            lut_l[i] = lut[i];\
        }\
        barrier(CLK_LOCAL_MEM_FENCE);\
    }

__kernel void LUT(__global const uchar * srcptr, int src_step, int src_offset,
                  __global const uchar * lutptr, int lut_step, int lut_offset,
                  __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols)
{
    __local dstT lut_l[256 * lcn];
    LOCAL_LUT_INIT;

    int x = get_global_id(0);
    int y = 4 * get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_index = mad24(y, src_step, mad24(x, (int)sizeof(srcT) * dcn, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(dstT) * dcn, dst_offset));
        __global dstT * dst;
        LUT_OP(0);
        if (y < rows - 1)
        {
            LUT_OP(1);
            if (y < rows - 2)
            {
                LUT_OP(2);
                if (y < rows - 3)
                {
                    LUT_OP(3);
                }
            }
        }

    }
}
