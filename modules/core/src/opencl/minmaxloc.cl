// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifdef DEPTH_0
#define MIN_VAL 0
#define MAX_VAL 255
#elif defined DEPTH_1
#define MIN_VAL -128
#define MAX_VAL 127
#elif defined DEPTH_2
#define MIN_VAL 0
#define MAX_VAL 65535
#elif defined DEPTH_3
#define MIN_VAL -32768
#define MAX_VAL 32767
#elif defined DEPTH_4
#define MIN_VAL INT_MIN
#define MAX_VAL INT_MAX
#elif defined DEPTH_5
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#elif defined DEPTH_6
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#endif

#define INDEX_MAX UINT_MAX

#ifdef NEED_MINLOC
#define CALC_MINLOC(inc) minloc = id + inc
#else
#define CALC_MINLOC(inc)
#endif

#ifdef NEED_MAXLOC
#define CALC_MAXLOC(inc) maxloc = id + inc
#else
#define CALC_MAXLOC(inc)
#endif

#ifdef NEED_MINVAL
#define CALC_MIN(p, inc) \
    if (minval > temp.p) \
    { \
        minval = temp.p; \
        CALC_MINLOC(inc); \
    }
#else
#define CALC_MIN(p, inc)
#endif

#ifdef NEED_MAXVAL
#define CALC_MAX(p, inc) \
    if (maxval < temp.p) \
    { \
        maxval = temp.p; \
        CALC_MAXLOC(inc); \
    }
#else
#define CALC_MAX(p, inc)
#endif

#define CALC_P(p, inc) \
    CALC_MIN(p, inc) \
    CALC_MAX(p, inc)

__kernel void minmaxloc(__global const uchar * srcptr, int src_step, int src_offset, int cols,
                        int total, int groupnum, __global uchar * dstptr
#ifdef HAVE_MASK
                        , __global const uchar * mask, int mask_step, int mask_offset
#endif
                        )
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int  id = get_global_id(0) * kercn;

    srcptr += src_offset;
#ifdef HAVE_MASK
    mask += mask_offset;
#endif

#ifdef NEED_MINVAL
    __local srcT1 localmem_min[WGS2_ALIGNED];
#ifdef NEED_MINLOC
    __local uint localmem_minloc[WGS2_ALIGNED];
#endif
#endif
#ifdef NEED_MAXVAL
    __local srcT1 localmem_max[WGS2_ALIGNED];
#ifdef NEED_MAXLOC
    __local uint localmem_maxloc[WGS2_ALIGNED];
#endif
#endif

    srcT1 minval = MAX_VAL, maxval = MIN_VAL;
    srcT temp;
    uint minloc = INDEX_MAX, maxloc = INDEX_MAX;
    int src_index;
#ifdef HAVE_MASK
    int mask_index;
#endif

    for (int grain = groupnum * WGS * kercn; id < total; id += grain)
    {
#ifdef HAVE_SRC_CONT
        src_index = mul24(id, (int)sizeof(srcT1));
#else
        src_index = mad24(id / cols, src_step, mul24(id % cols, (int)sizeof(srcT1)));
#endif

#ifdef HAVE_MASK
#ifdef HAVE_MASK_CONT
        mask_index = id;
#else
        mask_index = mad24(id / cols, mask_step, id % cols);
#endif
        if (mask[mask_index])
#endif
        {
            temp = *(__global const srcT *)(srcptr + src_index);
#if kercn == 1
#ifdef NEED_MINVAL
            if (minval > temp)
            {
                minval = temp;
#ifdef NEED_MINLOC
                minloc = id;
#endif
            }
#endif
#ifdef NEED_MAXVAL
            if (maxval < temp)
            {
                maxval = temp;
#ifdef NEED_MAXLOC
                maxloc = id;
#endif
            }
#endif
#elif kercn >= 2
            CALC_P(s0, 0)
            CALC_P(s1, 1)
#if kercn >= 4
            CALC_P(s2, 2)
            CALC_P(s3, 3)
#if kercn >= 8
            CALC_P(s4, 4)
            CALC_P(s5, 5)
            CALC_P(s6, 6)
            CALC_P(s7, 7)
#if kercn == 16
            CALC_P(s8, 8)
            CALC_P(s9, 9)
            CALC_P(sA, 10)
            CALC_P(sB, 11)
            CALC_P(sC, 12)
            CALC_P(sD, 13)
            CALC_P(sE, 14)
            CALC_P(sF, 15)
#endif
#endif
#endif
#endif
        }
    }

    if (lid < WGS2_ALIGNED)
    {
#ifdef NEED_MINVAL
        localmem_min[lid] = minval;
#endif
#ifdef NEED_MAXVAL
        localmem_max[lid] = maxval;
#endif
#ifdef NEED_MINLOC
        localmem_minloc[lid] = minloc;
#endif
#ifdef NEED_MAXLOC
        localmem_maxloc[lid] = maxloc;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
    {
        int lid3 = lid - WGS2_ALIGNED;
#ifdef NEED_MINVAL
        if (localmem_min[lid3] >= minval)
        {
#ifdef NEED_MINLOC
            if (localmem_min[lid3] == minval)
                localmem_minloc[lid3] = min(localmem_minloc[lid3], minloc);
            else
                localmem_minloc[lid3] = minloc,
#endif
                localmem_min[lid3] = minval;
        }
#endif
#ifdef NEED_MAXVAL
        if (localmem_max[lid3] <= maxval)
        {
#ifdef NEED_MAXLOC
            if (localmem_max[lid3] == maxval)
                localmem_maxloc[lid3] = min(localmem_maxloc[lid3], maxloc);
            else
                localmem_maxloc[lid3] = maxloc,
#endif
                localmem_max[lid3] = maxval;
        }
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;

#ifdef NEED_MINVAL
            if (localmem_min[lid] >= localmem_min[lid2])
            {
#ifdef NEED_MINLOC
                if (localmem_min[lid] == localmem_min[lid2])
                    localmem_minloc[lid] = min(localmem_minloc[lid2], localmem_minloc[lid]);
                else
                    localmem_minloc[lid] = localmem_minloc[lid2],
#endif
                    localmem_min[lid] = localmem_min[lid2];
            }
#endif
#ifdef NEED_MAXVAL
            if (localmem_max[lid] <= localmem_max[lid2])
            {
#ifdef NEED_MAXLOC
                if (localmem_max[lid] == localmem_max[lid2])
                    localmem_maxloc[lid] = min(localmem_maxloc[lid2], localmem_maxloc[lid]);
                else
                    localmem_maxloc[lid] = localmem_maxloc[lid2],
#endif
                    localmem_max[lid] = localmem_max[lid2];
            }
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        int pos = 0;
#ifdef NEED_MINVAL
        *(__global srcT1 *)(dstptr + mad24(gid, (int)sizeof(srcT1), pos)) = localmem_min[0];
        pos = mad24(groupnum, (int)sizeof(srcT1), pos);
#endif
#ifdef NEED_MAXVAL
        *(__global srcT1 *)(dstptr + mad24(gid, (int)sizeof(srcT1), pos)) = localmem_max[0];
        pos = mad24(groupnum, (int)sizeof(srcT1), pos);
#endif
#ifdef NEED_MINLOC
        *(__global uint *)(dstptr + mad24(gid, (int)sizeof(uint), pos)) = localmem_minloc[0];
        pos = mad24(groupnum, (int)sizeof(uint), pos);
#endif
#ifdef NEED_MAXLOC
        *(__global uint *)(dstptr + mad24(gid, (int)sizeof(uint), pos)) = localmem_maxloc[0];
#endif
    }
}
