// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/** @brief Linear Assignment Problem solver
 *
 *  lap is a linear assignment problem solver using Jonker-Volgenant algorithm for dense (LAPJV) or
 *  sparse (LAPMOD) matrices.
 *  Both algorithms are implemented from scratch based solely on the papers [1,2] and the public
 *  domain Pascal implementation provided by A. Volgenant [3].
 *  In my tests the LAPMOD implementation seems to be faster than the LAPJV implementation for
 *  matrices with a side of more than ~5000 and with less than 50% finite coefficients.
 *
 *  Tomas Kazmar, 2012-2017, BSD 2-clause license, see LICENSE.
 */


#include <stdlib.h>
#ifndef LAPJV_H
#define LAPJV_H

#define LARGE 1000000

#if !defined TRUE
#define TRUE 1
#endif
#if !defined FALSE
#define FALSE 0
#endif

#define NEW(x, t, n) if ((x = (t *)malloc(sizeof(t) * (n))) == 0) { return -1; }
#define FREE(x) if (x != 0) { free(x); x = 0; }
#define SWAP_INDICES(a, b) { int_t _temp_index = a; a = b; b = _temp_index; }

#if 0
#include <assert.h>
#define ASSERT(cond) assert(cond)
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define PRINT_COST_ARRAY(a, n) \
    while (1) { \
        printf(#a" = ["); \
        if ((n) > 0) { \
            printf("%f", (a)[0]); \
            for (uint_t j = 1; j < n; j++) { \
                printf(", %f", (a)[j]); \
            } \
        } \
        printf("]\n"); \
        break; \
    }
#define PRINT_INDEX_ARRAY(a, n) \
    while (1) { \
        printf(#a" = ["); \
        if ((n) > 0) { \
            printf("%d", (a)[0]); \
            for (uint_t j = 1; j < n; j++) { \
                printf(", %d", (a)[j]); \
            } \
        } \
        printf("]\n"); \
        break; \
    }
#else
#define ASSERT(cond)
#define PRINTF(fmt, ...)
#define PRINT_COST_ARRAY(a, n)
#define PRINT_INDEX_ARRAY(a, n)
#endif


typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;
typedef char boolean;
typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;

extern int_t lapjv_internal(
    const uint_t n, cost_t *cost[],
    int_t *x, int_t *y);

extern int_t lapmod_internal(
    const uint_t n, cost_t *cc, uint_t *ii, uint_t *kk,
    int_t *x, int_t *y, fp_t fp_version);

extern int_t _ccrrt_dense(uint_t, cost_t**, int_t*, int_t*, int_t*, cost_t*);
extern int_t _carr_dense(uint_t, cost_t**, uint_t, int_t*, int_t*, int_t*, cost_t*);
extern uint_t _find_dense(uint_t, uint_t, cost_t*, int_t*);
extern int_t _scan_dense(uint_t, cost_t**, uint_t*, uint_t*, cost_t*, int_t*, int_t*, int_t*, cost_t*);
extern int_t find_path_dense(uint_t, cost_t**, int_t, int_t*, cost_t*, int_t*);
extern int_t _ca_dense(uint_t, cost_t**, uint_t, int_t*, int_t*, int_t*, cost_t*);

#endif // LAPJV_H