// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>
//         Luca Bertaccini <lbertaccini@iis.ee.ethz.ch>
//         Luca Colagrande <colluca@iis.ee.ethz.ch>
//         Viviane Potocnik <vivianep@iis.ee.ethz.ch>

#include <stdint.h>

#include "snrt.h"

#pragma once

typedef float v2f32 __attribute__((vector_size(8)));
typedef __fp16 v4f16 __attribute__((vector_size(8)));
typedef char v8f8 __attribute__((vector_size(8)));

// Floating-point multiplications by zero cannot be optimized as in some
// edge cases they do not yield zero:
// - 0f * NaN = NaN
// - 0f * INFINITY == NaN
// Thus in order to optimize it, we need to test for zero. You can use this
// function for free when `multiplier` is a constant.
static inline double multiply_opt(double multiplicand, double multiplier) {
    if (multiplier)
        return multiplicand * multiplier;
    else
        return 0;
}

#include "gemm_fp16.h"
#include "gemm_fp32.h"
#include "gemm_fp64.h"
#include "gemm_fp8.h"

// define the gemm_fp function pointer
typedef void (*gemm_fp_t)(uint32_t m, uint32_t n, uint32_t k, void* a,
                          uint32_t lda, uint32_t transa, void* b,
                          uint32_t transb, uint32_t ldb, void* c, uint32_t ldc,
                          uint32_t beta, uint32_t setup_ssr);

typedef struct {
    double alpha;
    uint32_t prec;
    uint32_t setup_ssr;
    uint32_t parallelize_m;
    uint32_t parallelize_k;
    uint32_t m_tiles;
    uint32_t n_tiles;
    uint32_t k_tiles;
    uint32_t load_a;
    uint32_t load_b;
    uint32_t load_c;
    uint32_t transa;
    uint32_t transb;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    void* a;
    void* b;
    uint32_t beta;
    void* c;
    void* gemm_fp;
} gemm_args_t;

// BLAS compliant single-cluster single-tile GEMM kernel, with some additional
// arguments at the beginning to specify Snitch implementation details. Matrix
// sizes and pointers are for the whole cluster computation. Within a cluster
// the computation is parallelized by assigning distinct output rows to
// distinct cores.
// TODO: beta (and alpha) should be of floating-point type (same precision as
// operands)
void sc_st_gemm(gemm_args_t* gemm_args, void* a, void* b, uint32_t beta,
                void* c) {
    gemm_fp_t impl = (gemm_fp_t)gemm_args->gemm_fp;
    precision_t prec = gemm_args->prec;
    uint32_t setup_ssr = gemm_args->setup_ssr;
    uint32_t transa = gemm_args->transa;
    uint32_t transb = gemm_args->transb;

    uint32_t m = gemm_args->M / gemm_args->m_tiles;
    uint32_t n = gemm_args->N / gemm_args->n_tiles;
    uint32_t k = gemm_args->K;

    uint32_t lda = k;
    uint32_t ldb;
    if (transb) {
        ldb = k;
    } else {
        ldb = n;
    }
    uint32_t ldc = n;

    double alpha = gemm_args->alpha;

    if (snrt_is_compute_core()) {
        const uint32_t compute_num = snrt_cluster_compute_core_num();
        const uint32_t compute_id = snrt_cluster_core_idx();

        // Compute cores work not on contiguous blocks but on strided rows
        uint32_t lda_strided = compute_num * lda;
        uint32_t ldc_strided = compute_num * ldc;

        // Compute cores access A and C at offsets of one row from each other
        uint32_t offsetA = compute_id * lda * prec;
        uint32_t offsetC = compute_id * ldc * prec;

        // Compute fraction of C rows every core computes
        uint32_t frac_m = m / compute_num;
        uint32_t rem_m = m % compute_num;
        if (snrt_cluster_core_idx() < rem_m) frac_m++;

        if (frac_m > 0)
            impl(frac_m, n, k, a + offsetA, lda_strided, transa, b, ldb, transb,
                 c + offsetC, ldc_strided, (float)beta, setup_ssr);
    }
}

// Similar to `dma_broadcast_to_clusters()` defined in `mcast.h`, but loads
// the data also to cluster 0.
static inline void dma_broadcast_load_to_clusters(void* dst, void* src, size_t size) {
    int is_first_cluster_in_quad = (snrt_cluster_idx() % N_CLUSTERS_PER_QUAD) == 0;
    // Only the DM core of the first cluster in every quadrant is active,
    // and loads the data from memory.
    if (is_first_cluster_in_quad) {
        // Load data from memory
        snrt_dma_start_1d(
            dst,
            src,
            size
        );
        snrt_dma_wait_all();
        // When the data is available at cluster 0 in each quadrant, it
        // is forwarded to all other clusters in the quadrant.
        for (int i = 1; i < N_CLUSTERS_PER_QUAD; i++) {
            int remote_cluster_idx = snrt_cluster_idx() + i;
            snrt_dma_start_1d(
                snrt_remote_l1_ptr(dst, snrt_cluster_idx(), remote_cluster_idx),
                dst,
                size
            );
        }
        snrt_dma_wait_all();
        // Wake up other clusters when transfers are done
        for (int i = 1; i < N_CLUSTERS_PER_QUAD; i++) {
            int remote_cluster_idx = snrt_cluster_idx() + i;
            *(cluster_clint_set_ptr(remote_cluster_idx)) = 1 << snrt_cluster_core_idx();
        }
    }
    // Put clusters who don't participate in the broadcast to sleep, as if
    // they proceed directly to the global barrier, they will interfere with
    // the other clusters, by sending their atomics on the narrow interconnect.
    else {
        snrt_wfi();
        snrt_int_clr_mcip();
    }
}

#ifndef MEASURE_FIRST_TILE
#define MEASURE_FIRST_TILE 0
#endif

// Multiple-cluster multiple-tile GEMM implementation.
// If parallelize_m, assigns a distinct subset of M-tiles to distinct clusters.
// If parallelize_k, then K-tiles are distributed to distinct clusters; a
// binary reduction tree is implemented to accumulate these tiles together.
// Note: in the current implementation, parallelize_m and parallelize_k
// should be mutually-exclusive. The load_* options allow to bypass the DMA
// transfers and operate directly on the a, b and c inputs.
// m_tiles: number of tiles in M dimension
// k_tiles: number of tiles in K dimension
// n_tiles: number of tiles in N dimension
int gemm(gemm_args_t* args) {
    // Clear interrupt received from CVA6 for wakeup
    snrt_int_clr_mcip();

    gemm_args_t* local_args = snrt_l1_next();

    // Copy the arguments to local memory
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_args, args, sizeof(gemm_args_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t m = local_args->M;
    uint32_t n = local_args->N;
    uint32_t k = local_args->K;
    precision_t prec = (precision_t)local_args->prec;
    uint32_t setup_ssr = local_args->setup_ssr;
    uint32_t m_tiles = local_args->m_tiles;
    uint32_t n_tiles = local_args->n_tiles;
    uint32_t transa = local_args->transa;
    uint32_t transb = local_args->transb;
    double alpha = local_args->alpha;
    void* a = local_args->a;
    void* b = local_args->b;
    uint32_t beta = local_args->beta;
    void* c = local_args->c;

    // Copy inputs from L3 to L2 (wide SPM)
    uint32_t size_a = m * k * prec;
    uint32_t size_b = k * n * prec;
    uint32_t size_c = m * n * prec;
    void *l2_a = (void*)SPM_WIDE_BASE_ADDR;
    void *l2_b = (void*)l2_a + size_a;
    void *l2_c = (void*)l2_b + size_b;
    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        snrt_dma_start_1d(l2_a, a, size_a);
        snrt_dma_start_1d(l2_b, b, size_b);
        snrt_dma_wait_all();
    }
    snrt_global_barrier();

    // Calculate tile sizes
    uint32_t frac_m = m / m_tiles;
    uint32_t frac_n = n / n_tiles;
    uint32_t frac_k = k;
    uint32_t frac_a = frac_m * frac_k;
    uint32_t frac_c = frac_m * frac_n;
    uint32_t size_frac_a = frac_a * prec;
    uint32_t size_frac_b = frac_k * frac_n * prec;
    uint32_t size_frac_c = frac_c * prec;

    // Allocate space in TCDM
    void *local_a[2];
    void *local_b[2];
    void *local_c[2];
    void* heap_ptr = (void*)local_args + sizeof(gemm_args_t);
    local_a[0] = heap_ptr;
    heap_ptr += size_frac_a;
    local_b[0] = heap_ptr;
    heap_ptr += size_frac_b;
    local_c[0] = heap_ptr;
    heap_ptr += size_frac_c;
    local_a[1] = heap_ptr;
    heap_ptr += size_frac_a;
    local_b[1] = heap_ptr;
    heap_ptr += size_frac_b;
    local_c[1] = heap_ptr;

    // Calculate number of iterations
    int iterations = n_tiles + 2;
    int buff_idx;
    int i, i_dma_out, i_dma_in, i_compute;

    // Synchronize all clusters at the beginning
    snrt_global_barrier();

    // Iterate over all tiles
    for (i = 0; i < iterations; i++) {
        if (snrt_is_dm_core()) {
            // DMA out
            // (out before in to avoid overwriting data)
            if (i > 1) {
                snrt_mcycle();

                // Compute tile and buffer indices
                i_dma_out = i - 2;
                buff_idx = i_dma_out % 2;

                // Copy job outputs from TCDM
                snrt_dma_store_2d_tile(l2_c, local_c[buff_idx], snrt_cluster_idx(),
                                       i_dma_out, frac_m, frac_n, n, prec);
                snrt_dma_wait_all();

                snrt_mcycle();
            }

            // DMA in
            if (i < n_tiles) {
                snrt_mcycle();

                // Compute tile and buffer indices
                i_dma_in = i;
                buff_idx = i_dma_in % 2;

                // Copy job operands in TCDM
#if defined(SUPPORTS_MULTICAST) && defined(USE_MULTICAST)
                if (snrt_cluster_idx() == 0) {
                    snrt_dma_mcast_load_1d_tile(local_b[buff_idx], l2_b, i_dma_in,
                                                frac_n * frac_k, prec,
                                                BCAST_MASK_ALL);
                }
#elif defined(SW_MULTICAST)
                dma_broadcast_load_to_clusters(local_b[buff_idx],
                                               l2_b + i_dma_in * frac_n * frac_k * prec,
                                               frac_n * frac_k * prec);
#else
                snrt_dma_load_1d_tile(local_b[buff_idx], l2_b, i_dma_in,
                                      frac_n * frac_k, prec);
#endif
                // Load A tile only on first iteration
                if (i_dma_in == 0 || MEASURE_FIRST_TILE) {
                    snrt_dma_load_1d_tile(local_a[0], l2_a, snrt_cluster_idx(),
                                          frac_m * frac_k, prec);
                }
                snrt_dma_wait_all();
                snrt_mcycle();
            }
        }

        // Compute
        if (snrt_is_compute_core()) {
            if (i > 0 && i < (n_tiles + 1)) {
                snrt_mcycle();

                // Compute tile and buffer indices
                i_compute = i - 1;
                buff_idx = i_compute % 2;

                // Perform tile computation
                volatile uint32_t ldb = frac_n;
                volatile uint32_t ldc = frac_n;
                if (transb) {
                    ldb = frac_k;
                }
                sc_st_gemm(local_args, local_a[0], local_b[buff_idx], beta,
                           local_c[buff_idx]);

                snrt_mcycle();
            }
        }

        // Synchronize cores after first iteration, exclusively for benchmarking
        if (i == 0)
            snrt_global_barrier();
        else
            snrt_cluster_hw_barrier();
    }

    // Synchronize all clusters at the end
    snrt_global_barrier();

    // Copy output from L2 to L3
    if (snrt_is_dm_core() && snrt_cluster_idx() == 0) {
        snrt_dma_start_1d(c, l2_c, size_c);
        snrt_dma_wait_all();
    }

    return 0;
}
