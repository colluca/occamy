// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "snrt.h"

#define INITIALIZER 0xAAAAAAAA

#ifndef LENGTH
#define LENGTH 32
#endif

#define LENGTH_TO_CHECK 32

static inline int cluster_participates_in_bcast(int i) {
    int is_first_cluster_in_quad = (i % N_CLUSTERS_PER_QUAD) == 0;
    return is_first_cluster_in_quad && (i < N_CLUSTERS_TO_USE);
}

static inline void broadcast_wrapper(void* dst, void* src, size_t size) {
    snrt_global_barrier();
    dma_broadcast_to_clusters(dst, src, size);
    // Put clusters who don't participate in the broadcast to sleep, as if
    // they proceed directly to the global barrier, they will interfere with
    // the other clusters, by sending their atomics on the narrow interconnect.
    if (!cluster_participates_in_bcast(snrt_cluster_idx())) {
        snrt_wfi();
        snrt_int_clr_mcip();
    }
    // Wake these up when cluster 0 is done
    else if ((snrt_cluster_idx() == 0) && snrt_is_dm_core()) {
        for (int i = 0; i < snrt_cluster_num(); i++) {
            if (!cluster_participates_in_bcast(i))
                *(cluster_clint_set_ptr(i)) = 0x1FF;
        }
    }
}

int main() {
    snrt_int_clr_mcip();

    // Allocate destination buffer
    uint32_t *buffer_dst = snrt_l1_next_v2();
    uint32_t *buffer_src = buffer_dst + LENGTH;

    // First cluster initializes the source buffer and multicast-
    // copies it to the destination buffer in every cluster's TCDM.
    if (snrt_is_dm_core() && (snrt_cluster_idx() == 0)) {
        for (uint32_t i = 0; i < LENGTH; i++) {
            buffer_src[i] = INITIALIZER;
        }
    }

    // Initiate DMA transfer (twice to preheat the cache)
    for (volatile int i = 0; i < 2; i++) {
        broadcast_wrapper(buffer_dst, buffer_src, LENGTH * sizeof(uint32_t));
    }

    // All other clusters wait on a global barrier to signal the transfer
    // completion.
    snrt_global_barrier();

    // Every cluster except cluster 0 checks that the data in the destination
    // buffer is correct. To speed this up we only check the first 32 elements.
    if (snrt_is_dm_core() && (snrt_cluster_idx() < N_CLUSTERS_TO_USE) && (snrt_cluster_idx() != 0)) {
        uint32_t n_errs = LENGTH_TO_CHECK;
        for (uint32_t i = 0; i < LENGTH_TO_CHECK; i++) {
            if (buffer_dst[i] == INITIALIZER) n_errs--;
        }
        return n_errs;
    } else
        return 0;
}
