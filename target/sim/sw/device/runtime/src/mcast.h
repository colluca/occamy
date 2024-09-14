// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#ifndef N_CLUSTERS_TO_USE
#define N_CLUSTERS_TO_USE snrt_cluster_num()
#endif

#define BCAST_MASK_ACTIVE ((N_CLUSTERS_TO_USE - 1) << 18)
#define BCAST_MASK_ALL ((snrt_cluster_num() - 1) << 18)

inline void dma_broadcast_to_clusters(void* dst, void* src, size_t size) {
#if defined(SW_MULTICAST)
    int is_first_cluster_in_quad = (snrt_cluster_idx() % N_CLUSTERS_PER_QUAD) == 0;
    // Only the DM core of the first cluster in every quadrant is active.
    if (snrt_is_dm_core() && is_first_cluster_in_quad) {
        int is_cluster_zero = snrt_cluster_idx() == 0;
        // Cluster 0 broadcasts the data to cluster 0 in every other quadrant
        // and notifies them when the transfer is done.
        if (is_cluster_zero) {
            // Send data
            int nr_quadrants = N_CLUSTERS_TO_USE / N_CLUSTERS_PER_QUAD;
            for (int i = 1; i < nr_quadrants; i++) {
                int remote_cluster_idx = i * N_CLUSTERS_PER_QUAD;
                snrt_dma_start_1d(
                    snrt_remote_l1_ptr(dst, snrt_cluster_idx(), remote_cluster_idx),
                    src,
                    size
                );
            }
            snrt_dma_wait_all();
            // Send interrupts
            for (int i = 1; i < nr_quadrants; i++) {
                int remote_cluster_idx = i * N_CLUSTERS_PER_QUAD;
                *(cluster_clint_set_ptr(remote_cluster_idx)) = 1 << snrt_cluster_core_idx();
            }
        }
        // When the data is available at cluster 0 in each quadrant, it
        // is forwarded to all other clusters in the quadrant.
        if (snrt_cluster_idx() < N_CLUSTERS_TO_USE) {
            // Wait data
            if (!is_cluster_zero) {
                snrt_wfi();
                snrt_int_clr_mcip_unsafe();
            }
            // Forward data
            for (int i = 1; (i < N_CLUSTERS_TO_USE) && (i < N_CLUSTERS_PER_QUAD); i++) {
                int remote_cluster_idx = snrt_cluster_idx() + i;
                snrt_dma_start_1d(
                    snrt_remote_l1_ptr(dst, snrt_cluster_idx(), remote_cluster_idx),
                    is_cluster_zero ? src : dst,
                    size
                );
            }
            snrt_dma_wait_all();
        }
    }
#else
#if defined(SUPPORTS_MULTICAST) && defined(USE_MULTICAST)
    if (snrt_is_dm_core() && (snrt_cluster_idx() == 0)) {
        snrt_dma_start_1d_mcast(dst, src, size, BCAST_MASK_ACTIVE);
#else
    if (snrt_is_dm_core() && (snrt_cluster_idx() == 0)) {
        for (int i = 1; i < N_CLUSTERS_TO_USE; i++) {
            snrt_dma_start_1d(snrt_remote_l1_ptr(dst, snrt_cluster_idx(), i), src, size);
        }
#endif
        snrt_dma_wait_all();
    }
#endif
}
