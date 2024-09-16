// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "snrt.h"

#define DELAY 100

int main() {
    // Clear interrupt received from CVA6 for wakeup
    snrt_int_clr_mcip();

    // Only DM cores proceed beyond this point
    if (snrt_is_compute_core()) return 0;

    // All DM cores enter WFI, except for cluster 0's which wakes the others
    // up by multicasting a cluster interrupt
    if (snrt_cluster_idx() == 0) {

        // Wait some time to ensure all other DM cores are in WFI
        for (int i = 0; i < DELAY; i++) {
            snrt_nop();
        }

        // Multicast cluster interrupt to every DM core
        // Note: we need to address another cluster's address space
        //       because the cluster XBAR has not been extended to support
        //       multicast yet.
        uint32_t *cluster1_cluster_clint_set_ptr = (uint32_t *) ((uint32_t)snrt_cluster_clint_set_ptr() + SNRT_CLUSTER_OFFSET);
        snrt_enable_multicast(BCAST_MASK);
        *cluster1_cluster_clint_set_ptr = 1 << snrt_cluster_core_idx();
        snrt_disable_multicast();
    }
    else {
        snrt_wfi();
        snrt_int_clr_mcip();
    }

    return 0;
}
