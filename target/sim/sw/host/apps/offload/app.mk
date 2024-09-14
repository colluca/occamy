# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

APP                 = offload
SRC_DIR             = $(SW_DIR)/host/apps/$(APP)/src
SRCS                = $(SRC_DIR)/main.c
$(APP)_DEVICE_APPS  = $(SW_DIR)/device/apps/blas/axpy
$(APP)_DEVICE_APPS += $(SW_DIR)/device/apps/blas/gemm
$(APP)_DEVICE_APPS += $(SW_DIR)/device/apps/blas/gemm_v2
$(APP)_DEVICE_APPS += $(SW_DIR)/device/apps/test_cluster_dma_mcast
$(APP)_DEVICE_APPS += $(SW_DIR)/device/apps/test_snitch_narrow_mcast
$(APP)_DEVICE_APPS += $(SW_DIR)/device/apps/bench_mcast

include $(SW_DIR)/host/apps/common.mk
