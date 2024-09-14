#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

from pathlib import Path
import sys
import yaml
import common

from snitch.util.sim import sim_utils
from snitch.util.sim.Simulator import QuestaSimulator

HOST_APP = 'offload'
DEVICE_APP = 'bench_mcast'

FILE_DIR = Path(__file__).parent.resolve()
BUILD_DIR = FILE_DIR / 'build'


def build_sw(tests, dry_run=False):
    for test in tests:
        defines = {
            'N_CLUSTERS_TO_USE': test['n_clusters_to_use'],
            'LENGTH': int(test['transfer_size'] * 1024 / 4)
        }
        if test['multicast'] == 'true':
            defines['USE_MULTICAST'] = 1
        elif test['multicast'] == 'hybrid':
            defines['SW_MULTICAST'] = 1
        common.build_heterogeneous_app(HOST_APP, DEVICE_APP, test['elf'].parent, test['hw_cfg'],
                                       defines=defines, dry_run=dry_run)


def build_hw(tests, dry_run=False):
    for test in tests:
        common.build_simulator_binary(test['hw_cfg'], dry_run)


def process_traces(tests, dry_run=False):
    for test in tests:
        roi_vars = {
            'n_clusters_to_use': test['n_clusters_to_use'],
            'multicast': test['multicast']
        }
        common.process_traces(
            test['run_dir'],
            device_elf=test['device_elf'],
            roi_spec=FILE_DIR / 'roi.json',
            roi_vars=roi_vars,
            dry_run=dry_run
        )


def get_tests(testlist, run_dir, hw_cfg):

    # Get tests from test list file
    testlist_path = Path(testlist).absolute()
    with open(testlist_path, 'r') as f:
        tests = yaml.safe_load(f)['runs']

    # Derive information required for simulation
    for test in tests:

        # Get test axes
        axes = common.get_axes(
            test['multicast'],
            test['n_clusters_to_use'],
            test['transfer_size']
        )

        # Derive test parameters
        test['hw_cfg'] = f'M-{hw_cfg}'
        unique_build_dir = common.unique_dir(BUILD_DIR, axes)
        test['elf'] = unique_build_dir / f'{HOST_APP}-{DEVICE_APP}.elf'
        test['device_elf'] = unique_build_dir / 'device' / f'{DEVICE_APP}.elf'
        unique_run_dir = common.unique_dir(Path(run_dir).resolve(), axes)
        test['run_dir'] = unique_run_dir

    return tests


def get_simulations(tests):
    simulations = []
    for test in tests:
        sim_bin = common.simulator_binary(test['hw_cfg'])
        simulations.append(QuestaSimulator(sim_bin).get_simulation(test))
    return simulations


def main():
    # Get args
    parser = sim_utils.parser()
    parser.add_argument(
        '--post-process-only',
        action='store_true',
        help='Does not run the simulations, only post-processes the traces')
    parser.add_argument(
        '--hw-cfg',
        default='Q8C4',
        help='Occamy configuration string e.g. Q6C4 for 6 quadrants 4 clusters')
    args = parser.parse_args()

    # Get tests from test list and create simulation objects
    tests = get_tests(args.testlist, args.run_dir, args.hw_cfg)
    simulations = get_simulations(tests)

    # Build HW and SW for every test and run simulations
    if not args.post_process_only:
        build_sw(tests, dry_run=args.dry_run)
        build_hw(tests, dry_run=args.dry_run)
        status = sim_utils.run_simulations(simulations,
                                           n_procs=args.n_procs,
                                           dry_run=args.dry_run,
                                           early_exit=args.early_exit)
        if status:
            return status

    # Post process simulation traces
    process_traces(tests, dry_run=args.dry_run)

    return 0


if __name__ == '__main__':
    sys.exit(main())
