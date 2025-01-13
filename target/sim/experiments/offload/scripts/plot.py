#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# flake8: noqa

import argparse
import matplotlib.pyplot as plt
from SimResults import SimResults, SimRegion, MissingRegionError
import numpy as np
import pandas as pd
import common
from copy import deepcopy
from math import ceil


# TODO: make sure there are no gaps in the regions from the traces.
#       Otherwise we might be missing some computation time, if we
#       add region times, or simply if we take the start (end) of
#       a region in place of the end (start) of the previous (next).

# Experiment parameters
ALL_NR_CLUSTER_CFGS = [1, 2, 4, 8, 16, 32]
ALL_MCAST_CFGS = [False, True]

# Plot parameters
MARKERS = ['o', '^', 's', '*', 'd', 'X']
MARKER_SIZES = [3, 3, 3, 3, 3, 3]
A4_HEIGHT = 11.7
IEEE_TEXT_WIDTH = 7.244
IEEE_TWO_COLUMN_SEP = 0.157
IEEE_COL_WIDTH = (IEEE_TEXT_WIDTH - IEEE_TWO_COLUMN_SEP) / 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("plot", default="fig8", nargs='?', type=str)
    parser.add_argument("--export", action='store_true')
    return parser.parse_args()


def get_app_label(app):
    APP_LABELS = {
        'axpy': 'AXPY',
        'montecarlo': 'Monte Carlo',
        'gemm': 'Matmul',
        'atax': 'ATAX',
        'covariance': 'Covariance',
        'correlation': 'Correlation',
        'bfs': 'BFS',
    }
    return APP_LABELS[app]


def get_mcast_label(mcast):
    MCAST_LABELS = {False: 'Baseline', True: 'Optimized'}
    return MCAST_LABELS[mcast]


class OffloadSimResults(SimResults):

    def __init__(self, app, mcast, nr_clusters, **kwargs):
        self.app = app
        self.mcast = mcast
        self.nr_clusters = nr_clusters
        kwargs_path = '/'.join([f'{key}{val}' for key, val in kwargs.items()])
        if app == 'covariance':
            super().__init__(f'runs/{app}/{kwargs_path}/{common.get_mcast_prefix(mcast)}/N{nr_clusters}')
        elif app == 'gemm':
            super().__init__(f'runs/{app}/{kwargs_path}/{common.get_mcast_prefix(mcast)}/N{nr_clusters}')
        else:
            super().__init__(f'runs/{app}/{kwargs_path}/{common.get_mcast_prefix(mcast)}/N{nr_clusters}')

    def get_send_job_information_time(self):
        return self.get_timespan(
            SimRegion('hart_0', 'prepare data', occurrence=1),
            SimRegion('hart_0', 'prepare data', occurrence=1)
        )

    def get_wakeup_time(self):
        send_interrupt_region = SimRegion('hart_0', 'send interrupt', occurrence=1)
        cluster_wakeup_regions = [
            SimRegion(f'hart_{1+9*i+8}', 'clr interrupt', occurrence=1)
            for i in range(self.nr_clusters)
        ]
        send_interrupt_time = self.get_metric(send_interrupt_region, 'tstart')
        cluster_wakeup_times = self.get_metrics(cluster_wakeup_regions, 'tend')
        vals = [t - send_interrupt_time for t in cluster_wakeup_times]
        return np.min(vals), np.max(vals), np.average(vals)

    def get_retrieve_job_pointer_time(self):
        vals = [
            self.get_metric(SimRegion(f'hart_{1+9*i+8}', 'get job ptr', occurrence=1), 'cycles')
            for i in range(self.nr_clusters)
        ]
        return np.min(vals), np.max(vals), np.average(vals)

    def get_retrieve_job_arguments_time(self):
        vals = [
            self.get_metric(SimRegion(f'hart_{1+9*i+8}', 'get job args', occurrence=1), 'cycles')
            for i in range(self.nr_clusters)
        ]
        return np.min(vals), np.max(vals), np.average(vals)

    def get_retrieve_job_operands_time(self):
        try:
            vals = [
                self.get_metric(SimRegion(f'hart_{1+9*i+8}', 'copy data in', occurrence=1), 'cycles')
                for i in range(self.nr_clusters)
            ]
            return np.min(vals), np.max(vals), np.average(vals)
        except MissingRegionError:
            return 0, 0, 0

    def get_job_execution_time(self):
        vals = []
        for i in range(self.nr_clusters):
            # Cluster execution time is calculated from the start of the first core
            # to the end of the last
            if self.app in ['axpy', 'gemm']:
                start_regions = [
                    SimRegion(f'hart_{1+9*i+j}', 'compute', occurrence=1)
                    for j in range(8)
                ]
                end_regions = start_regions
            elif self.app == 'montecarlo':
                start_regions = [
                    SimRegion(f'hart_{1+9*i+j}', 'compute psum', occurrence=1)
                    for j in range(8)
                ]
                end_regions = start_regions
            elif self.app == 'atax':
                start_regions = [
                    SimRegion(f'hart_{1+9*i+j}', 'Ax', occurrence=1)
                    for j in range(8)
                ]
                end_regions = [
                    SimRegion(f'hart_{1+9*i+j}', 'AtAx', occurrence=1)
                    for j in range(8)
                ]
            elif self.app == 'covariance':
                start_regions = [
                    SimRegion(f'hart_{1+9*i+j}', 'compute mean', occurrence=1)
                    for j in range(8)
                ]
                end_regions = [
                    SimRegion(f'hart_{1+9*i+j}', 'normalize', occurrence=1)
                    for j in range(8)
                ]
            start_times = self.get_metrics(start_regions, 'tstart')
            end_times = self.get_metrics(end_regions, 'tend')
            vals.append(np.max(end_times) - np.min(start_times))
        return np.min(vals), np.max(vals), np.average(vals)

    def get_reduction_time(self):
        assert self.app == 'montecarlo', 'Reduction time only for Monte Carlo sims'

        return self.get_timespan(
            SimRegion('hart_1', 'intra-cluster reduction', occurrence=1),
            SimRegion('hart_1', 'inter-cluster reduction', occurrence=1)
        )

    def get_writeback_job_outputs_time(self):
        try:
            vals = [
                self.get_metric(SimRegion(f'hart_{1+9*i+8}', 'copy data out', occurrence=1), 'cycles')
                for i in range(self.nr_clusters)
            ]
            return np.min(vals), np.max(vals), np.average(vals)
        except MissingRegionError:
            return 0, 0, 0

    def get_notify_job_completion_time(self):
        # From last core arriving on the barrier to CVA6 waking up
        return_regions = [
            SimRegion(f'hart_{1+9*i+8}', 'return', occurrence=1)
            for i in range(self.nr_clusters)
        ]
        wakeup_time = self.get_metric(SimRegion('hart_0', 'clr interrupt', occurrence=1), 'tstart')
        send_interrupt_time = np.max(self.get_metrics(return_regions, 'tstart'))
        return wakeup_time - send_interrupt_time

    def get_resume_operation_time(self):
        return self.get_timespan(
            SimRegion('hart_0', 'clr interrupt', occurrence=1),
            SimRegion('hart_0', 'clr interrupt', occurrence=1)
        )

    def get_total_time(self):
        return self.get_timespan(
            SimRegion('hart_0', 'prepare data', occurrence=1),
            SimRegion('hart_0', 'clr interrupt', occurrence=1))

    def get_ideal_time(self):
        """Get the total job time w/o offload overheads.
        
        This does not consider the retrieve job operands and writeback
        phases as offload overheads, i.e. they are included in the ideal
        time.
        """
        assert self.mcast == True, 'Ideal time only for multicast sims'
        # TODO: add assertions to verify that all start times are the same
        #       across clusters/cores (in the case of multicast sims).
        if self.app == 'montecarlo':
            start_times = [
                self.get_metric(SimRegion(f'hart_{1+9*i+j}', 'compute psum', occurrence=1), 'tstart')
                for i in range(self.nr_clusters) for j in range(8)
            ]
            end_times = [self.get_metric(
                SimRegion(f'hart_1', 'inter-cluster reduction', occurrence=1), 'tend')]
        else:
            start_times = [
                self.get_metric(SimRegion(f'hart_{1+9*i+8}', 'copy data in', occurrence=1), 'tstart')
                for i in range(self.nr_clusters)
            ]
            if self.app == 'covariance':
                end_times = [self.get_metric(SimRegion('hart_9', 'copy data out', occurrence=1), 'tend')]
            else:
                end_times = [
                    self.get_metric(SimRegion(f'hart_{1+9*i+8}', 'copy data out', occurrence=1), 'tend')
                    for i in range(self.nr_clusters)
                ]
        return np.max(end_times) - np.min(start_times)


def fig8(data, export):
    apps, sizes = zip(*data.items())

    # Create subplots
    fig, ax = plt.subplots(1, len(apps), layout="constrained")
    fig.set_figwidth(3.8)

    # Make sure ax is a list even when there is only one subplot
    if not hasattr(ax, '__len__'):
        ax = [ax]

    # Fill different subplots with different apps
    mcast_overheads = []
    for i, app in enumerate(apps):

        # Get data
        x_data = ALL_NR_CLUSTER_CFGS
        base_sims = [OffloadSimResults(app, False, x, **sizes[i]) for x in x_data]
        ideal_sims = [OffloadSimResults(app, True, x, **sizes[i]) for x in x_data]
        t_ideal = [sim.get_ideal_time() for sim in ideal_sims]
        t_all = [sim.get_total_time() for sim in base_sims]
        t_mcast = [sim.get_total_time() for sim in ideal_sims]
        mcast_overheads += [t_mcast[j] - t_ideal[j] for j in range(len(t_ideal))]

        # Plot different curves for ideal runtime and actual runtime 
        ax[i].plot(
            x_data,
            t_all,
            marker=MARKERS[0],
            markersize=MARKER_SIZES[0],
            linestyle='-',
            linewidth=1,
            label='base'
        )
        ax[i].plot(
            x_data,
            t_mcast,
            marker=MARKERS[1],
            markersize=MARKER_SIZES[1],
            linestyle='-.',
            linewidth=1,
            label='mcast'
        )
        ax[i].plot(
            x_data,
            t_ideal,
            marker=MARKERS[2],
            markersize=MARKER_SIZES[2],
            linestyle='--',
            linewidth=1,
            label='ideal'
        )

        # Set subplot parameters
        ax[i].set_xticks(x_data)
        ax[i].set_xlim([0, 33])
        ax[i].set_title(get_app_label(app))
        ax[i].grid(color='gainsboro', which='both', linewidth=0.5)

    ax[1].legend(loc='upper left')

    # Set figure parameters
    fig.supxlabel('Nr. clusters')
    fig.supylabel('Runtime [ns]')

    if not export:
        plt.show()
    else:
        plt.gcf().set_size_inches(IEEE_COL_WIDTH, 0.15 * A4_HEIGHT)
        plt.savefig('results/runtime.pdf', bbox_inches='tight')

    # Return relevant plot figures
    mcast_overheads = np.array(mcast_overheads)
    figs = {
        'McastOverheadMean': '{:.0f}'.format(np.mean(mcast_overheads)),
        'McastOverheadStddev': '{:.0f}'.format(np.std(mcast_overheads)),
    }
    return figs


def fig8v2(sizes, export=False):
    apps = sizes.keys()

    # Get data
    data = {}
    # perc = []
    for app in apps:
        data[app] = {}
        for nr_clusters in ALL_NR_CLUSTER_CFGS:
            base_sim = OffloadSimResults(app, False, nr_clusters, **sizes[app])
            ideal_sim = OffloadSimResults(app, True, nr_clusters, **sizes[app])
            base_time = base_sim.get_total_time()
            ideal_time = ideal_sim.get_ideal_time()
            overhead = base_time - ideal_time
            data[app][nr_clusters] = overhead
            # perc.append(100 * overhead / base_time)
    df = pd.DataFrame(data)
    df.rename(columns=get_app_label, inplace=True)

    # Plot bars
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.7, linewidth=0.5, edgecolor='black')

    # Mark average bar height per group
    bar_positions = []
    average_heights = []
    for i, index in enumerate(df.index):
        group_heights = df.loc[index].values
        bar_positions.append(i)
        average_heights.append(np.mean(group_heights))
    ax.plot(
        bar_positions,
        average_heights,
        color='black',
        marker='*',
        markerfacecolor='white',
        markeredgewidth=0.5,
        linestyle='--',
        linewidth=0.5,
        label='Average'
    )

    # Customize plot
    ax.set_xlabel('Nr. clusters')
    ax.set_ylabel('Offloading overhead [ns]')
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(color='gainsboro', which='both', axis='y', linewidth=0.5)
    ax.tick_params(axis='x', labelrotation=0)

    # Extend range of y axis to make bar labels visible
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] * 1.05)

    if not export:
        plt.show()
    else:
        plt.gcf().set_size_inches(IEEE_COL_WIDTH, 0.15 * A4_HEIGHT)
        plt.savefig('results/offload_overhead.pdf', bbox_inches='tight')

    # Get relevant plot figures
    single_cluster_overheads = [data[app][1] for app in apps]
    thirty_two_cluster_overheads = [data[app][32] for app in apps]
    matmul_overheads = [data['gemm'][nr_clusters] for nr_clusters in ALL_NR_CLUSTER_CFGS]
    figs = {
        'OverheadSingleClusterMean': '{:.0f}'.format(np.mean(single_cluster_overheads)),
        'OverheadSingleClusterStddev': '{:.0f}'.format(np.std(single_cluster_overheads)),
        'OverheadMatmulMax': '{:.0f}'.format(np.max(matmul_overheads)),
        'OverheadThirtyTwoClusterStddev': '{:.0f}'.format(np.std(thirty_two_cluster_overheads)),
    }
    return figs


def fig9v2(sizes, export):
    apps = sizes.keys()

    # Get data
    data = {}
    ideal = {}
    perc = []
    perc_dict = {}
    offload_configs = [1, 2, 4, 8, 16, 32]
    for nr_clusters in offload_configs:
        data[nr_clusters] = {}
        ideal[nr_clusters] = {}
        perc_dict[nr_clusters] = {}
        for app in apps:
            base_sim = OffloadSimResults(app, False, nr_clusters, **sizes[app])
            mcast_sim = OffloadSimResults(app, True, nr_clusters, **sizes[app])
            speedup = base_sim.get_total_time() / mcast_sim.get_total_time()
            ideal_speedup = base_sim.get_total_time() / mcast_sim.get_ideal_time()
            data[nr_clusters][app] = speedup
            ideal[nr_clusters][app] = ideal_speedup
            perc.append(100 * speedup / ideal_speedup)
            perc_dict[nr_clusters][app] = 100 * speedup / ideal_speedup
    df = pd.DataFrame(data)
    df.rename(columns=lambda x: f'{x} clusters', inplace=True)
    df.rename(index=get_app_label, inplace=True)

    # Plot fill bars
    ax = df.plot(kind='bar', linewidth=0.5, edgecolor='black', zorder=3, width=0.8)

    # Plot edge bars
    df = pd.DataFrame(ideal)
    df.rename(index=get_app_label, inplace=True)
    df.plot(ax=ax, kind='bar', width=0.8, linewidth=0.5, edgecolor='black', facecolor='white', zorder=2)

    # Add iso-speedup line
    ax.axhline(y=1, color='black', linestyle='-', linewidth=0.5, zorder=1)

    # Configure plot
    ax.set_ylabel('Speedup')
    ax.set_axisbelow(True)
    ax.grid(color='gainsboro', which='both', axis='y', linewidth=0.5, zorder=0)
    ax.tick_params(axis='x', labelrotation=0)

    # Show only legend handles for fill bars
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[0:len(offload_configs)], l[0:len(offload_configs)], loc='upper right')

    # # Add custom labels on top of each bar
    # fontsize = 4 if export else plt.rcParams['axes.labelsize']
    # yoffset = 5 if export else 10
    # # Iterates apps first and then nr_clusters
    # for i, p in enumerate(ax.patches[len(apps)*len(offload_configs):]):
    #     ax.annotate(f'{perc[i]:.0f}%',
    #                 (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                 ha='center', va='center', xytext=(0, yoffset),
    #                 textcoords='offset points', fontsize=fontsize)
    # # Extend range of y axis to make bar labels visible
    # ylim = ax.get_ylim()
    # ax.set_ylim(ylim[0], ylim[1] * 1.05)

    if not export:
        plt.show()
    else:
        plt.gcf().set_size_inches(IEEE_COL_WIDTH, 0.15 * A4_HEIGHT)
        plt.savefig('results/speedup_vs_nr_clusters.pdf', bbox_inches='tight')

    # Get relevant plot figures
    gemm_ideal_speedups = [ideal[nr_clusters]['gemm'] for nr_clusters in ALL_NR_CLUSTER_CFGS]
    bfs_ideal_speedups = [ideal[nr_clusters]['bfs'] for nr_clusters in ALL_NR_CLUSTER_CFGS]
    axpy_mc_matmul_perc = [perc_dict[nr_clusters][app] for nr_clusters in ALL_NR_CLUSTER_CFGS for app in ['axpy', 'montecarlo', 'gemm']]
    atax_cov_bfs_perc = [perc_dict[nr_clusters][app] for nr_clusters in ALL_NR_CLUSTER_CFGS for app in ['atax', 'covariance', 'bfs']]
    figs = {
        'IdealSpeedupMatmulMax': '{:.2f}'.format(np.max(gemm_ideal_speedups)),
        'IdealSpeedupBFSMax': '{:.2f}'.format(np.max(bfs_ideal_speedups)),
        'IdealSpeedupFractionAXPYMonteCarloMatmulMin': '{:.0f}'.format(np.min(axpy_mc_matmul_perc)),
        'IdealSpeedupFractionAXPYMonteCarloMatmulMax': '{:.0f}'.format(np.max(axpy_mc_matmul_perc)),
        'IdealSpeedupFractionATAXCovarianceBFSMin': '{:.0f}'.format(np.min(atax_cov_bfs_perc)),
        'IdealSpeedupFractionATAXCovarianceBFSMax': '{:.0f}'.format(np.max(atax_cov_bfs_perc)),
    }
    return figs


def fig10(data, export):
    apps, appdata = zip(*data.items())

    # Create subplots
    fig = plt.figure(layout="constrained")
    ax = fig.subplot_mosaic(
        [
            apps[0:2],
            # apps[2:4],
        ],
    )

    # Make sure ax is a list even when there is only one subplot
    if not hasattr(ax, '__len__'):
        ax = [ax]

    # Fill different subplots with different apps
    max_speedup = 0
    for j, app in enumerate(apps):
    
        all_x = []
        key_to_scale = appdata[j]['key_to_scale']
        sizes = appdata[j]['sizes']

        # Create different curves for different nr clusters
        for i, nr_clusters in enumerate(ALL_NR_CLUSTER_CFGS[2:]):

            # Get full problem size, from the size per cluster
            x_data = deepcopy(sizes)
            for size in x_data:
                size[key_to_scale] *= nr_clusters

            # Get data
            sims = [
                [OffloadSimResults(app, mcast, nr_clusters, **x) for x in x_data]
                for mcast in ALL_MCAST_CFGS
            ]
            t_baseline = [sim.get_total_time() for sim in sims[0]]
            t_mcast = [sim.get_total_time() for sim in sims[1]]
            speedup = [a / b for a, b in zip(t_baseline, t_mcast)]
            max_speedup = max(max_speedup, np.max(speedup))

            # Plot speedup
            ax[app].plot(
                [x[key_to_scale] for x in x_data],
                speedup,
                marker=MARKERS[i],
                markersize=MARKER_SIZES[i],
                linestyle='-',
                linewidth=1,
                label=f'{nr_clusters} clusters'
            )

            # Get all x values as combination of x values found in all curves
            all_x += [x[key_to_scale] for x in x_data]

        # Set subplot parameters
        all_x = list(set(all_x))
        all_x.sort()
        # Hide second X axis label, as it overlaps with neighbours
        labels = [str(x) if i != 1 else "" for i, x in enumerate(all_x)]
        ax[app].set_xticks(all_x, labels, rotation=-45)
        ax[app].set_ylim([1, ax[app].get_ylim()[1]])
        ax[app].set_title(get_app_label(app))
        ax[app].grid(color='gainsboro', linewidth=0.5)

    # Create shared legend
    # h, l = ax[apps[0]].get_legend_handles_labels()
    # fig.legend(h, l, ncol=1, loc="upper center")
    ax[apps[1]].legend()

    # Set figure parameters
    fig.supxlabel('Problem size')
    fig.supylabel('Speedup')

    if not export:
        plt.show()
    else:
        plt.gcf().set_size_inches(IEEE_COL_WIDTH, 0.15 * A4_HEIGHT)
        plt.savefig('results/speedup_vs_problem_size.pdf', bbox_inches='tight')

    # Return relevant plot figures
    figs = {
        'MaxSpeedup': '{:.1f}'.format(max_speedup),
    }
    return figs


def simple_phase_plot(ax, data, title):
    x = ALL_NR_CLUSTER_CFGS
    ax.plot(x, data[0], marker=MARKERS[0], markersize=MARKER_SIZES[0], linestyle='-', label='baseline')
    ax.plot(x, data[1], marker=MARKERS[1], markersize=MARKER_SIZES[1], linestyle='-', label='w/ extensions')
    # # Annotate points
    # for point in zip(x, data[0]):
    #     plt.annotate(f'{point[1]:.2f}', point, xytext=(0, 3), textcoords='offset points',
    #         ha='center', va='bottom')
    # for point in zip(x, data[1]):
    #     plt.annotate(f'{point[1]:.2f}', point, xytext=(0, 3), textcoords='offset points',
    #         ha='center', va='bottom')
    # Adjust Y-axis range to be at least 40
    y_min, y_max = ax.get_ylim()
    min_range = 40
    if (y_max - y_min) < min_range:
        mid_point = (y_min + y_max) / 2
        new_y_min = max(0, mid_point - (min_range / 2))
        new_y_max = new_y_min + min_range
        ax.set_ylim(new_y_min, new_y_max)
    # Configure plot
    ax.set_xticks(x)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(color='gainsboro', which='both', linewidth=0.5)
    # ax.legend()


def statistic_phase_plot(ax, data, title):
    data = np.array(data)
    data = data.swapaxes(1, 2)
    x = ALL_NR_CLUSTER_CFGS
    ax.set_axisbelow(True)
    ax.grid(color='gainsboro', which='both', linewidth=0.5)
    ax.plot(x, data[0][2], marker=MARKERS[0], markersize=MARKER_SIZES[0], linestyle='-', label='baseline')
    ax.plot(x, data[1][2], marker=MARKERS[1], markersize=MARKER_SIZES[1], linestyle='-', label='w/ extensions')
    ax.fill_between(x, data[0][0], data[0][1], alpha=0.3)
    ax.fill_between(x, data[1][0], data[1][1], alpha=0.3)
    # # Annotate points
    # for point in zip(x, data[0][0]):
    #     ax.annotate(f'{point[1]:.2f}', point, xytext=(0, 3), textcoords='offset points',
    #         ha='center', va='bottom')
    # for point in zip(x, data[1][0]):
    #     ax.annotate(f'{point[1]:.2f}', point, xytext=(0, 3), textcoords='offset points',
    #         ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_title(title)
    # ax.legend()


def recursive_map(func, data):
    """Apply a function to a nested list.
    
    Similar to the built-in map() function, but preserves the
    structure of the nested list.
    """
    # Check if the data is a list
    if isinstance(data, list):
        return [recursive_map(func, item) for item in data]
    else:
        return func(data)


def fig11(data, export):
    app, kwargs = next(iter(data.items()))

    # Get simulation results
    sims = [
        [OffloadSimResults(app, mcast, nr_clusters, **kwargs) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        for mcast in ALL_MCAST_CFGS
    ]

    # Get data
    t_send_job_information   = recursive_map(OffloadSimResults.get_send_job_information_time, sims)
    t_wakeup                 = recursive_map(OffloadSimResults.get_wakeup_time, sims)
    t_retrieve_job_pointer   = recursive_map(OffloadSimResults.get_retrieve_job_pointer_time, sims)
    t_retrieve_job_arguments = recursive_map(OffloadSimResults.get_retrieve_job_arguments_time, sims)
    t_retrieve_job_operands  = recursive_map(OffloadSimResults.get_retrieve_job_operands_time, sims)
    t_job_execution          = recursive_map(OffloadSimResults.get_job_execution_time, sims)
    t_writeback_job_outputs  = recursive_map(OffloadSimResults.get_writeback_job_outputs_time, sims)
    t_notify_job_completion  = recursive_map(OffloadSimResults.get_notify_job_completion_time, sims)
    t_resume_operation       = recursive_map(OffloadSimResults.get_resume_operation_time, sims)
    t_total                  = recursive_map(OffloadSimResults.get_total_time, sims)

    # print(app, kwargs)
    # print("A", t_send_job_information[-1])
    # print("B", t_wakeup[-1])
    # print("C", t_retrieve_job_pointer[-1])
    # print("D", t_retrieve_job_arguments[-1])
    # print("E", t_retrieve_job_operands[-1])
    # print("F", t_job_execution[-1])
    # print("G", t_writeback_job_outputs[-1])
    # print("H", t_notify_job_completion[-1])
    # print("I", t_resume_operation[-1])
    # print("total", t_total[-1])

    fig, ax = plt.subplots(3, 3, layout="constrained")
    # fig, ax = plt.subplots(4, 3, layout="constrained")
    fig.set_figwidth(8.3)
    simple_phase_plot(ax[0][0], t_send_job_information, "A) Send job information")
    statistic_phase_plot(ax[0][1], t_wakeup, "B) Cluster wakeup")
    statistic_phase_plot(ax[0][2], t_retrieve_job_pointer, "C) Retrieve job pointer")
    statistic_phase_plot(ax[1][0], t_retrieve_job_arguments, "D) Retrieve job arguments")
    statistic_phase_plot(ax[1][1], t_retrieve_job_operands, "E) Retrieve job operands")
    statistic_phase_plot(ax[1][2], t_job_execution, "F) Job execution")
    statistic_phase_plot(ax[2][0], t_writeback_job_outputs, "G) Writeback job outputs")
    simple_phase_plot(ax[2][1], t_notify_job_completion, "H) Notify job completion")
    simple_phase_plot(ax[2][2], t_resume_operation, "I) Resume operation on host")
    # simple_phase_plot(ax[3][0], t_total, "Total")

    # Loop through the first two rows and remove the X axis ticks and labels
    for row in range(2):
        for col in range(3):
            ax[row][col].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.supxlabel('Nr. clusters')
    fig.supylabel('Runtime [ns]')
    h, l = ax[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower right', ncols=2)

    if not export:
        plt.show()
    else:
        plt.gcf().set_size_inches(IEEE_TEXT_WIDTH, 0.3 * A4_HEIGHT)
        plt.savefig('results/breakdown.pdf', bbox_inches='tight')


def latex_metrics(metrics):
    # Auxiliary function to format a metric as a LaTeX command
    def latex_metric(name, value):
        return f"\\newcommand{{\\Result{name}}}{{{value}\\xspace}}\n"

    # Create file
    with open('results/metrics.tex', 'w') as f:
        [f.write(latex_metric(name, value)) for name, value in metrics.items()]


def model_phase(phase, app, nr_clusters, **kwargs):
    if phase == 'send_job_information' or phase == 'A':
        if app in ['axpy', 'atax']:
            return 38
        elif app == 'gemm':
            # Only one more store instruction than AXPY
            return 39
        elif app == 'mc':
            return 18
    elif phase == 'wakeup' or phase == 'B':
        if app == 'axpy':
            return 47
        elif app == 'gemm':
            return 52
        elif app == 'atax':
            return 48
        elif app == 'mc':
            return 48
    elif phase == 'retrieve_job_pointer' or phase == 'C':
        if app == 'axpy':
            return 11
        elif app in ['gemm', 'atax']:
            return 9
        elif app == 'mc':
            return 48
    elif phase == 'retrieve_job_arguments' or phase == 'D':
        return 0
    elif phase == 'retrieve_job_operands' or phase == 'E':
        if app == 'axpy':
            return 108 + ceil(kwargs['L']/4)
        elif app == 'gemm':
            K = 4
            return 171 + kwargs['M']*K/8 + ceil(K*kwargs['N']/8)*nr_clusters
        elif app == 'atax':
            return 169 + (ceil(kwargs['N']/8) + ceil(kwargs['N']*kwargs['M']/8))*nr_clusters
        elif app == 'mc':
            return 11
    elif phase == 'job_execution' or phase == 'F':
        if app == 'axpy':
            return 55 + (1.46875*kwargs['L'])/(nr_clusters*8)
        elif app == 'gemm':
            # setup + calculate + terminate + hw barrier
            return 83 + (19*kwargs['M']*kwargs['N'])/(nr_clusters*8) + 17 + 3
        elif app == 'atax':
            # setup + calculate Ax + setup + calculate AtAx + terminate
            n_cores = nr_clusters * 8
            return 102 + (1020 / 256) * kwargs['N'] * kwargs['M'] + 9 + (612 / 32) * (kwargs['N'] / n_cores) + 30
        elif app == 'mc':
            return 61 + (34*kwargs['L'])/(nr_clusters*8)
    elif phase == 'writeback_job_outputs' or phase == 'G':
        if app == 'axpy':
            return 21 + 55 + kwargs['L']/(nr_clusters*8)
        elif app == 'gemm':
            return 70 + (kwargs['M']*kwargs['N']/nr_clusters)/8
        elif app == 'atax':
            return 13 + 55 + (kwargs['N']/nr_clusters)/8
    elif phase == 'notify_job_completion' or phase == 'H':
        if app == 'axpy':
            return 59
        elif app == 'gemm':
            return 89
        elif app == 'atax':
            return 87
        elif app == 'mc':
            return 40
    elif phase == 'resume_operation' or phase == 'I':
        return 6


def model(app, nr_clusters, **kwargs):
    runtime = 0
    if app in ['axpy', 'gemm', 'atax']:
        runtime = model_phase('send_job_information', app, nr_clusters, **kwargs) + model_phase('wakeup', app, nr_clusters, **kwargs) + \
            model_phase('retrieve_job_pointer', app, nr_clusters, **kwargs) + model_phase('retrieve_job_arguments', app, nr_clusters, **kwargs) + \
            model_phase('retrieve_job_operands', app, nr_clusters, **kwargs) + model_phase('job_execution', app, nr_clusters, **kwargs) + \
            model_phase('writeback_job_outputs', app, nr_clusters, **kwargs) + model_phase('notify_job_completion', app, nr_clusters, **kwargs) + \
            model_phase('resume_operation', app, nr_clusters, **kwargs)
    return runtime


def fig13(data, export):
    apps, sizes = zip(*data.items())

    fig, ax = plt.subplots(1, len(apps), layout='constrained')
    fig.set_figwidth(3.8)

    if not hasattr(ax, '__len__'):
        ax = [ax]

    max_err = 0
    for i, app in enumerate(apps):

        app_sizes = sizes[i]

        true = np.array([[OffloadSimResults(app, True, n, **size).get_total_time() for size in app_sizes] for n in ALL_NR_CLUSTER_CFGS])
        pred = np.array([[model(app, n, **size) for size in app_sizes] for n in ALL_NR_CLUSTER_CFGS])
        err = 100*np.abs((true - pred) / true)
        max_err = max(max_err, np.max(err))

        im = ax[i].imshow(err, aspect=1/2)

        # Show all ticks and label them with the respective list entries
        if i == 0:
            ax[i].set_yticks(np.arange(len(ALL_NR_CLUSTER_CFGS)), labels=ALL_NR_CLUSTER_CFGS)
        else:
            ax[i].tick_params(
                axis='y',        # changes apply to the x-axis
                which='both',    # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                labelleft=False) # labels along the bottom edge are off

        # Label to plot on x axis depends on kernel
        if app == 'axpy':
            key = 'L'
        elif app == 'gemm':
            key = 'N'
        elif app == 'atax':
            key = 'M'
        ax[i].set_xticks(np.arange(len(app_sizes)), labels=[app_size[key] for app_size in app_sizes])

        # Loop over data dimensions and create text annotations.
        for k in range(len(app_sizes)):
            for j in range(len(ALL_NR_CLUSTER_CFGS)):
                text = ax[i].text(k, j, f"{err[j, k]:0.1f}%",
                                  ha="center", va="center", color="w")

        ax[i].set_title(get_app_label(app))

    fig.supxlabel('Problem size')
    fig.supylabel('Nr. clusters')
    fig.tight_layout()

    if not export:
        plt.show()
    else:
        plt.gcf().set_size_inches(IEEE_COL_WIDTH, 0.15 * A4_HEIGHT)
        plt.savefig('results/model_accuracy.pdf', bbox_inches='tight')

    # Return relevant plot figures
    figs = {
        'MaxModelingErrorRounded': '{:.0f}'.format(5 * ceil(max_err/5)),
    }
    return figs


def phase_accuracy(data):
    app, kwargs = next(iter(data.items()))

    for size in kwargs:

        # Get simulation results
        sims = [OffloadSimResults(app, True, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]

        # Get actual runtimes
        t_send_job_information   = recursive_map(OffloadSimResults.get_send_job_information_time, sims)
        t_wakeup                 = recursive_map(lambda x: OffloadSimResults.get_wakeup_time(x)[1], sims)
        t_retrieve_job_pointer   = recursive_map(lambda x: OffloadSimResults.get_retrieve_job_pointer_time(x)[1], sims)
        t_retrieve_job_arguments = recursive_map(lambda x: OffloadSimResults.get_retrieve_job_arguments_time(x)[1], sims)
        t_retrieve_job_operands  = recursive_map(lambda x: OffloadSimResults.get_retrieve_job_operands_time(x)[1], sims)
        t_job_execution          = recursive_map(lambda x: OffloadSimResults.get_job_execution_time(x)[1], sims)
        t_writeback_job_outputs  = recursive_map(lambda x: OffloadSimResults.get_writeback_job_outputs_time(x)[1], sims)
        t_notify_job_completion  = recursive_map(OffloadSimResults.get_notify_job_completion_time, sims)
        t_resume_operation       = recursive_map(OffloadSimResults.get_resume_operation_time, sims)
        t_total                  = recursive_map(OffloadSimResults.get_total_time, sims)

        # Get model runtimes
        t_send_job_information_model = [model_phase('send_job_information', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_wakeup_model = [model_phase('wakeup', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_retrieve_job_pointer_model = [model_phase('retrieve_job_pointer', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_retrieve_job_arguments_model = [model_phase('retrieve_job_arguments', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_retrieve_job_operands_model = [model_phase('retrieve_job_operands', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_job_execution_model = [model_phase('job_execution', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_writeback_job_outputs_model = [model_phase('writeback_job_outputs', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_notify_job_completion_model = [model_phase('notify_job_completion', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_resume_operation_model = [model_phase('resume_operation', app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]
        t_total_model = [model(app, nr_clusters, **size) for nr_clusters in ALL_NR_CLUSTER_CFGS]

        # Calculate errors
        err_send_job_information = 100 * np.abs((np.array(t_send_job_information) - np.array(t_send_job_information_model)) / np.array(t_send_job_information))
        err_wakeup = 100 * np.abs((np.array(t_wakeup) - np.array(t_wakeup_model)) / np.array(t_wakeup))
        err_retrieve_job_pointer = 100 * np.abs((np.array(t_retrieve_job_pointer) - np.array(t_retrieve_job_pointer_model)) / np.array(t_retrieve_job_pointer))
        err_retrieve_job_arguments = 100 * np.abs((np.array(t_retrieve_job_arguments) - np.array(t_retrieve_job_arguments_model)) / np.array(t_retrieve_job_arguments))
        err_retrieve_job_operands = 100 * np.abs((np.array(t_retrieve_job_operands) - np.array(t_retrieve_job_operands_model)) / np.array(t_retrieve_job_operands))
        err_job_execution = 100 * np.abs((np.array(t_job_execution) - np.array(t_job_execution_model)) / np.array(t_job_execution))
        err_writeback_job_outputs = 100 * np.abs((np.array(t_writeback_job_outputs) - np.array(t_writeback_job_outputs_model)) / np.array(t_writeback_job_outputs))
        err_notify_job_completion = 100 * np.abs((np.array(t_notify_job_completion) - np.array(t_notify_job_completion_model)) / np.array(t_notify_job_completion))
        err_resume_operation = 100 * np.abs((np.array(t_resume_operation) - np.array(t_resume_operation_model)) / np.array(t_resume_operation))
        err_total = 100 * np.abs((np.array(t_total) - np.array(t_total_model)) / np.array(t_total))

        # Print data
        print(app, size)
        print("A".ljust(15), t_send_job_information)
        print("B".ljust(15), t_wakeup)
        print("C".ljust(15), t_retrieve_job_pointer)
        print("D".ljust(15), t_retrieve_job_arguments)
        print("E".ljust(15), t_retrieve_job_operands)
        print("F".ljust(15), t_job_execution)
        print("G".ljust(15), t_writeback_job_outputs)
        print("H".ljust(15), t_notify_job_completion)
        print("I".ljust(15), t_resume_operation)
        print("total".ljust(15), t_total)
        print("A (model)".ljust(15), t_send_job_information_model)
        print("B (model)".ljust(15), t_wakeup_model)
        print("C (model)".ljust(15), t_retrieve_job_pointer_model)
        print("D (model)".ljust(15), t_retrieve_job_arguments_model)
        print("E (model)".ljust(15), t_retrieve_job_operands_model)
        print("F (model)".ljust(15), t_job_execution_model)
        print("G (model)".ljust(15), t_writeback_job_outputs_model)
        print("H (model)".ljust(15), t_notify_job_completion_model)
        print("I (model)".ljust(15), t_resume_operation_model)
        print("total (model)".ljust(15), t_total_model)
        print("A (error)".ljust(15), err_send_job_information)
        print("B (error)".ljust(15), err_wakeup)
        print("C (error)".ljust(15), err_retrieve_job_pointer)
        print("D (error)".ljust(15), err_retrieve_job_arguments)
        print("E (error)".ljust(15), err_retrieve_job_operands)
        print("F (error)".ljust(15), err_job_execution)
        print("G (error)".ljust(15), err_writeback_job_outputs)
        print("H (error)".ljust(15), err_notify_job_completion)
        print("I (error)".ljust(15), err_resume_operation)
        print("total (error)".ljust(15), err_total)
        print("")


def main():

    # Parse arguments
    args = parse_args()
    plot = args.plot
    export = args.export

    # Change global plot settings for export
    if export:
        global MARKER_SIZES
        MARKER_SIZES = [2, 2, 2, 3, 2, 2]
        plt.rcParams['font.size'] = '6'
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['xtick.major.width'] = 0.5
        plt.rcParams['xtick.minor.width'] = 0.5
        plt.rcParams['ytick.major.width'] = 0.5
        plt.rcParams['ytick.minor.width'] = 0.5
        plt.rcParams['patch.linewidth'] = 0.5

    # Important figures extracted from plots
    metrics = {}

    # Plot
    if plot == 'fig8' or plot == 'all':
        data = {
            'axpy': {'L': 1024},
            'atax': {'M': 1, 'N': 256},
        }
        metrics.update(fig8(data, export))
    if plot == 'fig8v2' or plot == 'all':
        sizes = {
            'axpy': {'L': 1024},
            'montecarlo': {'L': 512},
            'atax': {'M': 1, 'N': 256},
            'covariance': {'M': 1, 'N': 256},
            'gemm': {'M': 256, 'N': 1},
            'bfs': {'L': 1024},
        }
        metrics.update(fig8v2(sizes, export))
    if plot == 'fig9v2' or plot == 'all':
        sizes = {
            'axpy': {'L': 1024},
            'montecarlo': {'L': 512},
            'gemm': {'M': 256, 'N': 1},
            'atax': {'M': 1, 'N': 256},
            'covariance': {'M': 1, 'N': 256},
            'bfs': {'L': 1024},
        }
        metrics.update(fig9v2(sizes, export))
    if plot == 'fig10' or plot == 'all':
        data = {
            'axpy': {
                'key_to_scale': 'L',
                'sizes': [{'L': 32}, {'L': 64}, {'L': 128}]
            },
            # 'montecarlo': {
            #     'key_to_scale': 'L',
            #     'sizes': [{'L': 32}, {'L': 64}, {'L': 128}]
            # },
            # 'gemm': {
            #     'key_to_scale': 'M',
            #     'sizes': [{'M': 8, 'N': 1}, {'M': 16, 'N': 1}, {'M': 32, 'N': 1}]
            # },
            'atax': {
                'key_to_scale': 'N',
                'sizes': [{'M': 1, 'N': 8}, {'M': 1, 'N': 16}, {'M': 1, 'N': 32}]
            }
        }
        metrics.update(fig10(data, export))
    if plot == 'fig11' or plot == 'all':
        data = {'axpy': {'L': 1024}}
        # data = {'montecarlo': {'L': 512}}
        # data = {'atax': {'M': 1, 'N': 256}}
        # data = {'covariance': {'M': 1, 'N': 256}}
        # data = {'gemm': {'M': 256, 'N': 1}}
        fig11(data, export)
    if plot == 'fig13' or plot == 'all':
        data = {
            'axpy': [{'L': 256}, {'L': 512}, {'L': 768}, {'L': 1024}],
            'atax': [{'M': 1, 'N': 256}, {'M': 2, 'N': 256}, {'M': 3, 'N': 256}, {'M': 4, 'N': 256}],
            # 'gemm': [{'M': 256, 'N': 1}, {'M': 256, 'N': 2}, {'M': 256, 'N': 3}, {'M': 256, 'N': 4}],
        }
        metrics.update(fig13(data, export))
    if plot == 'phase_accuracy':
        data = {
            'atax': [{'M': 1, 'N': 256}, {'M': 2, 'N': 256}, {'M': 3, 'N': 256}, {'M': 4, 'N': 256}]
        }
        phase_accuracy(data)
    if plot == 'all':
        latex_metrics(metrics)


if __name__ == '__main__':
    main()
