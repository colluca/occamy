#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import argparse
import json
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import numpy as np
import pandas as pd
from pathlib import Path
from statistics import geometric_mean

import common
from SimResults import SimResults, SimRegion


# Experiment parameters
ALL_NR_CLUSTER_CFGS = [2, 4, 8, 16, 32]
ALL_TRANSFER_SIZES = [2, 4, 8, 16, 32]
ALL_MCAST_CFGS = [False, True]

# Export parameters
A4_HEIGHT = 11.7
IEEE_TEXT_WIDTH = 7.244
IEEE_TWO_COLUMN_SEP = 0.157
IEEE_COL_WIDTH = (IEEE_TEXT_WIDTH - IEEE_TWO_COLUMN_SEP) / 2
RESULTS_DIR = Path('results')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("plot", default="plot1", nargs='?', type=str)
    parser.add_argument("--export", action='store_true')
    return parser.parse_args()


def transfer_time(mcast, nr_clusters, size):
    axes = common.get_axes(mcast, nr_clusters, size)
    results = SimResults(common.unique_dir(Path('runs'), axes))
    if mcast == 'true':
        start = SimRegion('dma_9', 'transfer')
        end = start
        return results.get_timespan(start, end)
    elif mcast == 'false':
        start = SimRegion('dma_9', '0')
        end = SimRegion('dma_9', f'{nr_clusters - 1}')
        return results.get_timespan(start, end)
    elif mcast == 'hybrid':
        last_quad_idx = nr_clusters // 4 - 1
        last_quad_core_offset = last_quad_idx * 9 * 4 + 1
        last_quad_dma_core = last_quad_core_offset + 8
        start = SimRegion('dma_9', '4')
        end = SimRegion(f'dma_{last_quad_dma_core}', f'{nr_clusters - 1}')
        return results.get_timespan(start, end)
    else:
        raise ValueError(f"Unknown multicast type: {mcast}")


def gemm_time(mcast, tile):
    runs = Path('../gemm/runs')
    axes = [tile, common.get_mcast_prefix(mcast)]
    start = []
    end = []
    for hartid in range(1, 289):
        if hartid % 9 != 0:
            start.append(common.metric(
                common.unique_dir(runs, axes),
                hartid=hartid,
                region=3,
                metric='tstart'
            ))
            end.append(common.metric(
                common.unique_dir(runs, axes),
                hartid=hartid,
                region=5,
                metric='tstart'
            ))
    return max(end) - min(start)


def plot1(export=False):
    all_noslvmst = [2, 4, 8, 16]
    all_mcast = ['Baseline', 'Multicast']

    # Get data
    with open('results/synth.json', 'r') as f:
        data = json.load(f)

    # Extract area and frequency data
    area = {mcast: {noslvmst: 0 for noslvmst in all_noslvmst} for mcast in all_mcast}
    freq = {mcast: {noslvmst: 0 for noslvmst in all_noslvmst} for mcast in all_mcast}
    for x2, noslvmst in enumerate(all_noslvmst):
        for x1, mcast in enumerate(all_mcast):
            x = x2 * len(all_mcast) + x1
            total_area = data[x]["comb_area_ge - buffer_area_ge"] + data[x]["non_comb_area_ge"]
            area[mcast][noslvmst] = total_area / 1000
            freq[mcast][noslvmst] = data[x]["max_frequency"]

    # Plot data
    df = pd.DataFrame(area)
    ax = df.plot(kind='bar', figsize=(10, 6))

    # Add area increase labels on top of each cluster
    area_increase = [area['Multicast'][noslvmst]/area['Baseline'][noslvmst] for noslvmst in all_noslvmst]
    for i, bar in enumerate(ax.patches[len(all_noslvmst):]):
        ax.text(
            bar.get_x(),
            bar.get_height() * 1.01,
            f"{100*(area_increase[i]-1):.1f}\%",
            ha='center',
            va='bottom',
        )

    # Customize plot
    ax.set_xlabel('Nr. slaves and masters')
    ax.set_ylabel('Area [kGE]')
    ax.set_axisbelow(True)
    ax.set_yscale('log')
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0]/2, ylim[1] * 1.4])
    ax.grid(color='gainsboro', which='both', axis='y', linewidth=0.5)
    ax.tick_params(axis='x', labelrotation=0)
    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='{:.1f}', label_type='center', rotation=90)

    # Export or show plot
    if not export:
        plt.show()
    else:
        file = RESULTS_DIR / 'plot1.pdf'
        file.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().set_size_inches(0.28*IEEE_TEXT_WIDTH, 0.10 * A4_HEIGHT)
        plt.gcf().subplots_adjust(
            left=0.18,
            bottom=0.25,
            right=1,
            top=1
        )
        plt.savefig(file)

    # Fit second-order polynomials to the data
    mcast_coefficients = np.polyfit(all_noslvmst, [area['Multicast'][noslvmst] for noslvmst in all_noslvmst], deg=2)
    base_coefficients = np.polyfit(all_noslvmst, [area['Baseline'][noslvmst] for noslvmst in all_noslvmst], deg=2)

    # Return metrics
    return {
        'EightByEightCrossbarOverheadkGE': '{:.1f}'.format(area['Multicast'][8] - area['Baseline'][8]),
        'EightByEightCrossbarOverheadPercent': '{:.1f}'.format(100 * (area['Multicast'][8] - area['Baseline'][8]) / area['Baseline'][8]),
        'AsymptoticOverheadPercent': '{:.1f}'.format(100 * (mcast_coefficients[0] / base_coefficients[0] - 1)),
        'SixteenBySixteenCrossbarFrequencyOverheadPercent': '{:.1f}'.format(100 * (1 - freq['Multicast'][16] / freq['Baseline'][16])),
    }


def plot2(export=False):
    # Get data
    mcast_speedup = {}
    hybrid_speedup = {}
    pfrac = {}
    for size in ALL_TRANSFER_SIZES:
        mcast_speedup[size] = {}
        hybrid_speedup[size] = {}
        pfrac[size] = {}
        for nr_clusters in ALL_NR_CLUSTER_CFGS:
            mcast = transfer_time('true', nr_clusters, size)
            base = transfer_time('false', nr_clusters, size)
            su_mcast = base / mcast
            parallel_frac = (nr_clusters * (su_mcast - 1)) / (su_mcast * (nr_clusters - 1))
            mcast_speedup[size][nr_clusters] = su_mcast
            pfrac[size][nr_clusters] = parallel_frac
            if nr_clusters in [8, 16, 32]:
                hybrid = transfer_time('hybrid', nr_clusters, size)
                su_hybrid = base / hybrid
                hybrid_speedup[size][nr_clusters] = su_hybrid
    df = pd.DataFrame(mcast_speedup)
    df.rename(columns=lambda x: str(x) + '\,KiB', inplace=True)

    # Plot speedup bars over baseline
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.7)

    # Add parallel fraction labels on top of every third bar
    # and superimpose speedup bars over hybrid for 8, 16 and 32 clusters
    for i, bar in enumerate(ax.patches):
        i_size = i // len(ALL_NR_CLUSTER_CFGS)
        i_nrc = i % len(ALL_NR_CLUSTER_CFGS)
        size_str = df.columns[i_size]
        size = int(size_str.replace('\,KiB', ''))
        nr_clusters = df.index[i_nrc]
        # Create parallel fraction label
        if size == 32:
            parallel_frac = pfrac[size][nr_clusters]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{100*parallel_frac:.0f}\%",
                ha='center',
                va='bottom',
            )
        if nr_clusters in [8, 16, 32]:
            ax.bar(
                bar.get_x() + bar.get_width() / 2,
                hybrid_speedup[size][nr_clusters],
                width=bar.get_width(),
                color='white',
                alpha=0.7
            )

    # Customize plot
    ax.set_xlabel('Nr. clusters')
    ax.set_ylabel('Speedup')
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1]*1.1])
    ax.legend(title='Transfer size', ncol=2)
    ax.set_axisbelow(True)
    ax.grid(color='gainsboro', which='both', axis='y', linewidth=0.5)
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(axis='y', which='minor', labelsize=0)
    ax.tick_params(axis='x', labelrotation=0)

    # Export or show plot
    if not export:
        plt.show()
    else:
        file = RESULTS_DIR / 'plot2.pdf'
        file.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().set_size_inches(0.34*IEEE_TEXT_WIDTH, 0.10 * A4_HEIGHT)
        plt.gcf().subplots_adjust(
            left=0.13,
            bottom=0.25,
            right=1,
            top=1
        )
        plt.savefig(file)

    # Return metrics
    geomean_speedup = geometric_mean([mcast_speedup[size][32]/hybrid_speedup[size][32] for size in ALL_TRANSFER_SIZES])
    return {
        'ThirtyTwoClusterEightKiBParallelFraction': '{:.0f}'.format(100*pfrac[8][32]),
        'ThirtyTwoClusterTwoKiBSpeedup': '{:.2f}'.format(mcast_speedup[2][32]),
        'ThirtyTwoClusterThirtyTwoKiBSpeedup': '{:.2f}'.format(mcast_speedup[32][32]),
        'ThirtyTwoClusterGeometricMeanSpeedup': '{:.2f}'.format(geomean_speedup),
    }


def plot3(export=False):
    # System parameters
    peak_memory_bandwidth = 64  # GB/s
    peak_compute_performance_32_clusters = 32 * 8 * 2  # GFLOPS

    # Define Operational Intensity range
    min_oi, max_oi = 1, 36
    oi_values = np.logspace(np.log10(min_oi), np.log10(max_oi), num=500)

    # Compute memory bandwidth and compute-bound ceilings
    memory_bound_performance = peak_memory_bandwidth * oi_values
    compute_bound_performance_32_clusters = np.full_like(oi_values, peak_compute_performance_32_clusters)

    # Create plot
    plt.figure(figsize=(10, 7))

    # Plot the roofline line
    roofline_performance = np.minimum(memory_bound_performance, compute_bound_performance_32_clusters)
    plt.loglog(oi_values, roofline_performance, label='Roofline', color='black')

    # Get GEMM performance data
    initial_base_time = gemm_time('false', 'first') / 10**9  # seconds
    initial_mcast_time = gemm_time('true', 'first') / 10**9
    initial_hybrid_time = gemm_time('hybrid', 'first') / 10**9
    ss_base_time = gemm_time('false', 'steady') / 10**9  # seconds
    ss_mcast_time = gemm_time('true', 'steady') / 10**9
    ss_hybrid_time = gemm_time('hybrid', 'steady') / 10**9
    ops = 32 * 8 * 16 * 256 * 2
    initial_base_perf = (ops / initial_base_time) / 10**9  # GFLOPs/sec
    initial_mcast_perf = (ops / initial_mcast_time) / 10**9
    initial_hybrid_perf = (ops / initial_hybrid_time) / 10**9
    ss_base_perf = (ops / ss_base_time) / 10**9  # GFLOPs/sec
    ss_mcast_perf = (ops / ss_mcast_time) / 10**9
    ss_hybrid_perf = (ops / ss_hybrid_time) / 10**9

    # Calculate operational intensities
    base_bytes = 32 * (8 * 256 + 16 * 256 + 8 * 16) * 8
    mcast_bytes = (32 * (8 * 256 + 8 * 16) + 16 * 256) * 8
    hybrid_bytes = (32 * (8 * 256 + 8 * 16) + 8 * 16 * 256) * 8
    base_oi = ops / base_bytes
    mcast_oi = ops / mcast_bytes
    hybrid_oi = ops / hybrid_bytes

    # Calculate steady-state operational intensities, i.e. neglect A transfers
    ss_base_bytes = 32 * (16 * 256 + 8 * 16) * 8
    ss_mcast_bytes = (32 * (8 * 16) + 16 * 256) * 8
    ss_hybrid_bytes = (32 * (8 * 16) + 8 * 16 * 256) * 8
    ss_base_oi = ops / ss_base_bytes
    ss_mcast_oi = ops / ss_mcast_bytes
    ss_hybrid_oi = ops / ss_hybrid_bytes

    # Plot the kernels
    N_IMPLS = 3
    kernels = [
        {'oi': base_oi, 'performance': initial_base_perf},
        {'oi': mcast_oi, 'performance': initial_mcast_perf},
        {'oi': hybrid_oi, 'performance': initial_hybrid_perf},
        {'oi': ss_base_oi, 'performance': ss_base_perf},
        {'oi': ss_mcast_oi, 'performance': ss_mcast_perf},
        {'oi': ss_hybrid_oi, 'performance': ss_hybrid_perf},
    ]
    colors = [mcolors.TABLEAU_COLORS['tab:blue'], mcolors.TABLEAU_COLORS['tab:orange'], mcolors.TABLEAU_COLORS['tab:green']]
    markers = ['^', '*']
    markersizes = [3, 5]
    for i, kernel in enumerate(kernels):
        tile = i // N_IMPLS
        impl = i % N_IMPLS

        # Draw achieved performance for the kernel
        plt.plot(
            kernel['oi'],
            kernel['performance'],
            color=colors[impl],
            marker=markers[tile],
            markersize=markersizes[tile]
        )

        # Add a text label for the performance value
        horizontal_alignment = 'left'
        xcoord = kernel['oi'] * 1.1
        if impl == 1 and tile == 1:
            horizontal_alignment = 'right'
            xcoord = kernel['oi'] * 0.9
        plt.text(
            xcoord,
            kernel['performance'] * 0.85,
            '{:.1f}'.format(kernel['performance']),
            horizontalalignment=horizontal_alignment
        )

    # Manually create legends
    tile_legend = [
        mpl.lines.Line2D(
            [], [], color='grey', marker=markers[0], linestyle='None',
            markersize=markersizes[0], label='1st tile'
        ),
        mpl.lines.Line2D(
            [], [], color='grey', marker=markers[1], linestyle='None',
            markersize=markersizes[1], label='Steady-state'
        )
    ]
    impl_legend = [
        mpl.patches.Patch(color=colors[0], label='Baseline'),
        mpl.patches.Patch(color=colors[2], label='Hierarchical'),
        mpl.patches.Patch(color=colors[1], label='Multicast'),
    ]
    legend_tile = plt.gca().legend(handles=tile_legend, loc='upper left', handlelength=1)
    plt.gca().legend(handles=impl_legend, loc='lower right')
    plt.gca().add_artist(legend_tile)

    # Customize plot
    plt.xlabel('Operational Intensity [FLOPs/Byte]')
    plt.ylabel('Performance [GFLOPS]', loc='top')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
    plt.gca().yaxis.get_minor_formatter().set_scientific(False)
    plt.grid(True, which='both', linewidth=0.5, color='gainsboro')

    # Remove overlapping Y tick labels
    labels = plt.gca().get_yticklabels(minor=True)
    for i, label in enumerate(labels):
        if label.get_position()[1] in [70, 90.0]:
            label.set_visible(False)

    # Export or show plot
    if not export:
        plt.show()
    else:
        file = RESULTS_DIR / 'plot3.pdf'
        file.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().set_size_inches(0.34*IEEE_TEXT_WIDTH, 0.10 * A4_HEIGHT)
        plt.gcf().subplots_adjust(
            left=0.15,
            bottom=0.25,
            right=1,
            top=1
        )
        plt.savefig(file)

    # Return metrics
    return {
        'BaselineTileNOperationalIntensity': '{:.1f}'.format(ss_base_oi),
        'BaselineTileNPerformanceGFLOPS': '{:.1f}'.format(ss_base_perf),
        'BaselineTileNPerformancePercentage': '{:.0f}'.format(100 * ss_base_perf / (ss_base_oi * peak_memory_bandwidth)),
        'HybridTileNOperationalIntensityIncrease': '{:.2f}'.format(ss_hybrid_oi / ss_base_oi),
        'HybridTileNPerformanceIncrease': '{:.2f}'.format(ss_hybrid_perf / ss_base_perf),
        'MulticastTileNOperationalIntensityIncrease': '{:.2f}'.format(ss_mcast_oi / ss_base_oi),
        'MulticastTileNPerformanceIncrease': '{:.2f}'.format(ss_mcast_perf / ss_base_perf),
        'MulticastTileNPerformanceIncreaseOverHybridPercentage': '{:.0f}'.format(100 * (ss_mcast_perf / ss_hybrid_perf - 1)),
        'MulticastTileNPerformanceGFLOPS': '{:.1f}'.format(ss_mcast_perf),
    }


def latex_metrics(metrics):
    # Auxiliary function to format a metric as a LaTeX command
    def latex_metric(name, value):
        return f"\\newcommand{{\\Result{name}}}{{{value}}}\n"

    # Create file
    with open(RESULTS_DIR / 'metrics.tex', 'w') as f:
        [f.write(latex_metric(name, value)) for name, value in metrics.items()]


def main():

    # Parse arguments
    args = parse_args()
    plot = args.plot
    export = args.export

    # Change global plot settings for export
    if export:
        plt.rcParams['font.family'] = 'Latin Modern Roman'
        mpl.rcParams['font.size'] = 6
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['xtick.major.pad'] = 2
        plt.rcParams['axes.labelpad'] = 2
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['xtick.major.width'] = 0.5
        plt.rcParams['xtick.minor.width'] = 0.5
        plt.rcParams['ytick.major.width'] = 0.5
        plt.rcParams['ytick.minor.width'] = 0.5
        plt.rcParams['patch.linewidth'] = 0.5
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['legend.handletextpad'] = 0.5
        plt.rcParams['legend.columnspacing'] = 1
        # Use Latex backend for rendering
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}\usepackage{lmodern}'


    # Plot
    metrics = {}
    if plot == 'plot1' or plot == 'all':
        metrics.update(plot1(export))
    if plot == 'plot2' or plot == 'all':
        metrics.update(plot2(export))
    if plot == 'plot3' or plot == 'all':
        metrics.update(plot3(export))
    if plot == 'all':
        latex_metrics(metrics)


if __name__ == '__main__':
    main()
