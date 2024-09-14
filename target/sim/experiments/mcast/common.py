# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import json
import json5
from mako.template import Template
import os
from pathlib import Path
import subprocess
import sys
from termcolor import cprint, colored

FILE_DIR = Path(__file__).parent.resolve()
TARGET_DIR = (FILE_DIR / '../../').resolve()
CFG_DIR = TARGET_DIR / 'cfg'
BIN_DIR = Path('bin')
VSIM_BUILDDIR = Path('work-vsim')

#####################################
# Specific to multicast experiments #
#####################################


def get_mcast_prefix(mcast):
    if mcast == 'true':
        return "M"
    elif mcast == 'false':
        return "U"
    elif mcast == 'hybrid':
        return "H"
    else:
        raise ValueError(f"Unknown multicast type: {mcast}")


def get_axes(mcast, n_clusters, size):
    return [
        get_mcast_prefix(mcast),
        n_clusters,
        size
    ]


######################
# Specific to Occamy #
######################


def extend_environment(vars, env=None):
    if not env:
        env = os.environ.copy()
    env.update(vars)
    return env


def run(cmd, env=None, dry_run=False):
    cmd = [str(arg) for arg in cmd]
    if dry_run:
        print(' '.join(cmd))
    else:
        p = subprocess.Popen(cmd, env=env)
        retcode = p.wait()
        if retcode != 0:
            sys.exit(retcode)


def make(target, vars=None, flags=[], dir=None, env=None, dry_run=False):
    var_assignments = [f'{key}={value}' for key, value in vars.items()]
    cmd = ['make', *var_assignments, target]
    if dir is not None:
        cmd.extend(['-C', dir])
    cmd.extend(flags)
    run(cmd, env=env, dry_run=dry_run)


def hw_config_file(hw_cfg=None):
    if hw_cfg is None:
        hw_cfg = 'default'
    return CFG_DIR / f'{hw_cfg}.hjson'


def unique_dir(base_dir, axes):
    dir = base_dir
    for axis in axes:
        dir = dir / str(axis)
    return dir


def simulator_binary(hw_cfg):
    return TARGET_DIR / BIN_DIR / hw_cfg / 'occamy_top.vsim'


def vsim_build_dir(hw_cfg):
    return VSIM_BUILDDIR / hw_cfg


def heterogeneous_elf(host_app, device_app, build_dir):
    return build_dir / f'{host_app}-{device_app}.elf'


def build_simulator_binary(hw_cfg, dry_run=False):
    cfg_file = hw_config_file(hw_cfg)
    sim_bin = simulator_binary(hw_cfg)
    cprint(f'Build hardware {colored(sim_bin, "cyan")}', attrs=['bold'])
    vars = {'CFG_OVERRIDE': cfg_file}
    make('rtl', vars, dir=TARGET_DIR, dry_run=dry_run)
    vars = {
        'VSIM_BUILDDIR': vsim_build_dir(hw_cfg),
        'BIN_DIR': sim_bin.parent,
    }
    make(sim_bin, vars, dir=TARGET_DIR, dry_run=dry_run)


def build_heterogeneous_app(host_app, device_app, build_dir, hw_cfg=None, defines=None, env=None,
                            dry_run=False):
    if defines:
        cflags = ' '.join([f'-D{name}={value}' for name, value in defines.items()])
        env = extend_environment({
            f'{device_app}_RISCV_CFLAGS': cflags,
            'SNRT_RISCV_CFLAGS': cflags
        }, env)
    elf = heterogeneous_elf(host_app, device_app, build_dir)
    target = f'{host_app}-{device_app}'
    vars = {
        'DEBUG': 'ON',
        'CFG_OVERRIDE': hw_config_file(hw_cfg),
        f'{host_app}_BUILDDIR': build_dir,
        f'{device_app}_BUILD_DIR': f'{build_dir}/device',
    }
    cprint(f'Build app {colored(elf, "cyan")}', attrs=['bold'])
    make(target, vars, flags=['-B'], dir=TARGET_DIR, env=env, dry_run=dry_run)


def process_traces(run_dir, device_elf=None, roi_spec=None, roi_vars={}, dry_run=False):
    logdir = run_dir / 'logs'
    cprint(f'Build traces {colored(logdir, "cyan")}', attrs=["bold"])
    vars = {'SIM_DIR': run_dir}
    flags = ['-j']
    if device_elf is not None:
        vars['BINARY'] = device_elf
    make('annotate', vars, flags=flags, dir=TARGET_DIR, dry_run=dry_run)

    # Read and render ROI specification template JSON
    if roi_spec is not None:
        target_roi_spec = run_dir / 'roi_spec.json'
        # Open ROI spec template
        with open(roi_spec, 'r') as f:
            spec = f.read()
        # Render template if required and copy to run directory
        if roi_vars:
            spec_template = Template(spec)
            spec = spec_template.render(**roi_vars)
            spec = json5.loads(spec)
            with open(target_roi_spec, 'w') as f:
                json.dump(spec, f, indent=4)
        else:
            with open(target_roi_spec, 'w') as f:
                f.write(spec)

        # Generate visual trace
        vars['ROI_SPEC'] = target_roi_spec
        make('visual-trace', vars, flags=flags, dir=TARGET_DIR, dry_run=dry_run)


def metric(run_dir, hartid, region, metric):
    perf = Path(run_dir) / 'logs' / f'hart_{hartid:05x}_perf.json'
    with open(perf, 'r') as f:
        data = json.load(f)
        return data[region][metric]


def runtime(run_dir, hartid, region):
    return metric(run_dir, hartid, region, 'tend') - metric(run_dir, hartid, region, 'tstart')
