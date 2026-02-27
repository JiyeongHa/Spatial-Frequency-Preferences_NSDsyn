#!/usr/bin/env python
"""Standalone worker script for permutation test jobs (replaces Snakemake rules).

Replicates three Snakemake rules:
  - shuffle_orientation
  - shuffle_eccentricity
  - run_model_shuffled_typed

Usage:
  # Shuffle step
  python run_perm_job.py shuffle --shuffle-type orientation --subj 01 --roi V1 --perm 0 \
      --output-dir /scratch/jh7685/projects/sfp_nsd/derivatives

  # Model fitting step
  python run_perm_job.py model --shuffle-type orientation --subj 01 --roi V1 --perm 0 \
      --output-dir /scratch/jh7685/projects/sfp_nsd/derivatives
"""
import argparse
import os
import sys

# Add repo root to path so sfp_nsdsyn is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def perm_batch(perm):
    """Return batch subdirectory name for a permutation index."""
    return f"batch_{int(perm) // 100 * 100:03d}"


# ── Path builders (match Snakemake output patterns exactly) ──────────────

def shuffle_input_paths(output_dir, dset, subj, roi, vs):
    subj_df = os.path.join(output_dir, 'dataframes', dset, 'model',
                           f'dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tavg-False.csv')
    precision = os.path.join(output_dir, 'dataframes', dset, 'precision',
                             f'precision-v_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    return subj_df, precision


def shuffle_output_path(output_dir, dset, subj, roi, vs, perm, shuffle_type):
    return os.path.join(output_dir, 'dataframes', dset, 'perm',
                        f'shuf-{shuffle_type}', subj,
                        f'perm-{perm}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_precision_merged.csv')


def model_output_path(output_dir, dset, subj, roi, vs, perm, shuffle_type, lr, max_epoch):
    return os.path.join(output_dir, 'sfp_model', 'results_2D', dset, 'perm',
                        f'shuf-{shuffle_type}', subj, perm_batch(perm),
                        f'perm-{perm}_model-params_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.pt')


# ── Step functions ───────────────────────────────────────────────────────

def run_shuffle(args):
    """Replicate shuffle_orientation / shuffle_eccentricity Snakemake rules."""
    from sfp_nsdsyn.bootstrapping import (shuffle_betas_within_freq_group,
                                          shuffle_eccentricities)

    subj_df_path, precision_path = shuffle_input_paths(
        args.output_dir, args.dset, args.subj, args.roi, args.vs)

    out_path = shuffle_output_path(
        args.output_dir, args.dset, args.subj, args.roi, args.vs,
        args.perm, args.shuffle_type)

    # Check inputs exist
    for p in [subj_df_path, precision_path]:
        if not os.path.isfile(p):
            sys.exit(f"Input not found: {p}")

    # Skip if output already exists
    if os.path.isfile(out_path):
        print(f"Output already exists, skipping: {out_path}")
        return

    np.random.seed(int(args.perm))

    subj_df = pd.read_csv(subj_df_path)
    precision_df = pd.read_csv(precision_path)
    df = subj_df.merge(precision_df, on=['sub', 'vroinames', 'voxel'])
    df = df.groupby(['sub', 'voxel', 'class_idx', 'vroinames']).mean().reset_index()

    if args.shuffle_type == 'orientation':
        df = shuffle_betas_within_freq_group(df, groupby_cols=['sub', 'voxel'], to_shuffle=['betas'])
    elif args.shuffle_type == 'eccentricity':
        df = shuffle_eccentricities(df, groupby_cols=['sub'])
    else:
        sys.exit(f"Unknown shuffle_type: {args.shuffle_type}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def run_model(args):
    """Replicate run_model_shuffled_typed Snakemake rule."""
    from sfp_nsdsyn import two_dimensional_model as model

    shuffled_csv = shuffle_output_path(
        args.output_dir, args.dset, args.subj, args.roi, args.vs,
        args.perm, args.shuffle_type)
    out_pt = model_output_path(
        args.output_dir, args.dset, args.subj, args.roi, args.vs,
        args.perm, args.shuffle_type, args.lr, args.max_epoch)

    if not os.path.isfile(shuffled_csv):
        sys.exit(f"Shuffled CSV not found: {shuffled_csv}")

    # Skip if output already exists
    if os.path.isfile(out_pt):
        print(f"Output already exists, skipping: {out_pt}")
        return

    os.makedirs(os.path.dirname(out_pt), exist_ok=True)

    subj_df = pd.read_csv(shuffled_csv)
    subj_model = model.SpatialFrequencyModel()
    subj_dataset = model.SpatialFrequencyDataset(subj_df, beta_col='betas')
    loss_history, model_history, _ = model.fit_model(
        subj_model, subj_dataset,
        learning_rate=float(args.lr),
        max_epoch=int(args.max_epoch),
        save_path=out_pt,
        print_every=10000,
        loss_all_voxels=False,
        anomaly_detection=False,
        amsgrad=False,
        eps=1e-8)
    print(f"Saved: {out_pt}")


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run one permutation job (shuffle or model fitting)")
    subparsers = parser.add_subparsers(dest='step', required=True)

    # Common args for both steps
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--shuffle-type', required=True, choices=['orientation', 'eccentricity'])
    common.add_argument('--subj', required=True, help='Subject ID, e.g. subj01')
    common.add_argument('--roi', required=True, help='ROI, e.g. V1')
    common.add_argument('--perm', required=True, type=int, help='Permutation index (0-99)')
    common.add_argument('--output-dir', required=True, help='Path to derivatives directory')
    common.add_argument('--dset', default='nsdsyn')
    common.add_argument('--vs', default='pRFsize')

    # Shuffle subcommand
    subparsers.add_parser('shuffle', parents=[common], help='Run shuffle step')

    # Model subcommand
    model_parser = subparsers.add_parser('model', parents=[common], help='Run model fitting step')
    model_parser.add_argument('--lr', default='0.0005')
    model_parser.add_argument('--max-epoch', default='30000')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.step == 'shuffle':
        run_shuffle(args)
    elif args.step == 'model':
        run_model(args)
