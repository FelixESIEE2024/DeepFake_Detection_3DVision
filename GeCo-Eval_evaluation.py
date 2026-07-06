#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Benchmark Evaluator for Video Consistency.

This script iterates over a structured dataset of video frames (Model -> Category -> Clip),
computes consistency scores using VGGT (structure) and UFM (motion), and aggregates results.

Outputs:
1. Per-clip CSVs: Saved as `<model_key>.csv` (clip_id, category, motion, depth, fused).
2. Summary: Prints aggregated mean scores per category to stdout.

Sampling Strategy:
Instead of processing every frame (which is slow), this script divides the video into 
a fixed number of temporal windows (e.g., 4 windows of 3 seconds each) and computes 
the score within those windows.
"""

import sys
import os
import math
import argparse
import random
import csv
import shlex
import numpy as np
from typing import List, Tuple

# Add paths to external submodules
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'external', 'UFM'))
sys.path.append(os.path.join(base_path, 'external', 'vggt'))

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n=1):
            return None

        def set_postfix_str(self, s):
            return None

        def close(self):
            return None

        @staticmethod
        def write(msg):
            print(msg)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from uniflowmatch.models.ufm import UniFlowMatchConfidence

from utils import (
    load_image_ufm, predict_correspondences, rigid_flow_from_camera_motion,
    create_confidence_mask_torch, normalize_flow_to_unitless, vggt_infer,
    compute_normalized_depth_error_unidirectional,masked_mean, build_fixed_len_cover_windows
)

# ---------------------- Configuration ----------------------

# Mapping of Model Name -> Native FPS
# Used to determine time-based sampling
dic_model_fps = {
    'Gen_CogVideoX_2b': 16,
    'Gen_CogVideoX_5b': 16,
    'Gen_CogVideoX1.5_5b': 16,
    'Gen_LTX': 30,
    'Gen_SORA2': 30,
    'Gen_Veo3.1': 24,
    'Gen_WAN2.2': 16,
    'Gen_HunyuanVideo': 24,
    'Gen_SORA2_480p': 30,
    'Gen_Veo3.1_480p': 24,
    'Gen_WAN2.2_480p': 16,
    'Gen_CogVideoX1.5_5b_480p': 16,
}

ALL_CATEGORIES = [
    "indoor_prompts",
    "object_centric_prompts",
    "outdoor_prompts",
    "stress_test_prompts",
]


def list_frame_files(d):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    return sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f.lower())[1] in exts])


def plan_video_sampling(video_dir, model_key, args):
    """Pre-compute sampled windows for logging and progress reporting."""
    all_files = list_frame_files(video_dir)
    if len(all_files) < 2:
        return {
            "all_files": all_files,
            "fps_native": float(dic_model_fps[model_key]),
            "eval_fps_eff": min(float(args.eval_fps), float(dic_model_fps[model_key])),
            "windows": [],
            "window_lengths": [],
            "total_sampled_frames": 0,
            "n_windows": 0,
        }

    fps_native = float(dic_model_fps[model_key])
    eval_fps_eff = min(float(args.eval_fps), fps_native)
    windows, win_sec_eff = build_fixed_len_cover_windows(
        all_files=all_files,
        fps_native=fps_native,
        eval_fps=eval_fps_eff,
        win_sec=args.win_sec,
        max_windows=args.max_windows
    )

    return {
        "all_files": all_files,
        "fps_native": fps_native,
        "eval_fps_eff": eval_fps_eff,
        "windows": windows,
        "window_lengths": [len(w) for w in windows],
        "total_sampled_frames": int(sum(len(w) for w in windows)),
        "n_windows": len(windows),
        "win_sec_eff": win_sec_eff,
    }


def print_run_configuration(args, model_keys, suffixes, tasks, task_plans, device, compute_dtype):
    total_raw_frames = sum(len(plan["all_files"]) for plan in task_plans.values())
    total_windows = sum(plan["n_windows"] for plan in task_plans.values())
    total_sampled_frames = sum(plan["total_sampled_frames"] for plan in task_plans.values())

    print("\n" + "=" * 80)
    print("GeCo Evaluation Run")
    print("=" * 80)
    print(f"Frames root        : {args.frames_root}")
    print(f"Models             : {', '.join(model_keys)}")
    print(f"Categories         : {', '.join(args.categories)}")
    print(f"Device             : {device}")

    print("")
    print("Taille de l'échantillon")
    print(f"  win_sec          : {args.win_sec}")
    print(f"  max_windows par video    : {args.max_windows}")
    print(f"  frame par seconde         : {args.eval_fps}")
    print(f"  pair_stride      : {args.pair_stride}")
    print(f"  output_dir       : {args.output_dir}")
    print(f"  save_frame_details : {args.save_frame_details}")
    print("")

    print("Planned workload")
    print(f"  nb de videos            : {len(tasks)}")
    print(f"  nb de windows en tout  : {total_windows}")
    print(f"  nb d'iteration total   : {total_sampled_frames}")
    print("=" * 80 + "\n")


def print_clip_header(clip_idx, total_clips, model_key, cat, clip_id, clip_plan):
    print("-" * 80)
    print(f"Clip {clip_idx}/{total_clips}  |  {model_key} / {cat} / {clip_id}")
    print(
        f"raw={len(clip_plan['all_files'])}  "
        f"sampled={clip_plan['total_sampled_frames']}  "
        f"windows={clip_plan['n_windows']}  "
        f"native_fps={clip_plan['fps_native']:.2f}  "
        f"eval_fps={clip_plan['eval_fps_eff']:.2f}"
    )
    if clip_plan["window_lengths"]:
        print(f"window frames={clip_plan['window_lengths']}")
    print("-" * 80)


def fmt_score(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{value:.6f}" if np.isfinite(value) else ""


def write_csv_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_command_file(output_dir):
    command_path = os.path.join(output_dir, "00_commande.txt")
    command_str = " ".join(shlex.quote(arg) for arg in [sys.executable] + sys.argv)
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(command_str + "\n")

# ---------------------- Evaluation Logic ----------------------

@torch.inference_mode()
def evaluate_one_window(image_files, vggt_model, ufm_model, device, compute_dtype, args, frame_pbar=None):
    """
    Evaluates consistency for a single window of frames.
    
    Steps:
    1. Run VGGT once on the whole window to get Camera Poses & Depth.
    2. Iterate through pairs (Source -> Target).
    3. Compute Motion Error (Flow vs Ego-motion).
    4. Compute Depth Error (Bidirectional Reprojection).
    5. Fuse errors based on occlusion logic.
    """
    # Load first image to get dimensions
    img0 = load_image_ufm(image_files[0]) 
    H, W = img0.shape[:2]

    # Cache UFM images to reduce disk I/O latency
    cached_imgs = [img0] + [load_image_ufm(fp) for fp in image_files[1:]]

    # 1. VGGT Inference (Batch)
    imgs = load_and_preprocess_images(image_files, "crop", patch_size=14).to(device)[None]
    geo = vggt_infer(
        vggt_model, imgs, upsample_size=(H, W),
        point_prediction=False, compute_dtype=compute_dtype, device=device
    )
    K = geo["intrinsic"]   # (N,3,3)
    E = geo["extrinsic"]   # (N,3,4)
    D = geo["depth_map"]   # (N,H,W,1)
    C = geo["vggt_conf"]   # (N,H,W,1)

    N = len(image_files)
    per_frame_motion, per_frame_depth, per_frame_fused = [], [], []
    frame_rows = []

    step = max(1, int(args.pair_stride))

    for src in range(N):
        src_img = cached_imgs[src]

        # Accumulators
        mot_sum, mot_cnt = torch.zeros((H, W), device=device), torch.zeros((H, W), device=device)
        dep_sum, dep_cnt = torch.zeros((H, W), device=device), torch.zeros((H, W), device=device)
        fus_sum, fus_cnt = torch.zeros((H, W), device=device), torch.zeros((H, W), device=device)

        # Source Confidence Mask
        conf_src_mask = create_confidence_mask_torch(
            C[src, ..., 0], percentile_val=args.conf_percentile, min_threshold=args.conf_min
        )

        for tgt in range(0, N, step):
            if tgt == src: continue

            tgt_img = cached_imgs[tgt]

            # 2. UFM Flow & Covisibility
            flow_uv, cov = predict_correspondences(ufm_model, src_img, tgt_img, str(args.ufm_longside))
            mask_covis = torch.isfinite(cov) & (cov > args.covis_thresh)

            # 3. Ego Motion Check
            ego_uv, mask_reproj = rigid_flow_from_camera_motion(D[src], K[[src, tgt]], E[[src, tgt]])
            mask_reproj = mask_reproj & conf_src_mask

            # Motion Residual
            residual_uv = flow_uv - ego_uv
            residual_unitless = normalize_flow_to_unitless(residual_uv, K[src])
            e_xy = torch.linalg.vector_norm(residual_unitless, dim=-1)

            # 4. Depth Consistency (Unidirectional for efficient computing)
            # Use the robust check to avoid false positives in occluded regions
            e_z, dp_valid_basic, _, dz, dz_rel_p = compute_normalized_depth_error_unidirectional(
                D[src], K[src], E[src], D[tgt], K[tgt], E[tgt], align_corners=True
            )
            depth_valid = dp_valid_basic & conf_src_mask

            # 5. Fusion Logic
            mot_valid      = mask_covis & mask_reproj
            depth_no_occ   = mask_covis & mask_reproj & depth_valid
            wrong_occ      = (~mask_covis) & mask_reproj & depth_valid & ~(depth_valid & (dz_rel_p > args.tau_z))

            # Accumulate
            mot_sum += e_xy * mot_valid.float()
            mot_cnt += mot_valid.float()

            depth_mask = depth_no_occ | wrong_occ
            dep_sum   += e_z * depth_mask.float()
            dep_cnt   += depth_mask.float()

            pair_fused = torch.zeros_like(e_xy)
            pair_fused[depth_no_occ] = (e_xy[depth_no_occ] + e_z[depth_no_occ]) / 2.0
            pair_fused[wrong_occ] = e_z[wrong_occ]
            pair_fused[mot_valid & ~depth_no_occ] = e_xy[mot_valid & ~depth_no_occ] # Motion only fallback

            fus_sum += pair_fused
            fus_cnt += (depth_no_occ | wrong_occ | (mot_valid & ~depth_no_occ)).float()

        # Compute scalars for this frame
        def safe_div(s, c): return s / c.clamp_min(1.0)
        
        motion_avg = safe_div(mot_sum, mot_cnt)
        depth_avg  = safe_div(dep_sum, dep_cnt)
        fused_avg  = safe_div(fus_sum, fus_cnt)

        frame_motion = float(masked_mean(motion_avg, mot_cnt > 0))
        frame_depth = float(masked_mean(depth_avg, dep_cnt > 0))
        frame_fused = float(masked_mean(fused_avg, fus_cnt > 0))

        per_frame_motion.append(frame_motion)
        per_frame_depth.append(frame_depth)
        per_frame_fused.append(frame_fused)

        if args.save_frame_details:
            frame_rows.append({
                "frame_idx_in_window": src,
                "frame_name": os.path.basename(image_files[src]),
                "motion": frame_motion,
                "depth": frame_depth,
                "fused": frame_fused,
            })
        if frame_pbar is not None:
            frame_pbar.update(1)

    # Window Averaging
    def nanmean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)) if arr.size > 0 else float('nan')

    return (
        nanmean(per_frame_motion),
        nanmean(per_frame_depth),
        nanmean(per_frame_fused),
        len(image_files),
        frame_rows,
    )


def evaluate_video_folder(video_dir, model_key, vggt_model, ufm_model, device, compute_dtype, args, clip_plan=None):
    """Entry point for a single video folder."""
    clip_plan = clip_plan or plan_video_sampling(video_dir, model_key, args)
    all_files = clip_plan["all_files"]
    if len(all_files) < 2:
        return float('nan'), float('nan'), float('nan'), [], []
    windows = clip_plan["windows"]
    
    if not windows:
        return float('nan'), float('nan'), float('nan'), [], []

    # Run windows
    motion_vals, depth_vals, fused_vals, weights = [], [], [], []
    window_rows, frame_rows = [], []
    for win_idx, wfiles in enumerate(windows, start=1):
        print(f"Window {win_idx}/{len(windows)}  |  {len(wfiles)} sampled frames")
        frame_pbar = tqdm(
            total=len(wfiles),
            desc="Frames",
            unit="frame",
            leave=True,
            dynamic_ncols=True,
        )
        try:
            m, d, f, w, frame_rows_window = evaluate_one_window(
                wfiles, vggt_model, ufm_model, device, compute_dtype, args, frame_pbar=frame_pbar
            )
        finally:
            frame_pbar.close()

        window_rows.append({
            "window_idx": win_idx,
            "window_frames": len(wfiles),
            "motion": m,
            "depth": d,
            "fused": f,
        })

        if args.save_frame_details:
            for row in frame_rows_window:
                row_with_window = dict(row)
                row_with_window["window_idx"] = win_idx
                frame_rows.append(row_with_window)

        if np.isfinite(m) and np.isfinite(d) and np.isfinite(f) and w >= 2:
            motion_vals.append(m); depth_vals.append(d); fused_vals.append(f); weights.append(w)

    if not weights:
        return float('nan'), float('nan'), float('nan'), window_rows, frame_rows

    # Weighted Average across windows
    W = np.array(weights, dtype=float)
    W = W / W.sum()

    motion_score = float((np.array(motion_vals) * W).sum())
    depth_score  = float((np.array(depth_vals)  * W).sum())
    fused_score  = float((np.array(fused_vals)  * W).sum())
    return motion_score, depth_score, fused_score, window_rows, frame_rows


# ---------------------- Helper: Filter Directories ----------------------
def pick_suffix_subdirs(cat_dir: str, suffixes: List[str] | None) -> List[str]:
    """Returns subdirectories ending with specific suffixes (e.g., '_1')."""
    if not os.path.isdir(cat_dir): return []
    subs = [d for d in sorted(os.listdir(cat_dir)) if os.path.isdir(os.path.join(cat_dir, d))]
    
    if suffixes is None: return subs
    
    keep = []
    suffixes = [str(s) for s in suffixes]
    for d in subs:
        for s in suffixes:
            if d.endswith(f"_{s}"):
                keep.append(d)
                break
    return keep

# ---------------------- Main Execution ----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Batch evaluate video consistency across models and categories."
    )
    p.add_argument("--frames_root", type=str, required=True,
                   help="Root folder containing model subfolders (Gen_*).")
    p.add_argument("--models", type=str, nargs="+", required=True,
                   help="Model keys (e.g. Gen_SORA2) or 'all'.")
    p.add_argument("--categories", type=str, nargs="+", default=ALL_CATEGORIES,
                   choices=ALL_CATEGORIES, help="Categories to evaluate.")
    p.add_argument("--suffixes", type=str, nargs="+", default=["all"],
                   help="Filter clip directories by suffix (e.g. '1', '2' or 'all').")

    # Sampling
    p.add_argument("--win_sec", type=float, default=3.0, help="Window duration (sec).")
    p.add_argument("--max_windows", type=int, default=4, help="Max windows per clip.")
    p.add_argument("--eval_fps", type=float, default=8.0, help="Sampling FPS.")

    # Algorithm
    p.add_argument("--pair_stride", type=int, default=1, help="Stride for pair comparison within window.")
    p.add_argument("--covis_thresh", type=float, default=0.5, help="Flow covisibility threshold.")
    p.add_argument("--conf_percentile", type=float, default=20.0, help="Depth confidence percentile.")
    p.add_argument("--conf_min", type=float, default=0.2, help="Depth confidence min.")
    p.add_argument("--tau_z", type=float, default=0.02, help="Occlusion depth margin.")
    p.add_argument("--ufm_longside", type=int, default=255, help="UFM inference resolution.")

    p.add_argument("--print_per_clip", action="store_true", help="Print per-clip scores to stdout.")
    p.add_argument("--output_dir", type=str, default=".", help="Directory where CSV outputs are saved.")
    p.add_argument("--save_frame_details", action="store_true", help="Save one CSV row per sampled source frame.")
    return p.parse_args()

def main():
    args = parse_args()

    # 1. Resolve Models
    if len(args.models) == 1 and args.models[0].lower() == "all":
        model_keys = sorted([m for m in dic_model_fps.keys() if os.path.isdir(os.path.join(args.frames_root, m))])
    else:
        model_keys = args.models

    # 2. Resolve Suffixes
    suffixes_arg = [s.lower() for s in args.suffixes]
    suffixes = None if (len(suffixes_arg) == 1 and suffixes_arg[0] == "all") else suffixes_arg

    # 3. Build Task List
    tasks = []
    for model_key in model_keys:
        for cat in args.categories:
            cat_dir = os.path.join(args.frames_root, model_key, cat)
            chosen = pick_suffix_subdirs(cat_dir, suffixes)
            for clip_id in chosen:
                tasks.append((model_key, cat, clip_id, os.path.join(cat_dir, clip_id)))

    if not tasks:
        print("No clips found matching criteria.")
        return

    task_plans = {}
    for (model_key, cat, clip_id, video_dir) in tasks:
        task_plans[(model_key, cat, clip_id)] = plan_video_sampling(video_dir, model_key, args)

    # 4. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32

    print_run_configuration(args, model_keys, suffixes, tasks, task_plans, device, compute_dtype)
    print("\n Loading models...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    ufm_model  = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(dtype=torch.float32, device=device).eval()
    print("=" * 80 + "\n")
    # 5. Run Evaluation
    per_model_category = {}
    per_model_clip_rows = {}
    clip_summary_rows = []
    window_detail_rows = []
    frame_detail_rows = []

    total_clips = len(tasks)
    for clip_idx, (model_key, cat, clip_id, video_dir) in enumerate(tasks, start=1):
        clip_plan = task_plans[(model_key, cat, clip_id)]
        print_clip_header(clip_idx, total_clips, model_key, cat, clip_id, clip_plan)
        
        try:
            m, d, f, window_rows, frame_rows = evaluate_video_folder(
                video_dir, model_key, vggt_model, ufm_model, device, compute_dtype, args, clip_plan=clip_plan
            )
        except Exception as e:
            print(f"Error on {video_dir}: {e}", file=sys.stderr)
            m, d, f, window_rows, frame_rows = float('nan'), float('nan'), float('nan'), [], []

        if args.print_per_clip:
            print(f"{model_key},{cat},{clip_id},{m:.4f},{d:.4f},{f:.4f}")

        category_bucket = per_model_category.setdefault(
            (model_key, cat),
            {"scores": [], "n_windows": 0, "n_frames": 0}
        )
        category_bucket["scores"].append((m, d, f))
        category_bucket["n_windows"] += len(window_rows)
        category_bucket["n_frames"] += sum(row["window_frames"] for row in window_rows)
        per_model_clip_rows.setdefault(model_key, []).append((clip_id, cat, m, d, f))
        clip_summary_rows.append({
            "model_key": model_key,
            "category": cat,
            "clip_id": clip_id,
            "raw_frames": len(clip_plan["all_files"]),
            "sampled_frames": clip_plan["total_sampled_frames"],
            "n_windows": clip_plan["n_windows"],
            "motion": m,
            "depth": d,
            "fused": f,
        })

        for row in window_rows:
            window_detail_rows.append({
                "model_key": model_key,
                "category": cat,
                "clip_id": clip_id,
                "window_idx": row["window_idx"],
                "window_frames": row["window_frames"],
                "motion": row["motion"],
                "depth": row["depth"],
                "fused": row["fused"],
            })

        if args.save_frame_details:
            for row in frame_rows:
                frame_detail_rows.append({
                    "model_key": model_key,
                    "category": cat,
                    "clip_id": clip_id,
                    "window_idx": row["window_idx"],
                    "frame_idx_in_window": row["frame_idx_in_window"],
                    "frame_name": row["frame_name"],
                    "motion": row["motion"],
                    "depth": row["depth"],
                    "fused": row["fused"],
                })

    # 6. Save CSVs
    os.makedirs(args.output_dir, exist_ok=True)
    write_command_file(args.output_dir)

    for model_key, rows in per_model_clip_rows.items():
        rows_sorted = sorted(rows, key=lambda r: (r[1], r[0]))
        with open(os.path.join(args.output_dir, f"{model_key}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_id", "category", "motion", "depth", "fused"])
            for r in rows_sorted:
                writer.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}"])

    total_raw_frames = sum(len(plan["all_files"]) for plan in task_plans.values())
    total_windows = sum(plan["n_windows"] for plan in task_plans.values())
    total_sampled_frames = sum(plan["total_sampled_frames"] for plan in task_plans.values())

    run_config_rows = [{
        "frames_root": args.frames_root,
        "models": "|".join(model_keys),
        "categories": "|".join(args.categories),
        "suffixes": "all" if suffixes is None else "|".join(suffixes),
        "win_sec": args.win_sec,
        "max_windows": args.max_windows,
        "eval_fps": args.eval_fps,
        "pair_stride": args.pair_stride,
        "covis_thresh": args.covis_thresh,
        "conf_percentile": args.conf_percentile,
        "conf_min": args.conf_min,
        "tau_z": args.tau_z,
        "ufm_longside": args.ufm_longside,
        "output_dir": os.path.abspath(args.output_dir),
        "save_frame_details": args.save_frame_details,
        "n_clips": len(tasks),
        "raw_frames": total_raw_frames,
        "sampled_windows": total_windows,
        "sampled_frames": total_sampled_frames,
    }]

    category_summary_rows = []
    for (model_key, cat), bucket in sorted(per_model_category.items()):
        arr = np.array(bucket["scores"], dtype=float)
        valid_count = int(np.sum(np.isfinite(arr).all(axis=1))) if arr.size else 0
        means = tuple(np.nanmean(arr, axis=0)) if arr.size else (float('nan'),) * 3
        category_summary_rows.append({
            "model_key": model_key,
            "category": cat,
            "n_clips": len(bucket["scores"]),
            "n_valid_clips": valid_count,
            "n_windows": bucket["n_windows"],
            "n_frames": bucket["n_frames"],
            "motion_mean": means[0],
            "depth_mean": means[1],
            "fused_mean": means[2],
        })

    write_csv_rows(
        os.path.join(args.output_dir, "01_run_config.csv"),
        [
            "frames_root", "models", "categories", "suffixes", "win_sec", "max_windows",
            "eval_fps", "pair_stride", "covis_thresh", "conf_percentile", "conf_min",
            "tau_z", "ufm_longside", "output_dir", "save_frame_details", "n_clips",
            "raw_frames", "sampled_windows", "sampled_frames"
        ],
        run_config_rows,
    )
    write_csv_rows(
        os.path.join(args.output_dir, "02_category_summary.csv"),
        [
            "model_key", "category", "n_clips", "n_valid_clips", "n_windows", "n_frames",
            "motion_mean", "depth_mean", "fused_mean"
        ],
        [{
            **row,
            "motion_mean": fmt_score(row["motion_mean"]),
            "depth_mean": fmt_score(row["depth_mean"]),
            "fused_mean": fmt_score(row["fused_mean"]),
        } for row in category_summary_rows],
    )
    write_csv_rows(
        os.path.join(args.output_dir, "03_clip_summary.csv"),
        [
            "model_key", "category", "clip_id", "raw_frames", "sampled_frames",
            "n_windows", "motion", "depth", "fused"
        ],
        [{
            **row,
            "motion": fmt_score(row["motion"]),
            "depth": fmt_score(row["depth"]),
            "fused": fmt_score(row["fused"]),
        } for row in clip_summary_rows],
    )
    write_csv_rows(
        os.path.join(args.output_dir, "04_window_details.csv"),
        [
            "model_key", "category", "clip_id", "window_idx", "window_frames",
            "motion", "depth", "fused"
        ],
        [{
            **row,
            "motion": fmt_score(row["motion"]),
            "depth": fmt_score(row["depth"]),
            "fused": fmt_score(row["fused"]),
        } for row in window_detail_rows],
    )
    if args.save_frame_details:
        write_csv_rows(
            os.path.join(args.output_dir, "05_frame_details.csv"),
            [
                "model_key", "category", "clip_id", "window_idx", "frame_idx_in_window",
                "frame_name", "motion", "depth", "fused"
            ],
            [{
                **row,
                "motion": fmt_score(row["motion"]),
                "depth": fmt_score(row["depth"]),
                "fused": fmt_score(row["fused"]),
            } for row in frame_detail_rows],
        )

    # 7. Print Summary
    
    print("\n")
    print("=" * 80)
    
    print("\n# SUMMARY_HEADER,model_key,category,n_clips,motion_mean,depth_mean,fused_mean")
    def nanmean_triplet(rows):
        arr = np.array(rows, dtype=float)
        if arr.size == 0: return (float('nan'),)*3, 0
        n = int(np.sum(np.isfinite(arr).all(axis=1)))
        return tuple(np.nanmean(arr, axis=0)), n

    for (model_key, cat) in sorted(per_model_category.keys()):
        (m, d, f), n = nanmean_triplet(per_model_category[(model_key, cat)]["scores"])
        print(f"# SUMMARY,{model_key},{cat},{n},{m:.6f},{d:.6f},{f:.6f}")
        print("\n")    
        print("=" * 180)
        print("=" * 180)

    print(f"\nCSV outputs saved to: {os.path.abspath(args.output_dir)}")
    print("  - 00_commande.txt")
    print("  - 01_run_config.csv")
    print("  - 02_category_summary.csv")
    print("  - 03_clip_summary.csv")
    print("  - 04_window_details.csv")
    if args.save_frame_details:
        print("  - 05_frame_details.csv")
    

if __name__ == "__main__":
    main()
