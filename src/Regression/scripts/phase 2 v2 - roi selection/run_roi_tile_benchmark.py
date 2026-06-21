#!/usr/bin/env python3
"""
End-to-end ROI tile binary benchmark using focus_binary_benchmark.

Builds tile images + manifest from per-image 10x10 labels, splits by stack,
then runs focus_binary_benchmark tuning + baselines + compare_best.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image


GRID_ROWS = 10
GRID_COLS = 10
TILE_SIZE = 200
CANVAS_SIZE = GRID_ROWS * TILE_SIZE

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MANIFEST_FIELDS = [
    "dataset",
    "image_path",
    "label",
    "stack_id",
    "patient_id",
    "source",
    "source_image",
    "tile_row",
    "tile_col",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ROI tile binary benchmark.")
    parser.add_argument(
        "--data-root",
        default="/home/dineth/focus_measure/datasets/New folder/bma pbf tfa",
        help="Root containing <dataset>_imgs and <dataset>_labels folders.",
    )
    parser.add_argument("--datasets", default="bma,pbs,tbf", help="Comma-separated dataset names.")
    parser.add_argument(
        "--out-root",
        default="/home/dineth/focus_measure/journal/Regression/data/roi_tile_benchmark",
        help="Output root for tiles, manifests, runs, and reports.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.6)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", default=True, help="Resume if outputs exist (default).")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume behavior.")
    parser.add_argument("--limit-images", type=int, default=None, help="Limit images per dataset (debug).")
    parser.add_argument("--families", default="cnn,cnn_attention,transfer,vit,hybrid_vit,focus_dnn,cnn_focus_hybrid")
    parser.add_argument("--tuner", default=None, choices=["hyperband", "bayesian", "random"])
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--backbone-set", default=None, choices=["light", "all"])
    parser.add_argument("--light-mode", action="store_true")
    parser.add_argument(
        "--leakage-check",
        action="store_true",
        help="Enable hash-based leakage checks (disabled by default for tile data).",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("roi_tile_benchmark")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers = [fh, sh]
    return logger


def find_focus_binary_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        candidate = parent / "focus_binary_benchmark"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("focus_binary_benchmark directory not found from script location.")


def ensure_focus_binary_importable(fb_root: Path) -> None:
    src_path = fb_root / "src"
    if not src_path.exists():
        return
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if src_str not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([src_str] + parts)


def load_image_rgb_uint8(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.array(img)
        if arr.size:
            max_val = float(arr.max())
            if max_val > 0:
                arr = arr.astype(np.float32) / max_val * 255.0
            else:
                arr = arr.astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = img.convert("RGB")
    img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.BILINEAR)
    return np.array(img)


def list_stack_dirs(images_root: Path) -> List[Path]:
    return sorted([p for p in images_root.iterdir() if p.is_dir()])


def list_stack_images(stack_dir: Path) -> List[Path]:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(stack_dir.rglob(f"*{ext}"))
        images.extend(stack_dir.rglob(f"*{ext.upper()}"))
    return sorted({p for p in images if p.is_file()})


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.npy"


def load_label_matrix(label_path: Path) -> np.ndarray | None:
    if not label_path.is_file():
        return None
    try:
        arr = np.load(label_path)
    except Exception:
        return None
    if arr.shape != (GRID_ROWS, GRID_COLS):
        return None
    return (arr > 0).astype(np.uint8)


def load_processed_images(progress_path: Path, manifest_path: Path) -> Set[str]:
    processed: Set[str] = set()
    if progress_path.is_file():
        processed.update(p.strip() for p in progress_path.read_text().splitlines() if p.strip())
        return processed
    if manifest_path.is_file():
        with manifest_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "source_image" in reader.fieldnames:
                for row in reader:
                    src = (row.get("source_image") or "").strip()
                    if src:
                        processed.add(src)
    return processed


def append_processed(progress_path: Path, image_path: str) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(image_path + "\n")


def build_tiles_and_manifest(
    dataset: str,
    images_root: Path,
    labels_root: Path,
    tiles_root: Path,
    manifest_path: Path,
    progress_path: Path,
    limit_images: int | None,
    resume: bool,
    logger: logging.Logger,
) -> None:
    images_dir = images_root / f"{dataset}_imgs"
    labels_dir = labels_root / f"{dataset}_labels"
    if not images_dir.is_dir():
        logger.warning("missing images dir: %s", images_dir)
        return
    if not labels_dir.is_dir():
        logger.warning("missing labels dir: %s", labels_dir)
        return

    processed = load_processed_images(progress_path, manifest_path) if resume else set()
    manifest_exists = manifest_path.is_file()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tiles_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if not manifest_exists:
            writer.writeheader()

        stack_dirs = list_stack_dirs(images_dir)
        image_count = 0
        for stack_dir in stack_dirs:
            stack_id = stack_dir.name
            image_paths = list_stack_images(stack_dir)
            for image_path in image_paths:
                image_path_str = str(image_path)
                if resume and image_path_str in processed:
                    continue
                label_path = label_path_for(image_path, labels_dir)
                labels = load_label_matrix(label_path)
                if labels is None:
                    logger.warning("missing/invalid label: %s", label_path)
                    continue
                try:
                    img_arr = load_image_rgb_uint8(image_path_str)
                except Exception as exc:
                    logger.warning("failed to load %s: %s", image_path_str, exc)
                    continue

                dataset_tiles_dir = tiles_root / dataset / stack_id
                dataset_tiles_dir.mkdir(parents=True, exist_ok=True)
                image_stem = image_path.stem

                for r in range(GRID_ROWS):
                    for c in range(GRID_COLS):
                        y0 = r * TILE_SIZE
                        y1 = y0 + TILE_SIZE
                        x0 = c * TILE_SIZE
                        x1 = x0 + TILE_SIZE
                        tile = img_arr[y0:y1, x0:x1]
                        tile_name = f"{image_stem}_r{r:02d}_c{c:02d}.png"
                        tile_path = dataset_tiles_dir / tile_name
                        if not tile_path.is_file():
                            Image.fromarray(tile).save(tile_path)
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "image_path": str(tile_path),
                                "label": int(labels[r, c]),
                                "stack_id": stack_id,
                                "patient_id": stack_id,
                                "source": "roi_tile_labels",
                                "source_image": image_path_str,
                                "tile_row": r,
                                "tile_col": c,
                            }
                        )

                append_processed(progress_path, image_path_str)
                processed.add(image_path_str)
                image_count += 1
                if limit_images and image_count >= limit_images:
                    logger.info("limit reached for %s: %s images", dataset, image_count)
                    return
            logger.info("processed stack %s (%s images)", stack_id, len(image_paths))


def write_manifest_with_splits(
    manifest_path: Path,
    manifest_with_splits: Path,
    seed: int,
    train: float,
    val: float,
    test: float,
    logger: logging.Logger,
) -> None:
    from focus_binary.data.splits import split_manifest, assert_no_leak

    df = pd.read_csv(manifest_path)
    if df.empty:
        raise ValueError("Tile manifest is empty; cannot split.")
    split_df = split_manifest(
        df,
        seed=seed,
        train=train,
        val=val,
        test=test,
        group_col="stack_id",
        stratify_col="label",
        by_dataset=True,
    )
    assert_no_leak(split_df, group_col="stack_id", split_col="split")
    manifest_with_splits.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(manifest_with_splits, index=False)
    logger.info("wrote manifest_with_splits: %s", manifest_with_splits)


def run_cmd(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    logger: logging.Logger,
) -> int:
    logger.info("running: %s", " ".join(cmd))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            sys.stdout.write(line)
    return proc.wait()


def write_tuning_config(config_path: Path, leakage_check: bool) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"leakage_check: {'true' if leakage_check else 'false'}",
        f"leakage_sha1: {'true' if leakage_check else 'false'}",
        f"leakage_phash: {'true' if leakage_check else 'false'}",
    ]
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def resolve_tuner(requested: str | None, logger: logging.Logger) -> str:
    if requested and requested != "hyperband":
        return requested
    try:
        import keras_tuner as kt
    except Exception:
        if requested == "hyperband":
            logger.warning("keras_tuner not available; falling back to random tuner")
        return "random"
    has_hyperband = hasattr(kt, "oracles") and hasattr(kt.oracles, "Hyperband")
    if requested == "hyperband" and not has_hyperband:
        logger.warning("Hyperband not available in keras_tuner; falling back to random tuner")
        return "random"
    if requested:
        return requested
    return "hyperband" if has_hyperband else "random"


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    tiles_root = out_root / "tiles"
    manifest_path = out_root / "manifests" / "tiles_manifest.csv"
    manifest_with_splits = out_root / "manifests" / "tiles_manifest_with_splits.csv"
    progress_path = out_root / "progress" / "processed_images.txt"
    runs_dir = out_root / "runs"
    reports_dir = out_root / "reports" / "final"
    log_path = out_root / "logs" / "pipeline.log"

    logger = setup_logging(log_path)
    logger.info("starting ROI tile benchmark")
    os.environ.setdefault("FOCUS_BINARY_DISABLE_CACHE", "1")

    fb_root = find_focus_binary_root(Path(__file__).resolve())
    ensure_focus_binary_importable(fb_root)

    data_root = Path(args.data_root).expanduser().resolve()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tuning_config = write_tuning_config(out_root / "configs" / "tuning_override.yaml", args.leakage_check)
    tuner_choice = resolve_tuner(args.tuner, logger)

    if not (args.resume and manifest_with_splits.is_file()):
        for dataset in datasets:
            build_tiles_and_manifest(
                dataset=dataset,
                images_root=data_root,
                labels_root=data_root,
                tiles_root=tiles_root,
                manifest_path=manifest_path,
                progress_path=progress_path,
                limit_images=args.limit_images,
                resume=args.resume,
                logger=logger,
            )
        write_manifest_with_splits(
            manifest_path=manifest_path,
            manifest_with_splits=manifest_with_splits,
            seed=args.seed,
            train=args.train,
            val=args.val,
            test=args.test,
            logger=logger,
        )
    else:
        logger.info("manifest_with_splits exists, skipping tile build/split")

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for family in families:
        family_dir = runs_dir / family
        best_model = family_dir / "best_model.keras"
        if args.resume and best_model.exists():
            logger.info("skip %s (best_model exists)", family)
            continue
        cmd = [
            sys.executable,
            "-m",
            "focus_binary.scripts.tune_family",
            "--family",
            family,
            "--manifest",
            str(manifest_with_splits),
            "--out",
            str(runs_dir),
            "--seed",
            str(args.seed),
            "--config",
            str(tuning_config),
        ]
        if args.resume:
            cmd += ["--resume"]
        else:
            cmd += ["--no-resume"]
        if tuner_choice:
            cmd += ["--tuner", tuner_choice]
        if args.max_trials is not None:
            cmd += ["--max-trials", str(args.max_trials)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.input_size is not None:
            cmd += ["--input-size", str(args.input_size)]
        if args.backbone_set is not None:
            cmd += ["--backbone-set", args.backbone_set]
        if args.light_mode:
            cmd += ["--light-mode"]
        code = run_cmd(cmd, cwd=fb_root, log_path=log_path, logger=logger)
        if code != 0:
            raise RuntimeError(f"tune_family failed for {family} with code {code}")

    classical_dir = runs_dir / "classical_ml"
    if not (args.resume and (classical_dir / "metrics.csv").exists()):
        cmd = [
            sys.executable,
            "-m",
            "focus_binary.scripts.run_classical_ml",
            "--manifest",
            str(manifest_with_splits),
            "--out-dir",
            str(classical_dir),
            "--seed",
            str(args.seed),
        ]
        if args.input_size is not None:
            cmd += ["--input-size", str(args.input_size)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        code = run_cmd(cmd, cwd=fb_root, log_path=log_path, logger=logger)
        if code != 0:
            raise RuntimeError(f"classical_ml failed with code {code}")
    else:
        logger.info("skip classical_ml (metrics exists)")

    threshold_dir = runs_dir / "threshold_baselines" / "latest"
    if not (args.resume and (threshold_dir / "metrics.csv").exists()):
        cmd = [
            sys.executable,
            "-m",
            "focus_binary.scripts.run_threshold_baselines",
            "--manifest",
            str(manifest_with_splits),
            "--out-dir",
            str(threshold_dir),
            "--seed",
            str(args.seed),
        ]
        if args.input_size is not None:
            cmd += ["--input-size", str(args.input_size)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        code = run_cmd(cmd, cwd=fb_root, log_path=log_path, logger=logger)
        if code != 0:
            raise RuntimeError(f"threshold baselines failed with code {code}")
    else:
        logger.info("skip threshold baselines (metrics exists)")

    reports_dir.mkdir(parents=True, exist_ok=True)
    compare_cmd = [
        sys.executable,
        "-m",
        "focus_binary.scripts.compare_best",
        "--manifest",
        str(manifest_with_splits),
        "--runs-dir",
        str(runs_dir),
        "--out-dir",
        str(reports_dir),
    ]
    code = run_cmd(compare_cmd, cwd=fb_root, log_path=log_path, logger=logger)
    if code != 0:
        raise RuntimeError(f"compare_best failed with code {code}")

    logger.info("pipeline complete")


if __name__ == "__main__":
    main()
