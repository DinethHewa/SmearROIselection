#!/usr/bin/env python3
"""
CNN-guided ROI selection:
  - Use a trained CNN to estimate how many tiles to keep (k).
  - Use roi_selection_v2 scoring to rank tiles and select top-k.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def _load_roi_selection_v2():
    # Import the legacy ROI scorer from the previous phase.
    legacy_dir = Path(__file__).resolve().parents[1] / "phase 2 - roi selection"
    if not legacy_dir.is_dir():
        raise FileNotFoundError(f"roi_selection_v2 not found at {legacy_dir}")
    sys.path.insert(0, str(legacy_dir))
    from roi_selection_v2 import (  # type: ignore
        compute_rin_scores_for_image,
        compute_R10,
        compute_Neff_norm,
        D_MIN_NON_DENSE,
        load_image_rgb,
        visualize_tiles,
    )
    return {
        "compute_rin_scores_for_image": compute_rin_scores_for_image,
        "compute_R10": compute_R10,
        "compute_Neff_norm": compute_Neff_norm,
        "D_MIN_NON_DENSE": D_MIN_NON_DENSE,
        "load_image_rgb": load_image_rgb,
        "visualize_tiles": visualize_tiles,
    }


_ROI = _load_roi_selection_v2()
compute_rin_scores_for_image = _ROI["compute_rin_scores_for_image"]
compute_R10 = _ROI["compute_R10"]
compute_Neff_norm = _ROI["compute_Neff_norm"]
D_MIN_NON_DENSE = _ROI["D_MIN_NON_DENSE"]
load_image_rgb = _ROI["load_image_rgb"]
visualize_tiles = _ROI["visualize_tiles"]


DENSE_RATIO_MIN = 0.27
SPARSE_RATIO_MIN = 0.10


def resolve_default_model_path() -> Path:
    # Default CNN model trained by the tile benchmark pipeline.
    base = Path(__file__).resolve().parents[2]
    return base / "data" / "roi_tile_benchmark" / "runs" / "cnn" / "best_model.keras"


def resolve_tbf_model_path() -> Path:
    # TBF-specific CNN model (best trial).
    base = Path(__file__).resolve().parents[2]
    return base / "data" / "roi_tile_benchmark" / "tbf" / "runs" / "cnn" / "kt" / "trial_02" / "best_model.keras"


def parse_args() -> argparse.Namespace:
    # CLI for manifests, calibration, model, and output paths.
    ap = argparse.ArgumentParser(description="CNN-guided ROI selection (top-k via roi_selection_v2)")
    ap.add_argument("--manifests", nargs="+", required=True, help="Manifest CSV(s)")
    ap.add_argument("--path-col", default="image_path", help="Column name for image path")
    ap.add_argument("--dataset-col", default="dataset", help="Optional dataset column")
    ap.add_argument("--calibration-csv", required=True, help="Calibration CSV (from calibration_script.py)")
    ap.add_argument("--model-path", default=str(resolve_default_model_path()), help="Path to best CNN model (.keras)")
    ap.add_argument("--tbf-model-path", default=str(resolve_tbf_model_path()), help="TBF-only CNN model (.keras)")
    ap.add_argument("--prob-threshold", type=float, default=0.5, help="CNN probability threshold for label=1")
    ap.add_argument("--input-size", type=int, default=None, help="Override model input size")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for CNN inference")
    ap.add_argument("--min-k", type=int, default=1, help="Minimum k to select (use 0 to allow empty)")
    ap.add_argument("--max-k-ratio", type=float, default=None, help="Optional max k as ratio of tiles")
    ap.add_argument("--use-cnn-mask", dest="use_cnn_mask", action="store_true", help="Restrict selection to CNN-positive tiles")
    ap.add_argument("--no-use-cnn-mask", dest="use_cnn_mask", action="store_false", help="Allow selection outside CNN-positive tiles")
    ap.set_defaults(use_cnn_mask=True)
    ap.add_argument("--viz-dir", default=None, help="Save per-image visualization PNGs")
    ap.add_argument("--tiles-dir", default=None, help="Save per-image *_tiles.npy")
    ap.add_argument("--diag-csv", default=None, help="Optional per-image diagnostics CSV")
    ap.add_argument("--limit-images", type=int, default=None, help="Limit images per manifest")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip if tiles already exist (default).")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume behavior.")
    return ap.parse_args()


def load_calibration(calibration_csv: str | Path) -> Dict[str, float]:
    # Load calibration stats produced by calibration_script.py.
    calib_df = pd.read_csv(calibration_csv)
    if calib_df.empty:
        raise RuntimeError("Calibration CSV is empty.")
    calib_row = calib_df.iloc[0].to_dict()
    calib: Dict[str, float] = {}
    for key, val in calib_row.items():
        try:
            calib[key] = float(val)
        except (TypeError, ValueError):
            continue
    return calib


def load_model(model_path: Path):
    # Load the trained CNN for tile classification.
    if tf is None:
        raise ImportError("TensorFlow is required for CNN inference.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def infer_model_shape(model, input_size: int | None) -> Tuple[int, int, int]:
    # Infer input tensor shape (or override via --input-size).
    if input_size is not None:
        return input_size, input_size, 3
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) != 4:
        return 224, 224, 3
    h = shape[1] or 224
    w = shape[2] or 224
    c = shape[3] or 3
    return int(h), int(w), int(c)


def prepare_tiles_for_model(tiles: np.ndarray, target_hw: Tuple[int, int], channels: int) -> np.ndarray:
    # Resize tiles and match channel count to the CNN input.
    arr = tiles.astype(np.float32)
    if channels == 1 and arr.shape[-1] == 3:
        arr = np.mean(arr, axis=-1, keepdims=True)
    elif channels == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    h, w = target_hw
    if arr.shape[1] != h or arr.shape[2] != w:
        arr = tf.image.resize(arr, (h, w)).numpy()
    return arr.astype(np.float32)


def predict_tile_probs(
    model,
    tiles: np.ndarray,
    target_hw: Tuple[int, int],
    channels: int,
    batch_size: int,
) -> np.ndarray:
    # Run CNN inference and return per-tile probabilities.
    if tiles.size == 0:
        return np.zeros((0,), dtype=np.float32)
    arr = prepare_tiles_for_model(tiles, target_hw=target_hw, channels=channels)
    preds = model.predict(arr, batch_size=batch_size, verbose=0)
    preds = np.asarray(preds).reshape(-1)
    return preds.astype(np.float32)


def label_from_cnn_ratio(ratio: float) -> str:
    # Map CNN-positive ratio to dense/sparse/extremely_sparse.
    if ratio >= DENSE_RATIO_MIN:
        return "dense"
    if ratio >= SPARSE_RATIO_MIN:
        return "sparse"
    return "extremely_sparse"


def select_topk_from_scores(
    scores: np.ndarray,
    grid_h: int,
    grid_w: int,
    k: int,
    sparsity_label: str,
) -> Tuple[List[int], int]:
    # Select top-k tiles, using diversity for non-dense cases.
    if k <= 0:
        return [], 0
    order = np.argsort(-scores)
    if sparsity_label == "dense":
        return [int(i) for i in order[:k]], 0
    from roi_selection_v2 import select_topk_tiles_diverse  # type: ignore
    return select_topk_tiles_diverse(scores, grid_h, grid_w, k=k, d_min=D_MIN_NON_DENSE)


def run_cnn_guided_selection(
    img_path: str,
    calib: Dict[str, float],
    model,
    prob_threshold: float,
    target_hw: Tuple[int, int],
    channels: int,
    batch_size: int,
    min_k: int,
    max_k_ratio: float | None,
    use_cnn_mask: bool,
) -> Dict[str, Any]:
    # 1) Load image and compute RIN scores.
    img_rgb = load_image_rgb(img_path)
    tiles, S, F_hat, G_hat, E_hat, T_hat, gh, gw, calc_label = compute_rin_scores_for_image(img_rgb, calib)
    N = len(S)
    # 2) CNN predicts tile positives -> determines k.
    probs = predict_tile_probs(model, tiles, target_hw, channels, batch_size=batch_size)
    mask = probs >= prob_threshold
    k_raw = int(mask.sum())
    ratio = (k_raw / N) if N else 0.0
    cnn_label = label_from_cnn_ratio(ratio)
    k_used = max(min_k, min(k_raw, N)) if N else 0
    if max_k_ratio is not None and N:
        k_used = min(k_used, int(np.ceil(max_k_ratio * N)))

    # 3) Optionally gate scores by CNN mask.
    scores = S
    if use_cnn_mask:
        k_used = min(k_used, k_raw)
        scores = np.where(mask, S, -np.inf)

    # 4) Pick top-k using the existing ROI selection rules.
    top_indices, d_min_used = select_topk_from_scores(scores, gh, gw, k_used, cnn_label)
    top_indices_arr = np.array(top_indices, dtype=np.int32)

    r10 = compute_R10(S[mask]) if np.any(mask) else 0.0
    neff = compute_Neff_norm(S[mask]) if np.any(mask) else 0.0

    # 5) Return tiles + metadata for visualization/output.
    return {
        "tiles": tiles,
        "scores": S,
        "grid_h": gh,
        "grid_w": gw,
        "top_indices": top_indices_arr,
        "top_tiles": tiles[top_indices_arr] if len(top_indices_arr) else np.empty((0,) + tiles.shape[1:], dtype=np.float32),
        "N_tiles": int(N),
        "K_selected": int(len(top_indices_arr)),
        "cnn_k_raw": int(k_raw),
        "cnn_k_used": int(k_used),
        "cnn_ratio": float(ratio),
        "cnn_threshold": float(prob_threshold),
        "cnn_label": cnn_label,
        "calc_label": calc_label,
        "d_min_used": int(d_min_used),
        "r10_cnn": float(r10),
        "neff_cnn": float(neff),
        "cnn_mask": mask.astype(bool),
    }


def overlay_cnn_mask(
    canvas_rgb: np.ndarray,
    cnn_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    # Draw blue boxes for CNN-positive tiles on the visualization canvas.
    if canvas_rgb is None or cnn_mask is None or len(cnn_mask) == 0:
        return canvas_rgb
    N = len(cnn_mask)
    if grid_h * grid_w != N or grid_h <= 0 or grid_w <= 0:
        grid_w = int(np.ceil(np.sqrt(N)))
        grid_h = int(np.ceil(N / grid_w))
    canvas = canvas_rgb.copy()
    for idx, is_pos in enumerate(cnn_mask):
        if not is_pos:
            continue
        r, c = divmod(idx, grid_w)
        x0 = c * 200
        y0 = r * 200
        x1 = x0 + 200 - 1
        y1 = y0 + 200 - 1
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)
    return canvas


def apply_selected_tint(
    canvas_rgb: np.ndarray,
    selected_indices: np.ndarray,
    grid_h: int,
    grid_w: int,
    tint_rgb: Tuple[float, float, float] = (1.0, 0.6, 0.6),
    alpha: float = 0.35,
) -> np.ndarray:
    # Tint selected tiles on top of the existing visualization.
    if canvas_rgb is None:
        return canvas_rgb
    if selected_indices is None or len(selected_indices) == 0:
        return canvas_rgb
    canvas = canvas_rgb.astype(np.float32)
    if canvas.max() > 1.5:
        canvas = canvas / 255.0
    if grid_h <= 0 or grid_w <= 0 or (grid_h * grid_w) <= 0:
        n = int(max(selected_indices)) + 1
        grid_w = int(np.ceil(np.sqrt(n)))
        grid_h = int(np.ceil(n / grid_w))
    tile_h = canvas.shape[0] // grid_h
    tile_w = canvas.shape[1] // grid_w
    tint = np.asarray(tint_rgb, dtype=np.float32).reshape(1, 1, 3)
    for idx in selected_indices:
        r, c = divmod(int(idx), grid_w)
        y0, y1 = r * tile_h, (r + 1) * tile_h
        x0, x1 = c * tile_w, (c + 1) * tile_w
        patch = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = (1.0 - alpha) * patch + alpha * tint
    return (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)


def main() -> None:
    # Parse CLI and load calibration + CNN model.
    args = parse_args()

    calib = load_calibration(args.calibration_csv)
    default_model_path = Path(args.model_path) if args.model_path else resolve_default_model_path()
    default_model_path = default_model_path.expanduser().resolve()
    tbf_model_path = Path(args.tbf_model_path).expanduser().resolve() if args.tbf_model_path else None
    model_cache: Dict[str, Tuple[Any, Tuple[int, int], int, Path]] = {}
    warned_tbf_missing = False

    def get_model_bundle(dataset_id: str) -> Tuple[Any, Tuple[int, int], int, Path]:
        nonlocal warned_tbf_missing
        use_tbf = dataset_id.lower() == "tbf" and tbf_model_path is not None
        if use_tbf and not tbf_model_path.is_file():
            if not warned_tbf_missing:
                print(f"[WARN] TBF model not found: {tbf_model_path}; falling back to default.")
                warned_tbf_missing = True
            use_tbf = False
        key = "tbf" if use_tbf else "default"
        model_path = tbf_model_path if use_tbf else default_model_path
        if key not in model_cache:
            model = load_model(model_path)
            input_h, input_w, channels = infer_model_shape(model, args.input_size)
            model_cache[key] = (model, (input_h, input_w), channels, model_path)
            print(f"[INFO] Loaded {key} CNN model: {model_path}")
        return model_cache[key]

    # Prepare output directories.
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    tiles_root = args.tiles_dir or args.viz_dir
    if tiles_root:
        os.makedirs(tiles_root, exist_ok=True)

    diag_rows: List[Dict[str, Any]] = []
    total_images = 0
    skipped = 0

    for man_path in args.manifests:
        # Load manifest rows.
        df = pd.read_csv(man_path)
        if args.limit_images is not None:
            df = df.head(args.limit_images)
            print(f"[INFO] Limiting {man_path} to {len(df)} images")

        total_rows = len(df)
        print(f"[INFO] Processing manifest: {man_path} (rows={total_rows})")

        man_base = os.path.splitext(os.path.basename(man_path))[0]
        viz_dir = os.path.join(args.viz_dir, man_base) if args.viz_dir else None
        tiles_dir = os.path.join(tiles_root, man_base) if tiles_root else None
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
        if tiles_dir:
            os.makedirs(tiles_dir, exist_ok=True)

        for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
            # Process each image.
            total_images += 1
            img_path = str(row[args.path_col])
            dataset_id = str(row[args.dataset_col]) if args.dataset_col in row else "unknown"
            print(f"[INFO] ({row_idx}/{total_rows}) {img_path}")
            if not os.path.isfile(img_path):
                print(f"[WARN] Missing image: {img_path}")
                skipped += 1
                continue

            base = os.path.splitext(os.path.basename(img_path))[0]
            tiles_path = os.path.join(tiles_dir, f"{base}_tiles.npy") if tiles_dir else None
            if args.resume and tiles_path and os.path.isfile(tiles_path):
                print(f"[SKIP] {tiles_path} exists")
                skipped += 1
                continue

            # Run CNN-guided selection.
            try:
                model, target_hw, channels, _ = get_model_bundle(dataset_id)
                res = run_cnn_guided_selection(
                    img_path=img_path,
                    calib=calib,
                    model=model,
                    prob_threshold=args.prob_threshold,
                    target_hw=target_hw,
                    channels=channels,
                    batch_size=args.batch_size,
                    min_k=args.min_k,
                    max_k_ratio=args.max_k_ratio,
                    use_cnn_mask=args.use_cnn_mask,
                )
            except Exception as exc:
                print(f"[WARN] Error processing {img_path}: {exc}")
                skipped += 1
                continue

            print(
                "[INFO] Selected {}/{} tiles (cnn_k_raw={}, label={}, ratio={:.3f})".format(
                    res["K_selected"],
                    res["N_tiles"],
                    res["cnn_k_raw"],
                    res["cnn_label"],
                    res["cnn_ratio"],
                )
            )

            # Save selected tiles.
            if tiles_path:
                np.save(tiles_path, res["top_tiles"])
                print(f"[TILES] Saved {tiles_path}")

            if viz_dir:
                # Render visualization with CNN mask + final selection.
                try:
                    img_rgb = load_image_rgb(img_path)
                    viz = visualize_tiles(
                        img_rgb,
                        res["tiles"],
                        res["scores"],
                        res["top_indices"],
                        res["grid_h"],
                        res["grid_w"],
                        sparsity_label=res.get("cnn_label"),
                        sparsity_metric=res.get("r10_cnn"),
                        sparsity_method="R10",
                        r10=res.get("r10_cnn"),
                        neff_norm=res.get("neff_cnn"),
                        k_selected=res.get("K_selected"),
                        sumS=float(res["scores"].sum()),
                    )
                    viz = overlay_cnn_mask(
                        viz,
                        res.get("cnn_mask"),
                        res["grid_h"],
                        res["grid_w"],
                        color=(0, 0, 255),
                        thickness=2,
                    )
                    viz = apply_selected_tint(
                        viz,
                        res["top_indices"],
                        res["grid_h"],
                        res["grid_w"],
                    )
                    out_path = os.path.join(viz_dir, f"{base}_cnn_rin.png")
                    cv2.imwrite(out_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    print(f"[VIZ] Saved {out_path}")
                except Exception as exc:
                    print(f"[WARN] Failed viz for {img_path}: {exc}")

            # Record per-image diagnostics.
            diag_rows.append(
                {
                    "image_path": img_path,
                    "dataset": dataset_id,
                    "N_tiles": res["N_tiles"],
                    "K_selected": res["K_selected"],
                    "cnn_k_raw": res["cnn_k_raw"],
                    "cnn_k_used": res["cnn_k_used"],
                    "cnn_ratio": res["cnn_ratio"],
                    "cnn_threshold": res["cnn_threshold"],
                    "cnn_label": res["cnn_label"],
                    "calc_label": res["calc_label"],
                    "d_min_used": res["d_min_used"],
                    "r10_cnn": res["r10_cnn"],
                    "neff_cnn": res["neff_cnn"],
                }
            )

    if args.diag_csv and diag_rows:
        pd.DataFrame(diag_rows).to_csv(args.diag_csv, index=False)
        print(f"[INFO] Saved diagnostics to {args.diag_csv}")

    if total_images > 0:
        print(f"[INFO] Images processed: {total_images}, skipped: {skipped}, skip rate: {skipped/total_images:.3f}")


if __name__ == "__main__":
    main()
