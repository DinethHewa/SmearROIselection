#!/usr/bin/env python3
"""
Build binary-classification datasets from manifest CSVs.

Outputs a .npy file containing a dict with:
  - image_paths: list[str]
  - labels: np.ndarray (N,) or (N, H, W)
  - label_mode: str
  - manifest: str
  - dataset: str

Label modes:
  - defocus-sign: label 1 if defocus >= 0 else 0 (uses defocus column)
  - labels-dir: load per-image .npy labels from a folder (basename match)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


DEFAULT_MANIFESTS = [
    "manifest_bma.csv",
    "manifest_tbf.csv",
    "manifest_pbs.csv",
    "manifest_focus_test.csv",
]


def resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data"


def default_name_from_manifest(path: Path) -> str:
    stem = path.stem
    if stem.startswith("manifest_"):
        return stem[len("manifest_") :]
    return stem


def load_manifest(manifest_path: Path, path_col: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if path_col not in df.columns:
        raise KeyError(f"Manifest missing '{path_col}' column: {manifest_path}")
    return df


def labels_from_defocus(df: pd.DataFrame, defocus_col: str) -> np.ndarray:
    if defocus_col not in df.columns:
        raise KeyError(f"Manifest missing '{defocus_col}' column")
    vals = pd.to_numeric(df[defocus_col], errors="coerce")
    if vals.isna().any():
        bad = int(vals.isna().sum())
        raise ValueError(f"Found {bad} NaN defocus values in '{defocus_col}'")
    return (vals.values >= 0).astype(np.uint8)


def resolve_labels_dir(template: str, dataset_name: str) -> Path:
    return Path(template.format(name=dataset_name)).expanduser().resolve()


def labels_from_dir(
    image_paths: List[str],
    labels_dir: Path,
    strict: bool,
) -> Tuple[List[str], np.ndarray, List[str]]:
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels path is not a directory: {labels_dir}")
    if not any(labels_dir.glob("*.npy")):
        example = Path(image_paths[0]).stem + ".npy" if image_paths else "<image>.npy"
        raise FileNotFoundError(
            f"No .npy labels found in {labels_dir}. Expected files like {example}"
        )
    kept_paths: List[str] = []
    label_paths: List[str] = []
    labels: List[np.ndarray] = []
    missing: List[str] = []

    for img_path in image_paths:
        base = Path(img_path).stem
        label_path = labels_dir / f"{base}.npy"
        if not label_path.is_file():
            missing.append(str(label_path))
            if strict:
                continue
            else:
                continue
        try:
            label_arr = np.load(label_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load label file: {label_path}") from exc
        kept_paths.append(img_path)
        label_paths.append(str(label_path))
        labels.append(label_arr)

    if missing and strict:
        raise FileNotFoundError(
            f"Missing {len(missing)} label files (first 5): {missing[:5]}"
        )

    if not labels:
        raise RuntimeError(
            f"No labels loaded from {labels_dir}. Check filenames match image basenames."
        )

    shapes = {lab.shape for lab in labels}
    if len(shapes) == 1:
        labels_arr = np.stack(labels, axis=0)
    else:
        labels_arr = np.array(labels, dtype=object)

    return kept_paths, labels_arr, label_paths


def build_dataset(
    manifest_path: Path,
    dataset_name: str,
    output_dir: Path,
    path_col: str,
    defocus_col: str,
    label_mode: str,
    labels_dir_template: str | None,
    strict_labels: bool,
) -> Path:
    df = load_manifest(manifest_path, path_col=path_col)
    image_paths = df[path_col].astype(str).tolist()

    if label_mode == "defocus-sign":
        labels = labels_from_defocus(df, defocus_col=defocus_col)
        label_paths = None
    elif label_mode == "labels-dir":
        if not labels_dir_template:
            raise ValueError("labels-dir mode requires --labels-dir")
        labels_dir = resolve_labels_dir(labels_dir_template, dataset_name)
        kept_paths, labels, label_paths = labels_from_dir(
            image_paths=image_paths,
            labels_dir=labels_dir,
            strict=strict_labels,
        )
        image_paths = kept_paths
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bin_class_{dataset_name}.npy"

    payload = {
        "image_paths": np.array(image_paths, dtype=object),
        "labels": labels,
        "label_mode": label_mode,
        "manifest": str(manifest_path),
        "dataset": dataset_name,
    }
    if label_paths is not None:
        payload["label_paths"] = np.array(label_paths, dtype=object)

    np.save(out_path, payload)
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build binary-classification dataset arrays from a manifest CSV."
    )
    ap.add_argument("--manifest", help="Path to manifest CSV")
    ap.add_argument("--name", help="Dataset name for output file suffix")
    ap.add_argument("--data-dir", default=None, help="Base data directory (default: Regression/data)")
    ap.add_argument("--path-col", default="image_path", help="Column with image paths")
    ap.add_argument("--defocus-col", default="defocus_um", help="Column with defocus values")
    ap.add_argument(
        "--label-mode",
        choices=["defocus-sign", "labels-dir"],
        default="defocus-sign",
        help="How to derive labels",
    )
    ap.add_argument(
        "--labels-dir",
        default=None,
        help="Folder of .npy labels; supports {name} template (expects <image_stem>.npy)",
    )
    ap.add_argument(
        "--strict-labels",
        action="store_true",
        help="Fail if any label file is missing (labels-dir mode)",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Process the 4 default manifests in Regression/data",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    if args.auto:
        for filename in DEFAULT_MANIFESTS:
            manifest_path = data_dir / filename
            if not manifest_path.is_file():
                print(f"[WARN] Missing manifest: {manifest_path}")
                continue
            dataset_name = default_name_from_manifest(manifest_path)
            out_path = build_dataset(
                manifest_path=manifest_path,
                dataset_name=dataset_name,
                output_dir=data_dir,
                path_col=args.path_col,
                defocus_col=args.defocus_col,
                label_mode=args.label_mode,
                labels_dir_template=args.labels_dir,
                strict_labels=args.strict_labels,
            )
            print(f"[OK] Wrote {out_path}")
        return

    if not args.manifest or not args.name:
        raise SystemExit("Provide --manifest and --name (or use --auto)")

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_path = build_dataset(
        manifest_path=manifest_path,
        dataset_name=args.name,
        output_dir=data_dir,
        path_col=args.path_col,
        defocus_col=args.defocus_col,
        label_mode=args.label_mode,
        labels_dir_template=args.labels_dir,
        strict_labels=args.strict_labels,
    )
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
