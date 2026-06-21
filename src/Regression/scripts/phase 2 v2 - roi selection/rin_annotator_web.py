# Requires: pip install streamlit opencv-python pillow numpy

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


GRID_ROWS = 10
GRID_COLS = 10
TILE_SIZE = 200
CANVAS_SIZE = GRID_ROWS * TILE_SIZE

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--path-col", default="image_path")
    parser.add_argument("--labels-dir", default=None)
    args, _ = parser.parse_known_args()
    return args


def normalize_dir(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))


def list_images(image_dir: str) -> List[str]:
    return sorted(
        str(p)
        for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_manifest_images(manifest_path: str, path_col: str) -> List[str]:
    images: List[str] = []
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    base_dir = manifest_file.parent
    with manifest_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or path_col not in reader.fieldnames:
            raise KeyError(f"Manifest missing column '{path_col}': {manifest_path}")
        for row in reader:
            raw_path = (row.get(path_col) or "").strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = base_dir / path
            images.append(str(path))
    return images


def safe_load_image(path: str) -> Tuple[Image.Image | None, str | None]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        return None, str(exc)
    img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.BILINEAR)
    return img, None


def draw_grid(
    image: Image.Image,
    selected_row: int | None = None,
    selected_col: int | None = None,
    selected_label: int | None = None,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    if selected_row is not None and selected_col is not None:
        x0 = selected_col * TILE_SIZE
        y0 = selected_row * TILE_SIZE
        x1 = x0 + TILE_SIZE
        y1 = y0 + TILE_SIZE
        if selected_label == 1:
            fill = (0, 255, 0, 60)
        else:
            fill = (255, 0, 0, 60)
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle([x0, y0, x1, y1], fill=fill)
    composite = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composite)
    for r in range(GRID_ROWS + 1):
        y = r * TILE_SIZE
        draw.line([(0, y), (CANVAS_SIZE, y)], fill=(200, 200, 200, 255), width=1)
    for c in range(GRID_COLS + 1):
        x = c * TILE_SIZE
        draw.line([(x, 0), (x, CANVAS_SIZE)], fill=(200, 200, 200, 255), width=1)
    return composite


def label_path_for(image_path: str, labels_dir: str) -> str:
    stem = Path(image_path).stem
    return str(Path(labels_dir) / f"{stem}.npy")


def load_labels(image_path: str, labels_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    warnings: List[str] = []
    label_path = label_path_for(image_path, labels_dir)
    if os.path.isfile(label_path):
        try:
            labels = np.load(label_path)
        except Exception as exc:
            warnings.append(f"Failed to load labels for {image_path}: {exc}")
            labels = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        else:
            if labels.shape != (GRID_ROWS, GRID_COLS):
                warnings.append(
                    f"Label shape mismatch for {image_path}; expected 10x10, got {labels.shape}."
                )
                labels = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
            labels = (labels > 0).astype(np.uint8)
        visited = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)
    else:
        labels = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        visited = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    return labels, visited, warnings


def save_labels(label_path: str, labels: np.ndarray) -> None:
    labels = (labels > 0).astype(np.uint8)
    np.save(label_path, labels)


def get_state_labels(image_path: str, labels_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels_map = st.session_state.setdefault("labels_map", {})
    visited_map = st.session_state.setdefault("visited_map", {})
    if image_path in labels_map:
        return labels_map[image_path], visited_map[image_path], []
    labels, visited, warnings = load_labels(image_path, labels_dir)
    labels_map[image_path] = labels
    visited_map[image_path] = visited
    return labels, visited, warnings


def rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def main() -> None:
    st.set_page_config(page_title="RIN Annotator (Web)", layout="wide")
    args = parse_args()
    image_dir = normalize_dir(args.image_dir)
    manifest_path = normalize_dir(args.manifest)
    labels_dir = normalize_dir(args.labels_dir) or os.path.abspath("labels")

    st.title("RIN Patch Annotation Tool (Web)")
    st.write("Label tiles with 0/1. Labels are saved as 10x10 uint8 .npy files.")

    os.makedirs(labels_dir, exist_ok=True)

    skip_set = st.session_state.setdefault("skip_set", set())
    skip_messages = st.session_state.setdefault("skip_messages", [])
    try:
        if manifest_path:
            images_all = load_manifest_images(manifest_path, path_col=args.path_col)
        elif image_dir and os.path.isdir(image_dir):
            images_all = list_images(image_dir)
        else:
            st.error("Provide --manifest or a valid --image-dir to begin.")
            st.stop()
    except Exception as exc:
        st.error(f"Failed to load images: {exc}")
        st.stop()

    images = [p for p in images_all if p not in skip_set]
    if not images:
        st.error("No images found (or all images were skipped).")
        st.stop()

    if "image_index" not in st.session_state:
        st.session_state.image_index = 0

    idx = min(st.session_state.image_index, len(images) - 1)

    loaded_img = None
    load_error = None
    attempts = 0
    while attempts < len(images):
        image_path = images[idx]
        loaded_img, load_error = safe_load_image(image_path)
        if load_error is None:
            break
        skip_set.add(image_path)
        skip_messages.append(f"Skipped {os.path.basename(image_path)}: {load_error}")
        attempts += 1
        idx = (idx + 1) % len(images)
        if attempts >= len(images):
            loaded_img = None
            break

    if loaded_img is None:
        st.error("No loadable images found.")
        st.stop()

    st.session_state.image_index = idx

    labels, visited, warnings = get_state_labels(image_path, labels_dir)
    for msg in warnings:
        st.warning(msg)
    if skip_messages:
        st.warning(skip_messages[-1])

    st.sidebar.header("Session")
    if manifest_path:
        st.sidebar.write(f"Manifest: {manifest_path}")
        st.sidebar.write(f"Path col: {args.path_col}")
    else:
        st.sidebar.write(f"Image dir: {image_dir}")
    st.sidebar.write(f"Labels dir: {labels_dir}")
    st.sidebar.write(f"Images: {len(images)}")
    if skip_set:
        st.sidebar.write(f"Skipped: {len(skip_set)}")

    nav_cols = st.columns([1, 1, 2, 1, 1])
    with nav_cols[0]:
        if st.button("Prev"):
            st.session_state.image_index = max(0, idx - 1)
            rerun()
    with nav_cols[1]:
        if st.button("Next"):
            st.session_state.image_index = min(len(images) - 1, idx + 1)
            rerun()
    with nav_cols[2]:
        st.write(f"Image {idx + 1} / {len(images)}")
        st.write(os.path.basename(image_path))

    col_img, col_ctrl = st.columns([3, 2])

    with col_ctrl:
        st.subheader("Tile Labeling")
        row = st.number_input("Row (0-9)", min_value=0, max_value=GRID_ROWS - 1, value=0, step=1)
        col = st.number_input("Col (0-9)", min_value=0, max_value=GRID_COLS - 1, value=0, step=1)
        current_val = int(labels[row, col])
        label_key = f"label_value_{row}_{col}"
        label_val = st.radio(
            "Label",
            options=[0, 1],
            index=current_val,
            horizontal=True,
            key=label_key,
        )
        if st.button("Apply label"):
            labels[row, col] = int(label_val)
            visited[row, col] = True
            save_labels(label_path_for(image_path, labels_dir), labels)
            st.success("Saved label.")

        visited_count = int(visited.sum())
        ones_count = int(labels.sum())
        st.progress(visited_count / float(GRID_ROWS * GRID_COLS))
        st.write(f"Tiles labeled this session: {visited_count} / {GRID_ROWS * GRID_COLS}")
        st.write(f"Tiles set to 1: {ones_count} / {GRID_ROWS * GRID_COLS}")

    with col_img:
        grid_img = draw_grid(loaded_img, row, col, int(labels[row, col]))
        st.image(grid_img, use_column_width=True)


if __name__ == "__main__":
    main()
