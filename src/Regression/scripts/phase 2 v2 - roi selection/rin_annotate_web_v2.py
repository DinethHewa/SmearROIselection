# Requires: pip install streamlit opencv-python pillow numpy

import argparse
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
    parser.add_argument("--labels-dir", default=None)
    args, _ = parser.parse_known_args()
    return args


def normalize_dir(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))


def list_stack_dirs(image_dir: str) -> List[Path]:
    base = Path(image_dir)
    return sorted([p for p in base.iterdir() if p.is_dir()])


def list_stack_images(stack_dir: Path) -> List[str]:
    return sorted(
        str(p)
        for p in stack_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_load_image(path: str) -> Tuple[Image.Image | None, str | None]:
    try:
        img = Image.open(path)
    except Exception as exc:
        return None, str(exc)
    try:
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
            img = Image.fromarray(arr, mode="RGB")
        else:
            img = img.convert("RGB")
    except Exception as exc:
        return None, str(exc)
    img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.BILINEAR)
    return img, None


def draw_grid_with_labels(image: Image.Image, labels: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if labels[r, c] == 1:
                x0 = c * TILE_SIZE
                y0 = r * TILE_SIZE
                x1 = x0 + TILE_SIZE
                y1 = y0 + TILE_SIZE
                draw_overlay.rectangle([x0, y0, x1, y1], fill=(0, 255, 0, 60))
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


def load_stack_labels(image_paths: List[str], labels_dir: str) -> Tuple[np.ndarray, List[str]]:
    warnings: List[str] = []
    labels = None
    for img_path in image_paths:
        label_path = label_path_for(img_path, labels_dir)
        if not os.path.isfile(label_path):
            continue
        try:
            arr = np.load(label_path)
        except Exception as exc:
            warnings.append(f"Failed to load {label_path}: {exc}")
            continue
        if arr.shape != (GRID_ROWS, GRID_COLS):
            warnings.append(
                f"Label shape mismatch in {label_path}; expected 10x10, got {arr.shape}."
            )
            continue
        arr = (arr > 0).astype(np.uint8)
        if labels is None:
            labels = arr
        elif not np.array_equal(labels, arr):
            warnings.append(
                f"Label mismatch across stack; using {os.path.basename(label_path)}."
            )
    if labels is None:
        labels = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
    return labels, warnings


def save_stack_labels(image_paths: List[str], labels_dir: str, labels: np.ndarray) -> None:
    labels = (labels > 0).astype(np.uint8)
    for img_path in image_paths:
        out_path = label_path_for(img_path, labels_dir)
        np.save(out_path, labels)


def get_stack_state(stack_name: str, image_paths: List[str], labels_dir: str) -> Tuple[np.ndarray, List[str]]:
    labels_map = st.session_state.setdefault("labels_map", {})
    if stack_name in labels_map:
        return labels_map[stack_name], []
    labels, warnings = load_stack_labels(image_paths, labels_dir)
    labels_map[stack_name] = labels
    return labels, warnings


def choose_rep_image(image_paths: List[str]) -> str:
    """
    Choose a deterministic representative image for a stack.
    Prefer filenames containing '_page_1' in the stem; otherwise use the first sorted path.
    """
    if not image_paths:
        return ""
    for p in image_paths:
        try:
            if "_page_1" in Path(p).stem:
                return p
        except Exception:
            continue
    return image_paths[0]


def load_representative_image(image_paths: List[str]) -> Tuple[Image.Image | None, str | None, str | None]:
    if not image_paths:
        return None, None, "No images found."
    img_path = choose_rep_image(image_paths)
    if not img_path:
        return None, None, "No images found."
    img, err = safe_load_image(img_path)
    if err is None:
        return img, img_path, None
    return None, None, f"{os.path.basename(img_path)}: {err}"


def rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def main() -> None:
    st.set_page_config(page_title="RIN Stack Annotator (Web)", layout="wide")
    args = parse_args()
    image_dir = normalize_dir(args.image_dir)
    labels_dir = normalize_dir(args.labels_dir) or os.path.abspath("labels")

    st.title("RIN Stack Annotation Tool (Web)")
    st.write("Click cells to mark 1. Labels are saved as 10x10 uint8 .npy files.")

    if not image_dir or not os.path.isdir(image_dir):
        st.error("Provide a valid --image-dir (root folder with stack subfolders).")
        st.stop()

    os.makedirs(labels_dir, exist_ok=True)

    stack_dirs = list_stack_dirs(image_dir)
    if not stack_dirs:
        st.error("No stack folders found in image directory.")
        st.stop()

    stacks = []
    for stack_dir in stack_dirs:
        images = list_stack_images(stack_dir)
        if images:
            stacks.append((stack_dir.name, images))

    if not stacks:
        st.error("No images found in any stack folders.")
        st.stop()

    skip_stacks = st.session_state.setdefault("skip_stacks", set())
    skip_messages = st.session_state.setdefault("skip_messages", [])

    available = [(name, imgs) for name, imgs in stacks if name not in skip_stacks]
    if not available:
        st.error("All stacks were skipped due to load errors.")
        st.stop()

    if "stack_index" not in st.session_state:
        st.session_state.stack_index = 0

    stack_names = [name for name, _ in available]
    current_idx = min(st.session_state.stack_index, len(available) - 1)

    selected_name = st.sidebar.selectbox(
        "Stack",
        options=stack_names,
        index=current_idx,
        key="stack_select",
    )
    current_idx = stack_names.index(selected_name)
    st.session_state.stack_index = current_idx

    stack_name, image_paths = available[current_idx]

    rep_image, rep_path, rep_error = load_representative_image(image_paths)
    if rep_image is None:
        skip_stacks.add(stack_name)
        skip_messages.append(f"Skipped stack {stack_name}: {rep_error}")
        rerun()

    labels, warnings = get_stack_state(stack_name, image_paths, labels_dir)
    for msg in warnings:
        st.warning(msg)
    if skip_messages:
        st.warning(skip_messages[-1])

    st.sidebar.header("Session")
    st.sidebar.write(f"Image dir: {image_dir}")
    st.sidebar.write(f"Labels dir: {labels_dir}")
    st.sidebar.write(f"Stacks: {len(available)}")
    if skip_stacks:
        st.sidebar.write(f"Skipped: {len(skip_stacks)}")

    nav_cols = st.columns([1, 1, 2, 1, 1])
    with nav_cols[0]:
        if st.button("Prev stack"):
            st.session_state.stack_index = max(0, current_idx - 1)
            rerun()
    with nav_cols[1]:
        if st.button("Next stack"):
            st.session_state.stack_index = min(len(available) - 1, current_idx + 1)
            rerun()
    with nav_cols[2]:
        st.write(f"Stack {current_idx + 1} / {len(available)}")
        st.write(f"{stack_name} ({len(image_paths)} images)")
        if rep_path:
            st.write(f"Example: {os.path.basename(rep_path)}")

    col_img, col_ctrl = st.columns([3, 2])

    with col_ctrl:
        st.subheader("Tile Grid (click to mark 1)")
        new_labels = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        for r in range(GRID_ROWS):
            cols = st.columns(GRID_COLS)
            for c in range(GRID_COLS):
                key = f"cell_{stack_name}_{r}_{c}"
                val = cols[c].checkbox("", value=bool(labels[r, c]), key=key)
                new_labels[r, c] = 1 if val else 0

        if not np.array_equal(new_labels, labels):
            labels[:] = new_labels
            st.session_state.labels_map[stack_name] = labels
            save_stack_labels(image_paths, labels_dir, labels)
            st.success("Saved labels for this stack.")

        ones_count = int(labels.sum())
        st.progress(ones_count / float(GRID_ROWS * GRID_COLS))
        st.write(f"Tiles set to 1: {ones_count} / {GRID_ROWS * GRID_COLS}")

        if st.button("Clear stack labels"):
            labels[:, :] = 0
            st.session_state.labels_map[stack_name] = labels
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    st.session_state[f"cell_{stack_name}_{r}_{c}"] = False
            save_stack_labels(image_paths, labels_dir, labels)
            rerun()

    with col_img:
        grid_img = draw_grid_with_labels(rep_image, labels)
        st.image(grid_img, use_column_width=True)


if __name__ == "__main__":
    main()
