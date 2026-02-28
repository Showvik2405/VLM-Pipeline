import os
import re
import cv2
import numpy as np
import torch
import ollama
from dataclasses import dataclass
from typing import Tuple, Optional

from ultralytics import SAM  # ✅ important


# =========================
# CONFIG
# =========================
@dataclass
class Config:
    sam3_model_path: str = "sam3.pt"
    vlm_model: str = "qwen3-vl:latest"
    mask_threshold: float = 0.5

    overlay_dir: str = "sam_stream_overlay"
    overlay_name: str = "overlay.jpg"

    final_dir: str = "final_outputs"
    final_name: str = "final.jpg"

    # optional (recommended)
    max_masks: int = 80
    min_area: int = 400


# =========================
# Module A: SAM3 (BLIND) via SAM loader
# =========================
def segment_everything_sam(model: SAM, image_path: str):
    """
    Blind segmentation (segment-all) through the high-level SAM loader.
    This avoids Predictor.model=None issues.
    """
    results = model(image_path)  # returns list-like Results
    return results


# =========================
# Helper: prune masks
# =========================
def prune_masks(results, min_area: int, max_masks: int, thr: float):
    masks = results[0].masks.data  # torch (N,H,W)
    if masks is None or masks.numel() == 0:
        return results

    m_bin = (masks > thr).to(torch.uint8)
    areas = m_bin.flatten(1).sum(1)

    keep = torch.where(areas >= int(min_area))[0]
    if keep.numel() == 0:
        return results

    keep = keep[torch.argsort(areas[keep], descending=True)]
    keep = keep[: int(max_masks)]
    results[0].masks.data = masks[keep]
    return results


# =========================
# Module B: overlay IDs
# =========================
def save_overlay(image_path: str, results, cfg: Config) -> str:
    os.makedirs(cfg.overlay_dir, exist_ok=True)
    out_path = os.path.join(cfg.overlay_dir, cfg.overlay_name)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    masks = results[0].masks.data
    overlay = img.copy()

    for i in range(masks.shape[0]):
        m = masks[i].detach().cpu().numpy()
        m_bin = (m > cfg.mask_threshold).astype(np.uint8)

        overlay[m_bin == 1] = (
            0.6 * overlay[m_bin == 1] + 0.4 * np.array([0, 255, 255])
        ).astype(np.uint8)

        ys, xs = np.where(m_bin == 1)
        if xs.size == 0:
            continue

        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            overlay, f"ID:{i}",
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2, cv2.LINE_AA
        )

    cv2.imwrite(out_path, overlay)
    return out_path


# =========================
# Module C: VLM chooses ID
# =========================
def vlm_choose_id(cfg: Config, image_path: str, overlay_path: str, target_text: str) -> Tuple[int, str]:
    prompt = (
        "You will see 2 images:\n"
        "1) original image\n"
        "2) segmentation overlay with IDs\n\n"
        f"Task: Which ID contains the {target_text}?\n"
        "Reply ONLY in this exact format: id=<number>\n"
        "Example: id=2"
    )

    res = ollama.chat(
        model=cfg.vlm_model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path, overlay_path]
        }]
    )

    text = res["message"]["content"].strip()
    m = re.search(r"id\s*=\s*(\d+)", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"VLM reply not parseable. Got: {text}")

    return int(m.group(1)), text


# =========================
# Module D: mask -> centroid + bbox
# =========================
def mask_to_geometry(results, mask_id: int, thr: float) -> Tuple[Tuple[float, float], Tuple[int, int, int, int]]:
    masks = results[0].masks.data
    if mask_id < 0 or mask_id >= masks.shape[0]:
        raise IndexError(f"mask_id {mask_id} out of range. total masks={masks.shape[0]}")

    mask = masks[mask_id]
    m = mask.detach().cpu().numpy()
    m_bin = (m > thr).astype(np.uint8)

    ys, xs = np.where(m_bin == 1)
    if xs.size == 0:
        raise ValueError("Selected mask is empty.")

    cx, cy = float(xs.mean()), float(ys.mean())
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return (cx, cy), (x1, y1, x2, y2)


# =========================
# Module E: final image
# =========================
def save_final(image_path: str, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int], label: str, cfg: Config) -> str:
    os.makedirs(cfg.final_dir, exist_ok=True)
    out_path = os.path.join(cfg.final_dir, cfg.final_name)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x1, y1, x2, y2 = bbox
    cx, cy = int(centroid[0]), int(centroid[1])

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

    text = f"{label}  centroid=({cx},{cy})"
    cv2.putText(
        img, text,
        (max(10, x1), max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 0, 255), 2, cv2.LINE_AA
    )

    cv2.imwrite(out_path, img)
    return out_path


# =========================
# PIPELINE
# =========================
def run_pipeline(image_path: str, target_text: str, label: Optional[str] = None, cfg: Optional[Config] = None):
    cfg = cfg or Config()
    label = label or target_text

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image_path not found: {image_path}")

    # ✅ Load SAM properly (model will NOT be None)
    sam = SAM(cfg.sam3_model_path)

    # A) blind segmentation
    results = segment_everything_sam(sam, image_path)

    # optional pruning
    results = prune_masks(results, cfg.min_area, cfg.max_masks, cfg.mask_threshold)

    # B) overlay
    overlay_path = save_overlay(image_path, results, cfg)

    # C) VLM choose ID
    mask_id, raw = vlm_choose_id(cfg, image_path, overlay_path, target_text)

    # D) centroid + bbox
    centroid, bbox = mask_to_geometry(results, mask_id, cfg.mask_threshold)

    # E) final image
    final_path = save_final(image_path, centroid, bbox, label, cfg)

    return final_path, overlay_path, mask_id, centroid, bbox, raw


# =========================
# CLI USER INPUT
# =========================
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    target_text = input("What object to detect?: ").strip()

    final_path, overlay_path, mask_id, centroid, bbox, raw = run_pipeline(image_path, target_text)

    print("\n✅ DONE")
    print("Overlay image:", overlay_path)
    print("Final image:", final_path)
    print("mask_id:", mask_id)
    print("Centroid:", centroid)
    print("BBox:", bbox)
    print("VLM reply:", raw)