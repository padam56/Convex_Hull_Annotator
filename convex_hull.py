"""Shape-aware probability-to-annotation pipeline.

threshold -> CC -> area filter -> shape-aware cleanup (convex hull /
closing / skeleton, routed by form factor) -> raster mask + VIA polygons.

    from convex_hull import CHA
    result = CHA.from_spec("domain_specs/rutting.json")(prob_map)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_dilation, closing as morph_closing,
    convex_hull_image, disk, skeletonize,
)


# ─── constants ───────────────────────────────────────────────────────────────
DEFAULT_AUTO_THRESHOLDS = {"convex_hull_max_form": 3.0, "closing_max_form": 10.0}


# ─── config + result dataclasses ─────────────────────────────────────────────
@dataclass
class CHAConfig:
    threshold_policy: str = "otsu"       # otsu | li | yen | mean | fixed
    fixed_threshold: float = 0.5
    min_area_frac: float = 0.001
    max_area_frac: float = 0.85
    shape_policy: str = "auto"            # auto | convex_hull | closing | skeleton | raw
    auto_thresholds: dict = field(default_factory=lambda: dict(DEFAULT_AUTO_THRESHOLDS))
    closing_radius: int = 3
    skeleton_dilate: int = 4
    preserve_holes: bool = False
    min_quality_score: float = 0.0
    domain: str = "default"
    source_spec: str | None = None

    @classmethod
    def from_dict(cls, d):
        d = {**d, **d.pop("cha", {})}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_spec(cls, path):
        p = Path(path)
        cfg = cls.from_dict(json.loads(p.read_text()))
        cfg.source_spec = str(p)
        return cfg


@dataclass
class CHAResult:
    raster_mask: np.ndarray
    polygons: list
    n_components_before_filter: int
    n_components_kept: int
    threshold_used: float
    config: CHAConfig

    def write_via_json(self, path, image_filename):
        """Write polygons as VIA-format JSON."""
        regions = [{
            "shape_attributes": {
                "name": "polygon",
                "all_points_x": [int(x) for x, _ in p["vertices"]],
                "all_points_y": [int(y) for _, y in p["vertices"]],
            },
            "region_attributes": {
                "policy":   p["policy"],
                "score":    round(p["score"], 4),
                "solidity": round(p["solidity"], 4),
                "form":     round(p["form"], 4),
                "domain":   self.config.domain,
            },
        } for p in self.polygons]
        H, W = self.raster_mask.shape[:2]
        rec = {
            "filename": image_filename,
            "size":     [H, W],
            "regions":  regions,
            "file_attributes": {
                "threshold":   round(self.threshold_used, 4),
                "n_total":     self.n_components_before_filter,
                "n_kept":      self.n_components_kept,
                "source_spec": self.config.source_spec or "(programmatic)",
            },
        }
        Path(path).write_text(json.dumps({image_filename: rec}, indent=2))


# ─── threshold + shape-policy dispatch ──────────────────────────────────────
_THRESH_FNS = {
    "otsu": threshold_otsu,
    "li":   threshold_li,
    "yen":  threshold_yen,
    "mean": np.mean,
}


def _threshold(p, cfg):
    if not np.isfinite(p).all() or p.max() <= 0:
        return float("inf")
    if cfg.threshold_policy == "fixed":
        return float(cfg.fixed_threshold)
    if np.allclose(p.min(), p.max()):
        return float(p.mean())
    fn = _THRESH_FNS.get(cfg.threshold_policy)
    if fn is None:
        raise ValueError(f"unknown threshold_policy: {cfg.threshold_policy}")
    return float(fn(p))


def _form_factor(region):
    """F = perimeter² / (4π·area).  F=1 circle, larger=thinner."""
    if region.area < 1:
        return float("inf")
    return float((region.perimeter ** 2) / (4.0 * math.pi * region.area))


def _decide_policy(region, cfg):
    if cfg.shape_policy != "auto":
        return cfg.shape_policy
    F, t = _form_factor(region), cfg.auto_thresholds
    if F < t["convex_hull_max_form"]:
        return "convex_hull"
    if F < t["closing_max_form"]:
        return "closing"
    return "skeleton"


def _apply_policy(region, policy, cfg):
    sub = region.image.astype(bool)
    if policy == "convex_hull":
        return (morph_closing(sub, disk(cfg.closing_radius))
                if cfg.preserve_holes else convex_hull_image(sub))
    if policy == "closing":
        return morph_closing(sub, disk(cfg.closing_radius))
    if policy == "skeleton":
        return binary_dilation(skeletonize(sub), disk(cfg.skeleton_dilate))
    if policy == "raw":
        return sub
    raise ValueError(f"unknown shape_policy: {policy}")


# ─── helpers ────────────────────────────────────────────────────────────────
def _collapse(p):
    """Reduce (H, W, C) -> (H, W) by per-pixel max."""
    if p.ndim == 3:
        return p[..., 0] if p.shape[-1] == 1 else p.max(axis=-1)
    return p


def _polygon(mask, bbox):
    minr, minc, _, _ = bbox
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_TC89_L1)
    if not cnts:
        return []
    cnt = max(cnts, key=lambda c: c.shape[0])
    return [(int(pt[0][0] + minc), int(pt[0][1] + minr)) for pt in cnt]


def _record_region(region, cfg, img_area):
    """Score, clean, and serialise one connected component.
    Returns (cleaned_mask, bbox, polygon_dict) or None to skip."""
    a = region.area / img_area
    if a < cfg.min_area_frac or a > cfg.max_area_frac:
        return None
    score = float(region.intensity_mean)
    if score < cfg.min_quality_score:
        return None
    policy = _decide_policy(region, cfg)
    cleaned = _apply_policy(region, policy, cfg)
    verts = _polygon(cleaned, region.bbox)
    if not verts:
        return None
    e = region.inertia_tensor_eigvals
    poly = {
        "vertices":   verts,
        "policy":     policy,
        "score":      score,
        "solidity":   float(region.solidity),
        "elongation": math.sqrt(e[0] / max(e[1], 1e-9)),
        "form":       _form_factor(region),
        "bbox":       [int(c) for c in region.bbox],
        "area_frac":  float(a),
    }
    return cleaned, region.bbox, poly


# ─── public API ──────────────────────────────────────────────────────────────
def run_cha(prob_map, cfg=None):
    cfg = cfg or CHAConfig()
    p = _collapse(np.asarray(prob_map, dtype=np.float32))
    H, W = p.shape
    tau = _threshold(p, cfg)
    if not np.isfinite(tau):
        return CHAResult(np.zeros((H, W), np.uint8), [], 0, 0, 0.0, cfg)

    binary = (p >= tau).astype(np.uint8)
    labelled = label(binary, connectivity=2)
    raster = np.zeros((H, W), np.uint8)
    polygons = []
    for region in regionprops(labelled, intensity_image=p):
        out = _record_region(region, cfg, H * W)
        if out is None:
            continue
        cleaned, (minr, minc, _, _), poly = out
        raster[minr:minr + cleaned.shape[0],
               minc:minc + cleaned.shape[1]][cleaned] = 255
        polygons.append(poly)
    return CHAResult(raster, polygons, int(labelled.max()),
                      len(polygons), tau, cfg)


class CHA:
    def __init__(self, cfg=None):
        self.cfg = cfg or CHAConfig()

    @classmethod
    def from_spec(cls, path):
        return cls(CHAConfig.from_spec(path))

    def __call__(self, prob_map):
        return run_cha(prob_map, self.cfg)


# ─── self-test ───────────────────────────────────────────────────────────────
def _synth_prob(kind, H=256, W=256):
    yy, xx = np.ogrid[:H, :W]
    if kind == "sandboil":
        m = ((yy - 128) ** 2 + (xx - 128) ** 2 < 50 ** 2).astype(np.float32)
        return cv2.GaussianBlur(m, (11, 11), 4) * 0.95
    if kind == "sinkhole":
        m = ((yy - 80) ** 2 + (xx - 180) ** 2 < 30 ** 2).astype(np.float32)
        return cv2.GaussianBlur(m, (11, 11), 4) * 0.9
    if kind == "rutting":
        m = ((np.abs(yy - 100) < 6) & (np.abs(xx - 128) < 90)).astype(np.float32)
        return cv2.GaussianBlur(m, (7, 7), 2) * 0.85
    if kind == "crack":
        m = np.zeros((H, W), np.float32)
        for t in np.linspace(0, 1, 600):
            x = int(20 + 220 * t)
            y = int(40 + 160 * t + 10 * math.sin(7 * t))
            m[y, x] = 1.0
        return cv2.GaussianBlur(m, (3, 3), 0.7) * 0.95
    raise ValueError(kind)


def _selftest():
    for name in ("sandboil", "sinkhole", "rutting", "crack"):
        r = run_cha(_synth_prob(name), CHAConfig(domain=name))
        if not r.polygons:
            print(f"  {name:10}  -> NO components"); continue
        p = r.polygons[0]
        print(f"  {name:10}  -> policy={p['policy']:12}  "
              f"score={p['score']:.3f}  form={p['form']:.1f}  "
              f"tau={r.threshold_used:.3f}")


if __name__ == "__main__":
    print("CHA self-test:")
    _selftest()
