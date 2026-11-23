#!/usr/bin/env python3
"""End-to-end: Keras segmenter -> CHA -> mask + VIA polygons.

Replace MODEL_PATH / IMAGE_DIR / DOMAIN_SPEC for your case.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convex_hull import CHA  # noqa: E402

MODEL_PATH  = "models/sandboilnet/best_model.h5"
IMAGE_DIR   = "datasets/images"
OUT_MASK    = Path("output/masks")
OUT_VIA     = Path("output/via_json")
DOMAIN_SPEC = "domain_specs/sandboil.json"
IMG_SIZE    = 512
CUSTOM_OBJECTS: dict = {}


def main() -> int:
    OUT_MASK.mkdir(parents=True, exist_ok=True)
    OUT_VIA.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False, custom_objects=CUSTOM_OBJECTS,
    )

    cha = CHA.from_spec(DOMAIN_SPEC)

    paths = sorted(p for p in Path(IMAGE_DIR).iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for i, img_path in enumerate(paths):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        prob = model.predict(x[None, ...], verbose=0)[0]
        result = cha(prob)
        cv2.imwrite(str(OUT_MASK / f"{img_path.stem}_mask.png"), result.raster_mask)
        result.write_via_json(OUT_VIA / f"{img_path.stem}.json", img_path.name)
        print(f"[{i+1}/{len(paths)}] {img_path.name}  "
              f"kept={result.n_components_kept}  τ={result.threshold_used:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
