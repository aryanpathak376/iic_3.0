"""
run_synthetic.py — validate the BEV module with no simulator and no download.

Everything here is numpy and OpenCV. It renders a road scene from a virtual
camera whose pose you set, so you know the exact metric truth for every object,
then pushes that render through the real BEV pipeline and measures the error.

Why this is not a downgrade from CARLA for THIS module:

  Your module solves a geometry problem. CARLA's value is photorealism, which
  matters enormously for training a detector and not at all for validating a
  homography. What you actually need is a camera with a known pose and objects
  at known metric positions, and that is exactly what this gives you, with
  ground truth that is exact rather than merely very good.

  It also does something CARLA cannot: isolate one error source at a time.
  Real footage mixes calibration error, detector jitter and suspension pitch
  together. Here you can turn on precisely one and measure it.

    python run_synthetic.py                    # clean geometry check
    python run_synthetic.py --pitch-noise 1.0  # simulate suspension movement
    python run_synthetic.py --pixel-noise 2.0  # simulate a jittery detector
    python run_synthetic.py --save-frames
"""

import argparse
import csv

import cv2
import numpy as np

from bev import BEVGrid, BEVProjector, GroundCalibration, Intrinsics

# Vehicle geometry, ego reference point = rear axle
CAM = dict(height=2.4, pitch=8.0, mount_x=2.9, width=800, img_h=600, fov=90.0)


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------

class Box:
    """A road user, as a 3D box in ego-frame metres."""

    def __init__(self, X, Y, length, width, height, yaw=0.0, kind="car"):
        self.X, self.Y = X, Y
        self.l, self.w, self.h = length, width, height
        self.yaw, self.kind = yaw, kind

    def corners(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        out = []
        for dx in (-self.l / 2, self.l / 2):
            for dy in (-self.w / 2, self.w / 2):
                for dz in (0.0, self.h):
                    out.append([self.X + dx * c - dy * s,
                                self.Y + dx * s + dy * c, dz])
        return np.array(out)

    @property
    def near_face_x(self) -> float:
        """Truth for comparison: the bbox bottom sees the nearest extent."""
        return self.corners()[:, 0].min()


def make_scene(t: float, rng: np.random.Generator) -> list[Box]:
    """A messy road: mixed vehicle types, no lane discipline, closing traffic.

    Deliberately unstructured. Vehicles drift laterally, a motorcycle weaves,
    and a bus occludes part of the scene, which is the situation your occupancy
    module has to reason about.
    """
    boxes = [
        Box(14 + 3 * np.sin(0.3 * t), -1.2 + 0.8 * np.sin(0.5 * t),
            4.2, 1.8, 1.5, kind="car"),
        Box(26 - 2 * np.sin(0.2 * t), 2.6 + 1.2 * np.sin(0.4 * t + 1),
            4.5, 1.9, 1.6, yaw=0.06, kind="car"),
        Box(9 + 2.5 * np.sin(0.7 * t), 2.2 * np.sin(0.9 * t),
            1.9, 0.7, 1.5, kind="motorcycle"),
        Box(34, -4.5, 11.0, 2.5, 3.2, kind="bus"),
        Box(19, 5.4, 3.0, 1.4, 1.7, yaw=-0.05, kind="rickshaw"),
    ]
    for b in boxes:
        b.Y += rng.normal(0, 0.02)
    return boxes


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------

def render(calib: GroundCalibration, boxes: list[Box], t: float) -> np.ndarray:
    W, H = calib.intr.width, calib.intr.height
    img = np.full((H, W, 3), (150, 160, 170), np.uint8)          # sky

    def poly(pts3d, colour):
        uv, d = calib.project_3d(pts3d)
        if np.any(d <= 0.1):
            return
        cv2.fillPoly(img, [uv.astype(np.int32)], colour)

    # Road surface, drawn as strips from far to near (painter's algorithm)
    for x0 in range(80, 2, -2):
        poly(np.array([[x0, 9, 0], [x0, -9, 0], [x0 - 2, -9, 0], [x0 - 2, 9, 0]]),
             (72, 72, 74))
    # Unpaved shoulders, because Indian roads usually have them
    for side in (1, -1):
        for x0 in range(80, 2, -2):
            poly(np.array([[x0, side * 12, 0], [x0, side * 9, 0],
                           [x0 - 2, side * 9, 0], [x0 - 2, side * 12, 0]]),
                 (96, 104, 118))

    # Dashed centre line. This is your calibration check: in a correct BEV
    # these must come out parallel and constant-width.
    for x0 in np.arange(4, 80, 9.0):
        poly(np.array([[x0, 0.08, 0.01], [x0, -0.08, 0.01],
                       [x0 + 3, -0.08, 0.01], [x0 + 3, 0.08, 0.01]]),
             (225, 225, 225))
    for side in (1, -1):
        for x0 in np.arange(2, 80, 2.0):
            poly(np.array([[x0, side * 9 - 0.1, 0.01], [x0, side * 9 + 0.1, 0.01],
                           [x0 + 2, side * 9 + 0.1, 0.01],
                           [x0 + 2, side * 9 - 0.1, 0.01]]), (210, 210, 205))

    palette = {"car": (60, 90, 180), "motorcycle": (40, 150, 60),
               "bus": (40, 170, 210), "rickshaw": (50, 200, 235)}
    for b in sorted(boxes, key=lambda b: -b.X):          # far to near
        c = b.corners()
        uv, d = calib.project_3d(c)
        if np.any(d <= 0.5):
            continue
        # corner order from Box.corners(): (dx, dy, dz) nested loops
        faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        col = palette[b.kind]
        for f in faces:
            cv2.fillPoly(img, [uv[list(f)].astype(np.int32)], col)
            cv2.polylines(img, [uv[list(f)].astype(np.int32)], True,
                          tuple(int(v * 0.6) for v in col), 1)
    return img


def detect(calib: GroundCalibration, boxes: list[Box],
           pixel_noise: float, rng) -> tuple[list[dict], list[Box]]:
    """The 2D boxes a perfect detector would emit, optionally with jitter."""
    dets, truth = [], []
    W, H = calib.intr.width, calib.intr.height
    for i, b in enumerate(boxes):
        uv, d = calib.project_3d(b.corners())
        if np.any(d <= 0.5):
            continue
        x1, y1 = uv[:, 0].min(), uv[:, 1].min()
        x2, y2 = uv[:, 0].max(), uv[:, 1].max()
        if x2 < 0 or x1 > W or y2 < 0 or y1 > H:
            continue
        if pixel_noise > 0:
            x1, y1, x2, y2 = [v + rng.normal(0, pixel_noise)
                              for v in (x1, y1, x2, y2)]
        dets.append({"bbox": (max(0, x1), max(0, y1),
                              min(W - 1, x2), min(H - 1, y2)),
                     "id": i, "cls": b.kind})
        truth.append(b)
    return dets, truth


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--pitch-noise", type=float, default=0.0,
                    help="std dev in degrees, simulates suspension pitch")
    ap.add_argument("--pixel-noise", type=float, default=0.0,
                    help="std dev in pixels on each bbox edge")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--out", default="bev_error.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    intr = Intrinsics.from_fov(CAM["width"], CAM["img_h"], CAM["fov"])

    # The calibration the module BELIEVES. It never changes.
    calib = GroundCalibration(intr, CAM["height"], pitch_deg=CAM["pitch"],
                              mount_x_m=CAM["mount_x"])
    grid = BEVGrid(x_min=-6, x_max=40, y_min=-15, y_max=15, resolution=0.2)
    proj = BEVProjector(calib, grid)
    print(f"grid {grid.shape}, ego cell {tuple(round(v) for v in grid.ego_cell)}, "
          f"horizon row {calib.horizon_v:.1f}")

    rows = []
    for n in range(args.frames):
        t = n * 0.05
        boxes = make_scene(t, rng)

        # The TRUE pose this frame. With pitch noise it drifts away from what
        # the module believes, which is exactly what a real suspension does.
        true_pitch = CAM["pitch"] + (rng.normal(0, args.pitch_noise)
                                     if args.pitch_noise else 0.0)
        true_calib = GroundCalibration(intr, CAM["height"], pitch_deg=true_pitch,
                                       mount_x_m=CAM["mount_x"])

        frame = render(true_calib, boxes, t)
        dets, truth = detect(true_calib, boxes, args.pixel_noise, rng)
        out = proj.project_detections(dets)

        for d, b in zip(out, truth):
            if not d["reliable"]:
                continue
            eX, eY = d["ground"]
            rows.append(dict(frame=n, kind=b.kind,
                             true_x=b.near_face_x, true_y=b.Y,
                             est_x=eX, est_y=eY,
                             err_x=eX - b.near_face_x, err_y=eY - b.Y))

        if args.save_frames and n % 40 == 0:
            bev = proj.render(frame, out)
            bev = cv2.resize(bev, None, fx=2, fy=2,
                             interpolation=cv2.INTER_NEAREST)
            h = max(frame.shape[0], bev.shape[0])
            pad = lambda im: cv2.copyMakeBorder(
                im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
            cv2.imwrite(f"synth_{n:04d}.png",
                        np.hstack([pad(frame), pad(bev)]))

    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summarise(rows, args)


def summarise(rows, args):
    if not rows:
        print("no samples")
        return
    tx = np.array([r["true_x"] for r in rows])
    ex = np.abs([r["err_x"] for r in rows])
    ey = np.abs([r["err_y"] for r in rows])
    print(f"\n{len(rows)} samples   "
          f"pitch noise {args.pitch_noise} deg, pixel noise {args.pixel_noise} px")
    print("  range bin | n    | mean |dX| | p95 |dX| | mean |dY|")
    print("  " + "-" * 55)
    for lo in range(0, 40, 5):
        m = (tx >= lo) & (tx < lo + 5)
        if m.sum() < 3:
            continue
        print(f"  {lo:2d}-{lo + 5:2d} m   |{m.sum():5d} | {ex[m].mean():8.2f} m |"
              f" {np.percentile(ex[m], 95):7.2f} m | {ey[m].mean():8.2f} m")

    print("\n  by object type (mean |dX|):")
    for k in sorted(set(r["kind"] for r in rows)):
        m = np.array([r["kind"] == k for r in rows])
        print(f"    {k:12s} {ex[m].mean():6.2f} m   (n={m.sum()})")


if __name__ == "__main__":
    main()