"""
calibrate_video.py — calibrate the BEV on real footage, where nobody tells you
the camera pose.

This is the "calibration method that works on multiple roads" deliverable.
Two modes, and you should support both:

  HORIZON MODE (fast, 2 clicks, needs a roughly straight road)
      Click the horizon line, type in the camera height. Done.
      Recovers pitch analytically. Best when you know the mounting height.

  FOUR-POINT MODE (robust, needs one known distance on the ground)
      Click 4 points on the road surface forming a rectangle, type its real
      width and length. Recovers the full homography with no pose assumptions.
      Use this when you do not know the camera height, which is most found
      footage. On an Indian road the easiest metric references are:
        - carriageway width: 3.5 m per lane on a National Highway,
          3.0 m on most urban arterials (IRC:86 / IRC:73)
        - lane marking segments on highways: 3 m painted, 6 m gap
        - a stationary car's wheelbase: ~2.5 m for a hatchback
      Measure with a tape if you can. Guessed scale means every velocity
      downstream is wrong by the same factor.

Usage:
    python calibrate_video.py road.mp4 --mode four_point --out calib.json
    python calibrate_video.py road.mp4 --mode horizon --height 1.35 --out calib.json
"""

import argparse
import json

import cv2
import numpy as np

from bev import BEVGrid, BEVProjector, GroundCalibration, Intrinsics

clicks: list[tuple[int, int]] = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"  point {len(clicks)}: ({x}, {y})")


def collect(frame, n, prompt):
    clicks.clear()
    win = "calibrate"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)
    print(prompt)
    while len(clicks) < n:
        disp = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(disp, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(disp, f"{len(clicks)}/{n}  (u = undo, q = quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and clicks:
            clicks.pop()
        elif k == ord("q"):
            raise SystemExit("cancelled")
    cv2.destroyWindow(win)
    return list(clicks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mode", choices=["horizon", "four_point"], default="four_point")
    ap.add_argument("--height", type=float, default=1.35,
                    help="camera height in metres (dashcam on a windscreen: 1.2-1.5)")
    ap.add_argument("--fov", type=float, default=None,
                    help="horizontal fov in degrees; dashcams are usually 90-140")
    ap.add_argument("--width-m", type=float, default=3.5,
                    help="real width of the clicked rectangle, metres")
    ap.add_argument("--length-m", type=float, default=15.0,
                    help="real length of the clicked rectangle, metres")
    ap.add_argument("--near-m", type=float, default=6.0,
                    help="distance from camera to the near edge, metres")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="calib.json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read frame")
    H, W = frame.shape[:2]

    fov = args.fov if args.fov else 90.0
    intr = Intrinsics.from_fov(W, H, fov)
    grid = BEVGrid(x_min=0, x_max=40, y_min=-12, y_max=12, resolution=0.15)

    if args.mode == "horizon":
        pts = collect(frame, 2,
                      "Click TWO points along the horizon (where road meets sky).")
        v_h = (pts[0][1] + pts[1][1]) / 2.0
        roll = np.rad2deg(np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
        pitch = GroundCalibration.pitch_from_horizon(intr, v_h)
        calib = GroundCalibration(intr, args.height, pitch_deg=pitch, roll_deg=-roll)
        print(f"\n  pitch {pitch:.2f} deg, roll {-roll:.2f} deg, height {args.height} m")
        proj = BEVProjector(calib, grid)
        payload = dict(mode="horizon", width=W, height=H, fov=fov,
                       camera_height_m=args.height, pitch_deg=pitch, roll_deg=-roll)
    else:
        pts = collect(frame, 4,
                      "Click 4 points ON THE ROAD SURFACE, in this order:\n"
                      "   1 near-left   2 near-right   3 far-right   4 far-left\n"
                      "Lane markings and the kerb line make good anchors.")
        near, far = args.near_m, args.near_m + args.length_m
        hw = args.width_m / 2.0
        # Road frame: X forward, Y LEFT. So left edge is +Y.
        ground = np.array([[near, +hw], [near, -hw], [far, -hw], [far, +hw]])
        H_i2g = GroundCalibration.homography_from_points(np.array(pts, float), ground)

        calib = GroundCalibration(intr, args.height)  # placeholder pose
        calib._H_i2g = H_i2g                          # override with the measured one
        calib._H_g2i = np.linalg.inv(H_i2g)
        proj = BEVProjector(calib, grid)
        payload = dict(mode="four_point", width=W, height=H,
                       image_points=[[float(a), float(b)] for a, b in pts],
                       ground_points=ground.tolist(),
                       H_image_to_ground=H_i2g.tolist())

    # --- the check that actually matters -------------------------------------
    # Warp the frame and look at it. On a correct calibration, lane markings run
    # PARALLEL in the BEV and keep constant width from near to far. If they fan
    # out, your pitch is too small; if they pinch, too large. This is faster and
    # more reliable than any numeric residual.
    bev = proj.render(frame)
    cv2.imwrite("calib_check.png", bev)
    print("\n  wrote calib_check.png")
    print("  CHECK: lane markings must be parallel and constant-width top to bottom.")
    print("         fanning out => pitch too small.  pinching in => pitch too large.")

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {args.out}")

    cv2.imshow("BEV check", bev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()