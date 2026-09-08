"""Numerical checks on bev.py. Run this before you trust a single frame."""

import numpy as np

from bev import BEVGrid, BEVProjector, GroundCalibration, Intrinsics

np.set_printoptions(precision=4, suppress=True)

intr = Intrinsics.from_fov(800, 600, 90.0)
calib = GroundCalibration(intr, height_m=2.4, pitch_deg=8.0, yaw_deg=0.0, roll_deg=0.0)
grid = BEVGrid(x_min=0, x_max=40, y_min=-15, y_max=15, resolution=0.2)
proj = BEVProjector(calib, grid)

print(f"fx = {intr.fx:.2f} px   horizon row = {calib.horizon_v:.2f}")

# 1. Horizon behaves correctly with pitch -------------------------------------
z = GroundCalibration(intr, 2.4, pitch_deg=0.0)
assert abs(z.horizon_v - intr.cy) < 1e-6, "zero pitch must put horizon at image centre"
prev = z.horizon_v
for p in [2, 5, 10, 20]:
    h = GroundCalibration(intr, 2.4, pitch_deg=p).horizon_v
    assert h < prev, "pitching down must raise the horizon in the image"
    prev = h
print("1. horizon vs pitch                 OK")

# 2. Ground -> image -> ground round trip --------------------------------------
X = np.array([3, 5, 10, 15, 20, 30, 40, 60], dtype=float)
Y = np.array([0, -2, 3.5, -7, 1, 0, 6, -4], dtype=float)
xy = np.stack([X, Y], axis=1)
uv = proj.ground_to_image(xy)
back = proj.image_to_ground(uv)
err = np.abs(back - xy)
assert np.nanmax(err) < 1e-8, f"round trip error {np.nanmax(err)}"
print(f"2. ground->image->ground round trip OK  (max err {np.nanmax(err):.2e} m)")

# 3. A point above the horizon must be rejected, not silently wrong ------------
bad = proj.image_to_ground(np.array([[400.0, calib.horizon_v - 20.0]]))
assert np.all(np.isnan(bad)), "above-horizon point must return NaN"
print("3. above-horizon rejection          OK")

# 4. Grid round trip -----------------------------------------------------------
c, r = grid.ground_to_grid(X, Y)
Xb, Yb = grid.grid_to_ground(c, r)
assert np.max(np.abs(Xb - X)) < 1e-9 and np.max(np.abs(Yb - Y)) < 1e-9
assert grid.shape == (200, 150), grid.shape
print(f"4. metres<->grid round trip         OK  (grid {grid.shape})")

# 5. Simulate a detection: project a real 3D box, then recover its position ----
#    This is the honest end-to-end test. We place a 1.8 m wide, 1.5 m tall box
#    on the road, render its 2D bounding box the way a detector would report it,
#    then feed that box back through project_detections.
def synth_bbox(Xc, Yc, w=1.8, h=1.5, l=4.0):
    corners = []
    for dx in (-l / 2, l / 2):
        for dy in (-w / 2, w / 2):
            for dz in (0.0, h):
                P = np.array([Xc + dx, Yc + dy, dz])
                pc = calib._R @ (P - np.array([0.0, 0.0, calib.height_m]))
                if pc[2] <= 0.01:
                    continue
                corners.append([intr.fx * pc[0] / pc[2] + intr.cx,
                                intr.fy * pc[1] / pc[2] + intr.cy])
    c = np.array(corners)
    return (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())

LEN, WID = 4.0, 1.8
print("   Truth is the NEAR FACE of the object, which is what the bbox bottom sees.")
print("\n   centre X  near face  |  est X   err  |  true span      est span")
print("   " + "-" * 66)
worst_r = worst_w = 0.0
for Xc, Yc in [(8, 0), (12, -3), (20, 2), (30, -5), (40, 0)]:
    box = synth_bbox(Xc, Yc, w=WID, l=LEN)
    out = proj.project_detections([{"bbox": box, "id": 1}])[0]
    if not out["reliable"]:
        print(f"   {Xc:8.1f}  |  rejected: {out['reject']}")
        continue
    ex, ey = out["ground"]
    near = Xc - LEN / 2
    worst_r = max(worst_r, abs(ex - near))
    lo, hi = out["extent"]
    tlo, thi = Yc - WID / 2, Yc + WID / 2
    worst_w = max(worst_w, abs((hi - lo) - WID))
    print(f"   {Xc:8.1f} {near:10.1f}  | {ex:6.2f} {ex-near:+6.2f} |"
          f" [{tlo:5.1f},{thi:5.1f}]  [{lo:5.1f},{hi:5.1f}]")

assert worst_r < 0.05, worst_r
print(f"\n5. near-face projection             OK  (max range err {worst_r:.3f} m,"
      f" max width err {worst_w:.2f} m)")

# 6. Observable mask should be a trapezoid that widens with distance -----------
m = proj.observable_mask
near = m[grid.rows - 20].sum()      # ~4 m ahead
far = m[20].sum()                   # ~36 m ahead
assert far > near, "camera footprint must widen with distance"
print(f"6. observable footprint             OK  (near {near} cells, far {far} cells, "
      f"{100 * m.mean():.1f}% of grid visible)")

# 7. Four-point calibration must recover the analytic homography ---------------
gp = np.array([[6.0, -3.0], [6.0, 3.0], [25.0, -3.0], [25.0, 3.0]])
ip = proj.ground_to_image(gp)
H_est = GroundCalibration.homography_from_points(ip, gp)
test = np.array([[15.0, 1.0], [30.0, -4.0], [8.0, 2.5]])
tu = proj.ground_to_image(test)
h = np.concatenate([tu, np.ones((3, 1))], axis=1).T
g = H_est @ h
g = (g[:2] / g[2]).T
assert np.max(np.abs(g - test)) < 1e-4, np.max(np.abs(g - test))
print("7. 4-point calibration agrees       OK")

# 8. How bad does a 1 degree roll error actually get? --------------------------
print("\n   Sensitivity of lateral position to a 1 deg roll miscalibration:")
wrong = BEVProjector(GroundCalibration(intr, 2.4, pitch_deg=8.0, roll_deg=1.0), grid)
for Xc in (10, 20, 30, 40):
    truth = np.array([[Xc, 0.0]])
    uv1 = proj.ground_to_image(truth)
    est = wrong.image_to_ground(uv1)[0]
    print(f"      object at {Xc:2d} m -> lateral error {abs(est[1]):.2f} m, "
          f"range error {abs(est[0] - Xc):.2f} m")