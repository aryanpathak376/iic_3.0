"""
run_carla.py — BEV module driven by CARLA, validated against ground truth.

Why start here rather than on real video: CARLA hands you the true position of
every actor. That turns "my BEV looks about right" into a measured error curve,
which is the single most valuable artefact you can put in your report.

    python run_carla.py --frames 300 --out bev_error.csv

Requires: pip install carla==0.9.16   (plus the packaged 0.9.16 server running)
"""

import argparse
import csv
import math

import cv2
import numpy as np

from bev import BEVGrid, BEVProjector, GroundCalibration, Intrinsics

# Camera mounting. These four numbers ARE your calibration in CARLA: no
# checkerboard, no clicking, no guessing. Keep them in one place.
CAM = dict(x=1.5, y=0.0, z=2.4, pitch=-8.0, width=800, height=600, fov=90.0)


def carla_to_road_frame(ego_tf, actor_tf, cam_offset):
    """CARLA world coords -> our road frame (X forward, Y left, origin under camera).

    CARLA is LEFT-handed with Y pointing RIGHT, so the Y sign flips. Getting
    this wrong mirrors your entire scene and is very hard to spot by eye,
    because a mirrored road still looks like a road.
    """
    dx = actor_tf.location.x - ego_tf.location.x
    dy = actor_tf.location.y - ego_tf.location.y
    yaw = math.radians(ego_tf.rotation.yaw)
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return fwd - cam_offset, -right          # (X forward, Y LEFT)


def main():
    import carla

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="bev_error.csv")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    # Synchronous mode. Without this your camera frame and your ground truth
    # come from different instants and every error measurement is polluted.
    settings = world.get_settings()
    original = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    bp = world.get_blueprint_library()
    actors = []
    try:
        ego_bp = bp.filter("vehicle.tesla.model3")[0]
        ego = world.spawn_actor(ego_bp, world.get_map().get_spawn_points()[0])
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)

        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM["width"]))
        cam_bp.set_attribute("image_size_y", str(CAM["height"]))
        cam_bp.set_attribute("fov", str(CAM["fov"]))
        cam_tf = carla.Transform(
            carla.Location(x=CAM["x"], y=CAM["y"], z=CAM["z"]),
            carla.Rotation(pitch=CAM["pitch"]))
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
        actors.append(cam)

        frames = []
        cam.listen(frames.append)

        # The calibration. Note the sign: CARLA pitch is negative for nose-down,
        # ours is positive, because CARLA rotates the other way about that axis.
        intr = Intrinsics.from_fov(CAM["width"], CAM["height"], CAM["fov"])
        calib = GroundCalibration(intr, height_m=CAM["z"], pitch_deg=-CAM["pitch"])
        grid = BEVGrid(x_min=0, x_max=40, y_min=-15, y_max=15, resolution=0.2)
        proj = BEVProjector(calib, grid)
        print(f"horizon row {calib.horizon_v:.1f}, grid {grid.shape}")

        rows = []
        for n in range(args.frames):
            world.tick()
            while not frames:
                pass
            img = frames.pop()
            frames.clear()

            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            frame = arr.reshape((img.height, img.width, 4))[:, :, :3].copy()

            ego_tf = ego.get_transform()

            # Ground truth for every nearby vehicle and walker. In your real
            # pipeline these come from YOLO; here they come from CARLA so you
            # can isolate BEV error from detector error.
            dets, truth = [], []
            for a in world.get_actors().filter("*vehicle*"):
                if a.id == ego.id:
                    continue
                X, Y = carla_to_road_frame(ego_tf, a.get_transform(), CAM["x"])
                if not (0 < X < grid.x_max and abs(Y) < grid.y_max):
                    continue
                box = project_actor_bbox(a, cam, intr)
                if box is None:
                    continue
                dets.append({"bbox": box, "id": a.id, "cls": "vehicle"})
                # Ground truth near face along the view ray, for a fair comparison.
                ext = a.bounding_box.extent
                truth.append((X - ext.x, Y))

            out = proj.project_detections(dets)
            for d, (tX, tY) in zip(out, truth):
                if d["reliable"]:
                    eX, eY = d["ground"]
                    rows.append(dict(frame=n, true_x=tX, true_y=tY,
                                     est_x=eX, est_y=eY,
                                     err_x=eX - tX, err_y=eY - tY))

            if n % 30 == 0:
                cv2.imwrite(f"bev_{n:04d}.png", proj.render(frame, out))
                print(f"frame {n}: {len(out)} objects")

        with open(args.out, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        summarise(rows)

    finally:
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass
        world.apply_settings(original)


def project_actor_bbox(actor, camera, intr):
    """Render a CARLA actor's 3D box as the 2D box a detector would report."""
    import carla

    bb = actor.bounding_box
    verts = [np.array([v.x, v.y, v.z, 1.0])
             for v in bb.get_world_vertices(actor.get_transform())]
    w2c = np.array(camera.get_transform().get_inverse_matrix())

    pts = []
    for v in verts:
        pc = w2c @ v
        # CARLA camera axes (x fwd, y right, z up) -> image axes (x right, y down, z fwd)
        p = np.array([pc[1], -pc[2], pc[0]])
        if p[2] <= 0.1:
            continue
        pts.append([intr.fx * p[0] / p[2] + intr.cx,
                    intr.fy * p[1] / p[2] + intr.cy])
    if len(pts) < 4:
        return None
    p = np.array(pts)
    x1, y1 = p[:, 0].min(), p[:, 1].min()
    x2, y2 = p[:, 0].max(), p[:, 1].max()
    if x2 < 0 or x1 > intr.width or y2 < 0 or y1 > intr.height:
        return None
    return (max(0, x1), max(0, y1), min(intr.width - 1, x2), min(intr.height - 1, y2))


def summarise(rows):
    """The plot that goes in your report: BEV error against true distance."""
    if not rows:
        print("no samples collected")
        return
    tx = np.array([r["true_x"] for r in rows])
    ex = np.array([abs(r["err_x"]) for r in rows])
    ey = np.array([abs(r["err_y"]) for r in rows])
    print(f"\n{len(rows)} samples")
    print("  range bin | n     | mean |err_x| | mean |err_y| | p95 |err_x|")
    print("  " + "-" * 62)
    for lo in range(0, 40, 5):
        m = (tx >= lo) & (tx < lo + 5)
        if m.sum() < 3:
            continue
        print(f"  {lo:2d}-{lo+5:2d} m   | {m.sum():5d} | {ex[m].mean():11.2f} m |"
              f" {ey[m].mean():11.2f} m | {np.percentile(ex[m], 95):9.2f} m")


if __name__ == "__main__":
    main()