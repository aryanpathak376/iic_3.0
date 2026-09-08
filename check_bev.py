"""Confirm bev.py is the current version. Run this after replacing the file."""
import inspect
from bev import BEVGrid, GroundCalibration

need_calib = ["mount_x_m", "mount_y_m"]
need_methods = [(GroundCalibration, "project_3d"),
                (BEVGrid, "ego_cell"), (BEVGrid, "ego_footprint_mask")]

sig = inspect.signature(GroundCalibration.__init__).parameters
missing = [p for p in need_calib if p not in sig]
missing += [f"{c.__name__}.{m}" for c, m in need_methods if not hasattr(c, m)]

if missing:
    print("OUTDATED bev.py — missing:", ", ".join(missing))
else:
    g = BEVGrid()
    print(f"bev.py is current. grid {g.shape}, ego cell "
          f"{tuple(round(v) for v in g.ego_cell)}, x_min {g.x_min} m")