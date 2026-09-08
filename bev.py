"""
bev.py — Bird's Eye View generation module.

Coordinate frames used throughout (pick these once, never change them):

  ROAD frame (metric, right-handed, origin on the ground directly below the camera)
      X : forward,  metres
      Y : LEFT,     metres
      Z : up,       metres   (ground plane is Z = 0)

  IMAGE frame
      u : column, pixels (0 = left)
      v : row,    pixels (0 = top)

  GRID frame (the occupancy grid the planner consumes)
      col, row : integer cell indices into the BEV raster

The only assumption in this whole file is that the road is a plane at Z = 0.
Everything else is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1. Camera intrinsics
# --------------------------------------------------------------------------

@dataclass
class Intrinsics:
    """Pinhole camera intrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> "Intrinsics":
        """Build intrinsics from horizontal field of view.

        This is the CARLA case: an RGB camera blueprint with attribute
        'image_size_x', 'image_size_y' and 'fov' gives you these exactly.
        Square pixels, principal point at the image centre.
        """
        f = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------
# 2. Extrinsic calibration -> the ground-plane homography
# --------------------------------------------------------------------------

def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Camera axes expressed in the ROAD frame when pitch = yaw = roll = 0:
#   image-right  = road -Y   (because road Y is LEFT)
#   image-down   = road -Z
#   image-forward= road +X
_R0 = np.array([[0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0]], dtype=np.float64)


@dataclass
class GroundCalibration:
    """Camera pose relative to the road plane.

    height_m : camera height above the road, metres
    pitch_deg: positive = nose down (the usual dashcam mounting)
    yaw_deg  : positive = aimed left
    roll_deg : positive = rotated clockwise in the image

    MEASURED ERROR BUDGET (800x600, 90 deg fov, h = 2.4 m, pitch = 8 deg).
    Run test_bev.py to reproduce these on your own configuration.

        source of error              range error @20 m   @40 m
        --------------------------   -----------------   ------
        pitch off by 1 deg                 2.6 m         9.1 m
        height off by 10 cm                0.8 m         1.7 m
        contact point off by 1 px          0.4 m         1.7 m
        roll off by 1 deg                  ~0            ~0 (lateral 0.06 m)

    Pitch dominates everything, and it is the one parameter that does not
    stay put: braking pitches the nose down, acceleration lifts it, road
    camber and a loaded boot both shift it. A calibration measured while
    parked is wrong the moment you touch the brakes.

    Two consequences you should design around now:
      1. Trust nothing past about 25 m. Size your occupancy grid accordingly.
      2. Re-estimate pitch per frame from the horizon or vanishing point
         rather than treating it as a constant. See pitch_from_horizon.
    """
    intr: Intrinsics
    height_m: float
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    # Camera position in the ego/road frame. X is forward, Y is left.
    mount_x_m: float = 0.0
    mount_y_m: float = 0.0

    _H_g2i: np.ndarray = field(init=False, repr=False)
    _H_i2g: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        R = (_rot_z(np.deg2rad(self.roll_deg))
             @ _rot_x(np.deg2rad(self.pitch_deg))
             @ _rot_y(np.deg2rad(self.yaw_deg))
             @ _R0)

        # A ground point is P = (X, Y, 0) in the ego/road frame.
        # The camera is mounted at C = (mount_x_m, mount_y_m, height_m).
        # p_cam = R (P - C), so the ground->camera map collapses to 3x3.
        C = np.array([self.mount_x_m, self.mount_y_m, self.height_m])
        t = -R @ C
        M = np.column_stack([R[:, 0], R[:, 1], t])

        self._H_g2i = self.intr.K @ M
        self._H_i2g = np.linalg.inv(self._H_g2i)
        self._R = R

    # -- the two matrices everything else is built on ----------------------

    @property
    def H_ground_to_image(self) -> np.ndarray:
        """3x3 mapping homogeneous (X, Y, 1) metres -> homogeneous (u, v, 1) px."""
        return self._H_g2i

    @property
    def H_image_to_ground(self) -> np.ndarray:
        """3x3 mapping homogeneous (u, v, 1) px -> homogeneous (X, Y, 1) metres."""
        return self._H_i2g

    def project_3d(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N,3) road/ego-frame points into image pixels.

        Returns (uv, depth), where depth is positive for points in front of
        the camera.
        """
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        C = np.array([self.mount_x_m, self.mount_y_m, self.height_m])
        pc = (self._R @ (xyz - C).T).T
        depth = pc[:, 2]
        uv = np.full((len(xyz), 2), np.nan, dtype=np.float64)
        good = depth > 1e-9
        uv[good, 0] = self.intr.fx * pc[good, 0] / depth[good] + self.intr.cx
        uv[good, 1] = self.intr.fy * pc[good, 1] / depth[good] + self.intr.cy
        return uv, depth

    # -- horizon ------------------------------------------------------------

    @property
    def horizon_v(self) -> float:
        """Image row of the horizon line.

        Anything at or above this row is either sky or a point infinitely far
        away. Projecting it through the homography produces mathematically
        valid but physically meaningless coordinates, often *behind* the
        camera. This is the single most common IPM bug. Always mask it out.
        """
        # A ground point infinitely far ahead: direction (1, 0, 0) in road frame.
        d_cam = self._R @ np.array([1.0, 0.0, 0.0])
        if abs(d_cam[2]) < 1e-9:
            return -np.inf
        return self.intr.fy * d_cam[1] / d_cam[2] + self.intr.cy

    # -- alternative calibration paths --------------------------------------

    @staticmethod
    def homography_from_points(image_pts: np.ndarray,
                               ground_pts_m: np.ndarray) -> np.ndarray:
        """Fallback for real footage where the camera pose is unknown.

        Give 4+ image points that lie on the road surface and their metric
        (X, Y) positions in the road frame. Sources of known distances on an
        Indian road: lane marking segment length and gap, standard road width,
        the wheelbase of a stationary vehicle, a measured tape on the ground.

        Returns H_image_to_ground.
        """
        image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 1, 2)
        ground_pts_m = np.asarray(ground_pts_m, dtype=np.float64).reshape(-1, 1, 2)
        if len(image_pts) == 4:
            H, _ = cv2.findHomography(image_pts, ground_pts_m, method=0)
        else:
            H, _ = cv2.findHomography(image_pts, ground_pts_m, method=cv2.RANSAC,
                                      ransacReprojThreshold=0.5)
        return H

    @staticmethod
    def pitch_from_horizon(intr: Intrinsics, horizon_row: float) -> float:
        """Estimate pitch in degrees by clicking the horizon line in one frame.

        Combined with a tape-measured camera height this calibrates a new
        road or a new dashcam in about thirty seconds.
        """
        return float(np.rad2deg(np.arctan2(intr.cy - horizon_row, intr.fy)))


# --------------------------------------------------------------------------
# 3. The metric grid the planner consumes
# --------------------------------------------------------------------------

@dataclass
class BEVGrid:
    """Definition of the BEV raster in metres.

    Keep this separate from the homography. The homography's job is
    image pixels -> metres. This class's job is metres -> grid cells.
    Fusing the two is the mistake that makes it impossible to change grid
    resolution later without re-deriving your calibration.
    """
    x_min: float = -6.0    # metres relative to ego (allows ego footprint)
    x_max: float = 40.0
    y_min: float = -15.0   # metres, negative = right
    y_max: float = 15.0
    resolution: float = 0.2   # metres per cell

    def __post_init__(self) -> None:
        if self.x_min >= 0.0:
            raise ValueError("BEVGrid must include the ego point: x_min must be < 0")
        if self.y_min >= 0.0 or self.y_max <= 0.0:
            raise ValueError("BEVGrid must include the ego point: y range must cross 0")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("invalid grid bounds")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")

    @property
    def rows(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def cols(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def ground_to_grid(self, X, Y):
        """Metres -> (col, row) float cell coordinates.

        Row 0 is the FAR edge (x_max) so that the rendered image looks the
        way a human expects: far away at the top, ego at the bottom.
        Column 0 is the LEFT edge (y_max) for the same reason.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        col = (self.y_max - Y) / self.resolution
        row = (self.x_max - X) / self.resolution
        return col, row

    def grid_to_ground(self, col, row):
        """(col, row) cell coordinates -> metres. Exact inverse of the above."""
        col = np.asarray(col, dtype=np.float64)
        row = np.asarray(row, dtype=np.float64)
        Y = self.y_max - col * self.resolution
        X = self.x_max - row * self.resolution
        return X, Y

    @property
    def ego_cell(self) -> tuple[float, float]:
        """Grid cell containing the ego reference point (0, 0)."""
        return self.ground_to_grid(0.0, 0.0)

    def ego_footprint_mask(self, length_m: float = 4.8, width_m: float = 2.0) -> np.ndarray:
        """Boolean mask covering a conservative rectangular ego footprint."""
        cc, rr = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        X, Y = self.grid_to_ground(cc + 0.5, rr + 0.5)
        return (X >= -length_m / 2.0) & (X <= length_m / 2.0) & \
               (Y >= -width_m / 2.0) & (Y <= width_m / 2.0)

    def in_bounds(self, X, Y) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        return ((X >= self.x_min) & (X < self.x_max)
                & (Y >= self.y_min) & (Y < self.y_max))


# --------------------------------------------------------------------------
# 4. The projector
# --------------------------------------------------------------------------

class BEVProjector:
    """Turns a forward camera frame into a metric bird's eye view.

    The expensive part (working out which source pixel feeds each BEV cell)
    is done once in the constructor. Per frame you pay only a cv2.remap,
    which is a memory copy.
    """

    def __init__(self, calib: GroundCalibration, grid: BEVGrid,
                 horizon_margin_px: float = 5.0):
        self.calib = calib
        self.grid = grid
        self.horizon_margin_px = horizon_margin_px
        self._build_lut()

    # -- setup --------------------------------------------------------------

    def _build_lut(self) -> None:
        """Inverse mapping: for every BEV cell, which image pixel does it come from?

        Going backwards like this is what stops the BEV having holes in it.
        Forward-warping every image pixel leaves gaps in the near field and
        piles thousands of pixels onto single far-field cells.
        """
        rows, cols = self.grid.shape
        cc, rr = np.meshgrid(np.arange(cols, dtype=np.float64),
                             np.arange(rows, dtype=np.float64))
        # Sample cell centres, not corners.
        X, Y = self.grid.grid_to_ground(cc + 0.5, rr + 0.5)

        pts = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], axis=0)
        img = self.calib.H_ground_to_image @ pts
        w = img[2]

        valid = w > 1e-6   # w <= 0 means the point is behind the camera
        u = np.where(valid, img[0] / np.where(valid, w, 1.0), -1.0)
        v = np.where(valid, img[1] / np.where(valid, w, 1.0), -1.0)

        intr = self.calib.intr
        valid &= (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
        valid &= v > (self.calib.horizon_v + self.horizon_margin_px)

        self._map_u = np.where(valid, u, -1.0).reshape(rows, cols).astype(np.float32)
        self._map_v = np.where(valid, v, -1.0).reshape(rows, cols).astype(np.float32)
        self._valid = valid.reshape(rows, cols)

    @property
    def observable_mask(self) -> np.ndarray:
        """Boolean grid: True where this cell is actually inside the camera's view.

        Hand this straight to the occupancy module. Cells that are False are
        UNKNOWN, not FREE. Failing to make that distinction is how a planner
        ends up confidently driving into a region it has never seen.
        """
        return self._valid

    # -- per-frame operations -----------------------------------------------

    def warp_image(self, frame: np.ndarray,
                   interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Warp a full camera frame into the BEV raster."""
        return cv2.remap(frame, self._map_u, self._map_v, interpolation,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        """Warp a binary segmentation mask (e.g. drivable area) into the BEV.

        Nearest-neighbour, so the output stays binary.
        """
        out = cv2.remap(mask.astype(np.uint8), self._map_u, self._map_v,
                        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
        return (out > 0) & self._valid

    def image_to_ground(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) image points on the road surface -> (N, 2) metric (X, Y).

        Points at or above the horizon come back as NaN rather than silently
        wrong numbers.
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        n = len(uv)
        out = np.full((n, 2), np.nan)
        if n == 0:
            return out

        ok = uv[:, 1] > (self.calib.horizon_v + self.horizon_margin_px)
        if not np.any(ok):
            return out

        pts = np.concatenate([uv[ok], np.ones((ok.sum(), 1))], axis=1).T
        g = self.calib.H_image_to_ground @ pts
        w = g[2]
        good = np.abs(w) > 1e-9
        res = np.full((ok.sum(), 2), np.nan)
        res[good, 0] = g[0, good] / w[good]
        res[good, 1] = g[1, good] / w[good]
        out[ok] = res
        return out

    def ground_to_image(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) metric (X, Y) -> (N, 2) image points. Useful for drawing checks."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        pts = np.concatenate([xy, np.ones((len(xy), 1))], axis=1).T
        img = self.calib.H_ground_to_image @ pts
        w = img[2]
        out = np.full((len(xy), 2), np.nan)
        good = w > 1e-9
        out[good, 0] = img[0, good] / w[good]
        out[good, 1] = img[1, good] / w[good]
        return out

    # -- detections ----------------------------------------------------------

    def project_detections(self, detections: list[dict],
                           edge_margin_px: int = 4) -> list[dict]:
        """Place tracked objects on the ground plane.

        Each detection dict may contain:
            'bbox'  : (x1, y1, x2, y2) in pixels          [required]
            'mask'  : bool array, same HxW as the frame   [optional, better]
            'id'    : track id
            'cls'   : class name

        Adds to each returned dict:
            'ground'    : (X, Y) metres of the NEAREST ground contact, or None
            'extent'    : (Y_right, Y_left) metres, the object's lateral span
            'contact_uv': the pixel treated as the ground contact point
            'reliable'  : bool
            'reject'    : reason string when not reliable

        The whole method rests on one idea: a monocular camera cannot measure
        depth, but it CAN measure where an object touches the road, and that
        contact point has a known depth under the flat-ground assumption.
        Everything therefore depends on seeing the bottom of the object.

        READ THIS BEFORE USING 'ground' AS A CENTROID.
        The bottom edge of a 2D box is the projection of whichever part of the
        object is CLOSEST to the camera, so 'ground' is the near face, not the
        centre. The gap is half the object's length along the view ray: about
        0.9 m for a motorcycle, 2 m for a car, 6 m for a bus. If you hand this
        to a tracker that assumes a centroid, every velocity estimate inherits
        a bias that changes as the object rotates.

        The right fix is not to guess a centroid. It is to stop treating
        objects as points: use 'ground' together with 'extent' to stamp a
        footprint into the occupancy grid. A near face plus a lateral span is
        exactly what a collision checker needs, and it is conservative in the
        direction that matters.
        """
        intr = self.calib.intr
        out = []

        for det in detections:
            d = dict(det)
            x1, y1, x2, y2 = [float(t) for t in d["bbox"]]

            if d.get("mask") is not None:
                cu, cv_ = self._contact_from_mask(d["mask"])
            else:
                cu, cv_ = (x1 + x2) / 2.0, y2

            d["contact_uv"] = (cu, cv_)
            d["ground"] = None
            d["reliable"] = False

            # Truncated at the bottom of the frame: the real contact point is
            # off-screen, so the box bottom underestimates the distance badly.
            if cv_ >= intr.height - edge_margin_px:
                d["reject"] = "truncated at image bottom"
                out.append(d)
                continue

            if cv_ <= self.calib.horizon_v + self.horizon_margin_px:
                d["reject"] = "at or above horizon"
                out.append(d)
                continue

            g = self.image_to_ground(np.array([[cu, cv_]]))[0]
            if not np.all(np.isfinite(g)):
                d["reject"] = "degenerate projection"
                out.append(d)
                continue

            d["ground"] = (float(g[0]), float(g[1]))

            # Lateral span: project the two bottom corners of the box at the
            # same image row. Same depth, so the difference is pure width.
            corners = self.image_to_ground(np.array([[x1, cv_], [x2, cv_]]))
            if np.all(np.isfinite(corners)):
                ys = sorted([float(corners[0, 1]), float(corners[1, 1])])
                d["extent"] = (ys[0], ys[1])
            else:
                d["extent"] = None

            d["reliable"] = True
            d["reject"] = ""
            out.append(d)

        return out

    @staticmethod
    def _contact_from_mask(mask: np.ndarray) -> tuple[float, float]:
        """Lowest row of the mask, averaged across its columns.

        More robust than the bounding box bottom edge for two-wheelers and
        partly occluded objects, where the box bottom can sit well below the
        actual wheel contact patch.
        """
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return (float("nan"), float("nan"))
        v_bottom = ys.max()
        band = ys >= v_bottom - 2
        return float(xs[band].mean()), float(v_bottom)

    # -- rendering ------------------------------------------------------------

    def render(self, frame: np.ndarray, projected: list[dict] | None = None,
               drivable_mask: np.ndarray | None = None) -> np.ndarray:
        """Debug view: warped road, drivable area tint, object footprints, range rings."""
        bev = self.warp_image(frame)
        if bev.ndim == 2:
            bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        bev = bev.copy()

        if drivable_mask is not None:
            d = self.warp_mask(drivable_mask)
            tint = np.zeros_like(bev)
            tint[d] = (0, 90, 0)
            bev = cv2.addWeighted(bev, 1.0, tint, 0.5, 0)

        # Range rings every 10 m so distance errors are visible by eye.
        for r in range(10, int(self.grid.x_max) + 1, 10):
            _, row = self.grid.ground_to_grid(r, 0.0)
            cv2.line(bev, (0, int(row)), (self.grid.cols, int(row)), (60, 60, 60), 1)
            cv2.putText(bev, f"{r}m", (4, int(row) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

        if projected:
            for d in projected:
                if not d.get("reliable"):
                    continue
                X, Y = d["ground"]
                if not self.grid.in_bounds(X, Y):
                    continue
                col, row = self.grid.ground_to_grid(X, Y)
                cv2.circle(bev, (int(col), int(row)), 4, (0, 200, 255), -1)
                label = str(d.get("id", d.get("cls", "")))
                if label:
                    cv2.putText(bev, label, (int(col) + 6, int(row)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

        # Ego marker at the origin.
        ec, er = self.grid.ground_to_grid(0.0, 0.0)
        cv2.drawMarker(bev, (int(ec), int(er)), (255, 255, 255),
                       cv2.MARKER_TRIANGLE_UP, 10, 2)
        return bev