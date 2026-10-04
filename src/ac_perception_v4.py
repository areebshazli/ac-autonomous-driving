"""
Assetto Corsa - real-time perception (YOLOP via ONNX Runtime + DirectML) + metric bird's-eye view.

Works on AMD, NVIDIA and Intel GPUs under Windows (DirectML). No PyTorch needed at runtime.

Install (once):
    pip uninstall -y onnxruntime onnxruntime-directml
    pip install onnxruntime-directml dxcam opencv-python numpy

Pipeline
    dxcam (DXGI screen capture) -> main thread -> LatestSlot(frames)
    inference worker thread     <- LatestSlot(frames), -> LatestSlot(results)
    main thread                 <- LatestSlot(results), blends overlay, draws BEV, shows/records

Hotkeys (focus the OpenCV window):
    q          quit
    r          start/stop recording (timestamped .mp4, encoded on a background thread)
    b          BEV: toggle dimmed camera underlay
    [ ]        camera pitch  -/+ 0.5 deg
    - =        camera vFOV   -/+ 1 deg     (Assetto Corsa's FOV slider is VERTICAL FOV)
    , .        camera height -/+ 5 cm
    p          print current calibration
"""
import ctypes
import math
import queue
import threading
import time
import traceback
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import cv2
import dxcam
import numpy as np
import onnxruntime as ort

# =============================== CONFIG ====================================
# Model ---------------------------------------------------------------------
MODEL_PATH = None              # None = auto-find yolop-384-640.onnx, then yolop-640-640.onnx
DML_DEVICE_ID = 0              # GPU index for DirectML. If the wrong GPU is busy, try 1.
CONF_THRESH = 0.25             # applied to obj_conf * cls_conf (same as YOLOP demo)
IOU_THRESH = 0.45
MAX_CANDIDATES = 300           # cap boxes entering NMS
MASK_EMA = 0.6                 # temporal smoothing of seg scores; 1.0 = off
OVERLAY_ALPHA = 0.45

# Capture -------------------------------------------------------------------
MONITOR_IDX = 0
CAPTURE_REGION = None          # (left, top, right, bottom) or None = whole monitor
CAPTURE_FPS = 60
DISPLAY_W, DISPLAY_H = 1280, 720
EXCLUDE_WINDOW_FROM_CAPTURE = True   # hides OUR window from dxcam -> no infinity-mirror

# Camera calibration for the BEV (tune live with hotkeys) ---------------------
CAM_HEIGHT_M = 1.15            # hood camera, metres above the road
CAM_HOOD_ROW_PX = 605          # display-space (1280x720) row where your OWN HOOD starts, measured from a clear-road
                               # screenshot. Nothing at or behind this row is real road - it is your car's bodywork,
                               # so any detection box bottom that reaches this row is occluded, not actually that far.
CAM_PITCH_DEG = -6.0           # + = camera looks down. AC's hood cam looks UP ~5-6 deg (measured from a real
                               # frame's lane-line vanishing point); press Ctrl+Alt+C in v5 to auto-calibrate.
CAM_VFOV_DEG = 60.0            # = the FOV value in Assetto Corsa (it is vertical FOV)
CAM_X_OFFSET_M = 0.0           # metres the camera sits to the RIGHT of the car's true centreline (+ = right).
                               # Nothing solves for this on its own - a wrong value shows up as the BEV's "blue"
                               # planned path never quite lining up with the hood, even when the car is actually
                               # centred. Calibrate with Ctrl+Alt+Z in v5 while driving straight and well-centred.
PITCH_ACCEL_GAIN_DEG_PER_G = 3.0  # dynamic pitch compensation: under acceleration the nose lifts (hood-cam looks
                                  # further up); this many degrees of EXTRA "looking up" per g of longitudinal
                                  # accel, added on top of the calibrated baseline each frame. NEEDS VERIFYING:
                                  # I don't know AC's sign convention for accG[2] without testing on real hardware.
                                  # Watch the BEV while accelerating hard: if it gets WORSE (lane flares/pinches
                                  # more than with this feature off), the sign is backwards - negate this constant.
PITCH_ACCEL_MAX_DEG = 3.0        # clamp: never let the dynamic term move pitch more than this from baseline
PITCH_ACCEL_SMOOTH_ALPHA = 0.06  # EMA smoothing on acc_lon_g before it's used - raw per-frame G is noisy (engine
                                  # vibration, road surface, gear shifts), and reacting to it directly made the
                                  # BEV visibly wobble/rebuild constantly instead of only during real acceleration
                                  # or braking. ~0.06 gives a time constant of roughly half a second at ~25-30fps -
                                  # lower = smoother but slower to react; raise it if compensation feels laggy.
PITCH_ACCEL_DEADZONE_DEG = 0.2   # only actually rebuild the BEV once the (already-smoothed) target pitch has
                                  # moved at least this far - a second layer of damping against residual jitter.

# Lighting robustness -----------------------------------------------------------
USE_CLAHE = True                 # local contrast normalisation before inference - meant to help lane markings
                                  # stay visible in harsh sun/shadow/glare. Toggle off if it ever seems to hurt.
CLAHE_CLIP_LIMIT = 2.5
CLAHE_GRID = 8

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_GRID, CLAHE_GRID))


def apply_clahe(bgr):
    """Local contrast normalisation on lightness only (LAB colour space), so lane paint stays distinguishable
    from asphalt under harsh sun, shadow bands under overpasses, or headlight glare, without shifting colour."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# BEV canvas ------------------------------------------------------------------
BEV_W, BEV_H = 360, 720
BEV_FORWARD_M = 50.0           # metres visible ahead; lateral range follows from aspect
BEV_NEAR_M = 3.0               # nearest ground row used to build the homography

# Output ----------------------------------------------------------------------
REC_FPS = 30
WINDOW_NAME = "Assetto Corsa - Perception & BEV"
WINDOW_POS = None              # e.g. (1920, 0) to place the window on a 2nd monitor
WINDOW_SIZE = None             # e.g. (1230, 540) to shrink the window (content is 1640x720)
PERF_LOG_EVERY_S = 5.0
# ===========================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX
Frame = namedtuple("Frame", "id t img")


@dataclass
class Result:
    frame_id: int
    t_capture: float
    overlay: np.ndarray        # HxWx3 uint8 BGR, black where nothing is drawn
    mask: np.ndarray           # HxW bool, True where overlay is drawn
    boxes: np.ndarray          # Nx4 float32, xyxy in display pixels
    scores: np.ndarray         # N
    pre_ms: float
    run_ms: float              # time inside the ONNX session (the GPU part)
    post_ms: float


class LatestSlot:
    """Single-item mailbox: put() overwrites, wait_new() blocks until something newer exists.
    No queue => no backlog => no accumulating latency."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None
        self._seq = 0

    def put(self, item):
        with self._cv:
            self._item = item
            self._seq += 1
            self._cv.notify_all()

    def peek(self):
        with self._cv:
            return self._item

    def wait_new(self, last_seq, timeout=0.1):
        with self._cv:
            if self._seq == last_seq:
                self._cv.wait(timeout)
            if self._seq == last_seq:
                return None, last_seq
            return self._item, self._seq


# ============================ MODEL & INFERENCE =============================
def find_model():
    names = ("yolop-384-640.onnx", "yolop-640-640.onnx")
    here = Path(__file__).resolve().parent
    hub = Path.home() / ".cache" / "torch" / "hub" / "hustvl_yolop_main" / "weights"
    cands = [Path(MODEL_PATH)] if MODEL_PATH else []
    for n in names:
        cands += [here / n, Path.cwd() / n, hub / n]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        "No YOLOP .onnx found. Expected one of: "
        + ", ".join(str(c) for c in cands[:6])
        + "\nSet MODEL_PATH at the top of the script.")


def nms_numpy(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


class YolopOnnx:
    def __init__(self, disp_w, disp_h):
        path = find_model()
        avail = ort.get_available_providers()
        providers = []
        if "DmlExecutionProvider" in avail:
            providers.append(("DmlExecutionProvider", {"device_id": DML_DEVICE_ID}))
        else:
            print("[WARN] DmlExecutionProvider not available -> running on CPU (expect much higher latency). "
                  "Run: pip uninstall -y onnxruntime && pip install onnxruntime-directml")
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.enable_mem_pattern = False                       # recommended for DirectML
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=providers)
        print(f"[INFO] Model: {path}")
        print(f"[INFO] Execution providers in use: {self.sess.get_providers()}")

        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        try:
            self.in_h, self.in_w = int(inp.shape[2]), int(inp.shape[3])
        except (TypeError, ValueError):
            self.in_h, self.in_w = 384, 640                 # dynamic-shape export fallback

        # Letterbox geometry (aspect-preserving, gray padding) - computed once.
        self.w, self.h = disp_w, disp_h
        self.ratio = min(self.in_w / disp_w, self.in_h / disp_h)
        self.new_w, self.new_h = round(disp_w * self.ratio), round(disp_h * self.ratio)
        dw, dh = self.in_w - self.new_w, self.in_h - self.new_h
        self.pad_l, self.pad_t = dw // 2, dh // 2
        self.pad_r, self.pad_b = dw - self.pad_l, dh - self.pad_t
        self.pad_off = np.array([self.pad_l, self.pad_t, self.pad_l, self.pad_t], np.float32)

        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
        self._prev = {}

        dummy = np.zeros((disp_h, disp_w, 3), np.uint8)     # warm-up: DirectML graph compile
        for _ in range(3):
            self.infer(dummy)
        self._prev.clear()
        print(f"[INFO] Model input {self.in_w}x{self.in_h} "
              f"(content {self.new_w}x{self.new_h}, pad top/bottom {self.pad_t}/{self.pad_b})")

    def _seg_mask(self, name, seg):
        """Crop letterbox padding, turn 2-class logits into one signed score, EMA-smooth,
        upsample the smooth score (not the blocky mask) and threshold."""
        t, l, h, w = self.pad_t, self.pad_l, self.new_h, self.new_w
        score = seg[0, 1, t:t + h, l:l + w] - seg[0, 0, t:t + h, l:l + w]   # >0 <=> class 1 wins
        prev = self._prev.get(name)
        if prev is not None and MASK_EMA < 1.0:
            score = MASK_EMA * score + (1.0 - MASK_EMA) * prev
        score = np.ascontiguousarray(score, dtype=np.float32)
        self._prev[name] = score
        return cv2.resize(score, (self.w, self.h), interpolation=cv2.INTER_LINEAR) > 0

    def _detections(self, det):
        """YOLOP rows are (cx, cy, w, h, obj, cls) in letterboxed pixels."""
        empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
        scores = det[:, 4] * det[:, 5]
        keep = scores > CONF_THRESH
        if not keep.any():
            return empty
        d, s = det[keep], scores[keep]
        if len(s) > MAX_CANDIDATES:
            top = np.argpartition(-s, MAX_CANDIDATES)[:MAX_CANDIDATES]
            d, s = d[top], s[top]
        boxes = np.stack((d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2,
                          d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2), axis=1).astype(np.float32)
        idx = nms_numpy(boxes, s, IOU_THRESH)
        boxes, s = boxes[idx], s[idx]
        boxes = (boxes - self.pad_off) / self.ratio                      # undo letterbox
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, self.w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, self.h - 1)
        return boxes, s.astype(np.float32)

    def infer(self, frame_bgr):
        t0 = time.perf_counter()
        small = cv2.resize(frame_bgr, (self.new_w, self.new_h), interpolation=cv2.INTER_AREA)
        if USE_CLAHE:
            small = apply_clahe(small)
        padded = cv2.copyMakeBorder(small, self.pad_t, self.pad_b, self.pad_l, self.pad_r,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (self.in_w, self.in_h), (0, 0, 0),
                                     swapRB=True, crop=False)              # BGR->RGB, NCHW, /255
        blob -= self.mean                                                  # ImageNet normalisation
        blob /= self.std
        t1 = time.perf_counter()

        det, da, ll = self.sess.run(None, {self.in_name: blob})
        t2 = time.perf_counter()

        da_m = self._seg_mask("da", da)
        ll_m = self._seg_mask("ll", ll)
        overlay = np.zeros((self.h, self.w, 3), np.uint8)                  # BGR
        overlay[..., 1] = (da_m & ~ll_m).view(np.uint8) * 255              # drivable = green
        overlay[..., 2] = ll_m.view(np.uint8) * 255                        # lane lines = red (on top)
        boxes, scores = self._detections(det[0] if det.ndim == 3 else det)
        t3 = time.perf_counter()

        return (overlay, da_m | ll_m, boxes, scores,
                (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)

    def run(self, frames, results, stop):
        last = 0
        try:
            while not stop.is_set():
                fr, last = frames.wait_new(last, timeout=0.1)
                if fr is None:
                    continue
                overlay, mask, boxes, scores, pre, run, post = self.infer(fr.img)
                results.put(Result(fr.id, fr.t, overlay, mask, boxes, scores, pre, run, post))
        except Exception:
            traceback.print_exc()
            stop.set()


# ============================== BIRD'S-EYE VIEW =============================
class BirdsEyeView:
    """Metric inverse-perspective map from a pinhole camera model (height, pitch, vertical FOV).

    The homography is built by projecting four *ground-plane points in metres* into the image,
    so BEV pixels have a real scale (pixels per metre) and distances can be read off.
    Assumes flat ground - hills/bumps will bias distances."""

    def __init__(self, disp_w, disp_h):
        self.w, self.h = disp_w, disp_h
        self.height_m, self.pitch_deg, self.vfov_deg = CAM_HEIGHT_M, CAM_PITCH_DEG, CAM_VFOV_DEG
        self.x_offset_m = CAM_X_OFFSET_M              # how far RIGHT of the car's true centreline the camera
                                                       # sits; see zero_lateral_calibration() in v5 to measure it
        self.base_pitch = CAM_PITCH_DEG               # the calibrated/manually-set BASELINE pitch. self.pitch_deg
                                                       # is the LIVE value actually used each frame - it may equal
                                                       # base_pitch + a live acceleration correction (see v5's main
                                                       # loop) but must never be saved as if it were the baseline.
        self.ego_y = BEV_H - 30                       # BEV row of the camera / ego position
        self.ppm = self.ego_y / BEV_FORWARD_M         # pixels per metre (same on both axes)
        self.show_cam = False
        self.rebuild()

    def rebuild(self):
        f = (self.h / 2) / math.tan(math.radians(self.vfov_deg) / 2)
        cx, cy = self.w / 2, self.h / 2
        s, c = math.sin(math.radians(self.pitch_deg)), math.cos(math.radians(self.pitch_deg))

        def to_px(X, Z):      # ground point (X right OF THE CAR'S TRUE CENTRELINE, Z forward)
            Xc = X - self.x_offset_m                  # shift into camera-relative coordinates
            y_c = c * self.height_m - s * Z
            z_c = max(s * self.height_m + c * Z, 1e-3)
            return [f * Xc / z_c + cx, f * y_c / z_c + cy]

        half = BEV_W / 2 / self.ppm
        zf, zn = BEV_FORWARD_M, BEV_NEAR_M
        src = np.float32([to_px(-half, zf), to_px(half, zf), to_px(-half, zn), to_px(half, zn)])
        yf, yn = self.ego_y - zf * self.ppm, self.ego_y - zn * self.ppm
        dst = np.float32([[0, yf], [BEV_W, yf], [0, yn], [BEV_W, yn]])
        self.M = cv2.getPerspectiveTransform(src, dst)

    def hood_floor_z(self):
        """Ground-plane distance that maps to CAM_HOOD_ROW_PX at the current pitch: the closest distance this
        camera can EVER report. A box whose bottom edge is at or below this row is occluded by your own hood -
        the true distance could be anything from here down to zero (bumper contact), and we cannot tell which."""
        f = (DISPLAY_H / 2) / math.tan(math.radians(self.vfov_deg / 2))
        horizon = DISPLAY_H / 2 + f * math.tan(math.radians(-self.pitch_deg))
        ang = math.atan((CAM_HOOD_ROW_PX - horizon) / f)
        return self.height_m / math.tan(ang) if ang > 1e-6 else float("inf")

    def project(self, boxes):
        """Bottom-centre of each box (where the tyres touch the road) -> BEV px and metres.
        `reliable` is False where the box bottom is occluded by the hood: Z is then only an upper bound
        (the true distance may be much smaller, down to 0), not a measurement - see hood_floor_z()."""
        if len(boxes) == 0:
            e = np.zeros(0, np.float32)
            return np.zeros((0, 2), np.float32), e, e, np.zeros(0, bool)
        pts = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
        bev = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), self.M).reshape(-1, 2)
        X = (bev[:, 0] - BEV_W / 2) / self.ppm
        Z = (self.ego_y - bev[:, 1]) / self.ppm
        reliable = Z > self.hood_floor_z() * 1.03      # small margin; hood_floor_z() adapts to the CURRENT pitch
                                                        # (including the dynamic acceleration compensation) -
                                                        # a fixed pixel-row threshold would drift stale by up to
                                                        # ~1 m across that range and silently mistrust or over-
                                                        # trust some near detections depending on how hard you're
                                                        # accelerating or braking at that exact moment
        return bev, X, Z, reliable

    def render(self, disp, result, det):
        if self.show_cam:
            canvas = cv2.warpPerspective(disp, self.M, (BEV_W, BEV_H), flags=cv2.INTER_LINEAR)
            canvas = cv2.convertScaleAbs(canvas, alpha=0.35)
        else:
            canvas = np.full((BEV_H, BEV_W, 3), 18, np.uint8)
        if result is not None:
            warped = cv2.warpPerspective(result.overlay, self.M, (BEV_W, BEV_H), flags=cv2.INTER_LINEAR)
            canvas = cv2.add(canvas, warped)

        cx = BEV_W // 2
        for k in (-5.25, -1.75, 1.75, 5.25):     # lane edges if you are centred in a 3.5 m lane
            x = int(cx + k * self.ppm)
            cv2.line(canvas, (x, 0), (x, self.ego_y), (55, 55, 55), 1)
        for d in range(10, int(BEV_FORWARD_M) + 1, 10):          # distance rings, in metres
            y = int(self.ego_y - d * self.ppm)
            cv2.line(canvas, (0, y), (BEV_W, y), (90, 90, 90), 1)
            cv2.putText(canvas, f"{d}m", (4, y - 4), FONT, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, 0), (cx, self.ego_y), (90, 90, 90), 1)
        cv2.fillPoly(canvas, [np.int32([[cx, self.ego_y - 14], [cx - 8, self.ego_y + 8], [cx + 8, self.ego_y + 8]])],
                     (255, 255, 255))

        bev, X, Z, _rel = det
        half_w, length = 0.95 * self.ppm, 4.5 * self.ppm         # ~1.9 m x 4.5 m car footprint
        for (bx, by), x_m, z_m in zip(bev, X, Z):
            if not (0 < z_m < BEV_FORWARD_M * 1.1 and abs(x_m) < BEV_W / 2 / self.ppm):
                continue
            cv2.rectangle(canvas, (int(bx - half_w), int(by - length)), (int(bx + half_w), int(by)),
                          (0, 255, 255), 2)
            cv2.putText(canvas, f"{z_m:.0f}m", (int(bx + half_w) + 3, int(by) - 4),
                        FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, "BEV (metric)", (10, 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def summary(self):
        return (f"height={self.height_m:.2f} m  pitch={self.pitch_deg:.1f} deg (base {self.base_pitch:.1f})  "
                f"vfov={self.vfov_deg:.0f} deg  x_offset={self.x_offset_m:+.2f} m")


# ================================ RECORDER ==================================
class Recorder:
    """Video encoding on its own thread so it never stalls the display loop."""

    def __init__(self, size, fps):
        self.size, self.fps = size, fps
        self.q, self.thread, self.dropped = None, None, 0

    @property
    def active(self):
        return self.thread is not None

    def start(self):
        path = time.strftime("ac_perception_%Y%m%d_%H%M%S.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        self.q, self.dropped = queue.Queue(maxsize=90), 0

        def loop():
            while True:
                f = self.q.get()
                if f is None:
                    break
                writer.write(f)
            writer.release()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        print(f"[REC] Recording to {path}")

    def push(self, frame):
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def stop(self):
        if self.thread:
            self.q.put(None)
            self.thread.join()
            self.thread = None
            print(f"[REC] Stopped ({self.dropped} frames dropped)")


# ================================== MAIN ====================================
def exclude_window_from_capture(title):
    """Windows 10 2004+: make this window invisible to screen capture (DXGI/dxcam included),
    while still visible on the monitor. Prevents the 'infinity mirror' feedback loop."""
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if not hwnd:
            return False
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    except Exception:
        return False


def draw_detections(view, boxes, scores, Z, reliable=None):
    for i, ((x1, y1, x2, y2), sc, z) in enumerate(zip(boxes, scores, Z)):
        rel = True if reliable is None else bool(reliable[i])
        col = (255, 255, 0) if rel else (0, 140, 255)                                   # orange: hood-occluded
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        label = f"{sc:.2f}" + (f" {z:.0f}m" if rel and 0 < z < 150 else "")   # occluded: score only, no fake/word label
        cv2.putText(view, label, (int(x1), max(int(y1) - 6, 14)), FONT, 0.5, col, 1, cv2.LINE_AA)


def handle_key(key, bev, recorder):
    """Returns False when the user asks to quit."""
    if key == ord("q"):
        return False
    if key == ord("r"):
        recorder.stop() if recorder.active else recorder.start()
    elif key == ord("b"):
        bev.show_cam = not bev.show_cam
    elif key in (ord("["), ord("]"), ord("-"), ord("="), ord(","), ord(".")):
        if key == ord("["):
            bev.base_pitch -= 0.5                     # adjust the BASELINE, not the live (possibly dynamically
        elif key == ord("]"):                         # compensated) pitch_deg - otherwise the very next frame's
            bev.base_pitch += 0.5                     # acceleration correction would silently overwrite this
        elif key == ord("-"):
            bev.vfov_deg = max(10.0, bev.vfov_deg - 1)
        elif key == ord("="):
            bev.vfov_deg = min(120.0, bev.vfov_deg + 1)
        elif key == ord(","):
            bev.height_m = max(0.2, bev.height_m - 0.05)
        elif key == ord("."):
            bev.height_m += 0.05
        bev.pitch_deg = bev.base_pitch                # keep pitch_deg in sync for immediate visual feedback here;
        bev.rebuild()                                 # v5's main loop will re-derive it (base_pitch + dynamic
                                                       # correction) again on the next perception frame regardless
        print("[CAL]", bev.summary())
    elif key == ord("p"):
        print("[CAL]", bev.summary())
    return True


def main():
    frames, results = LatestSlot(), LatestSlot()
    stop = threading.Event()

    runner = YolopOnnx(DISPLAY_W, DISPLAY_H)
    bev = BirdsEyeView(DISPLAY_W, DISPLAY_H)
    recorder = Recorder((DISPLAY_W + BEV_W, DISPLAY_H), REC_FPS)

    cam = dxcam.create(output_idx=MONITOR_IDX, output_color="BGR")
    cam.start(region=CAPTURE_REGION, target_fps=CAPTURE_FPS)

    worker = threading.Thread(target=runner.run, args=(frames, results, stop), daemon=True)
    worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if WINDOW_SIZE:
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)
    if WINDOW_POS:
        cv2.moveWindow(WINDOW_NAME, *WINDOW_POS)

    print("[INFO] Running. q=quit  r=record  b=BEV underlay  [ ] pitch  - = vfov  , . height  p=print")
    fid, disp_fps, last_res_id = 0, 0.0, -1
    perf = [0.0, 0.0, 0.0, 0.0]            # smoothed pre, run, post, lag
    t_prev, t_rec, t_log = time.perf_counter(), 0.0, time.perf_counter()
    need_exclude = EXCLUDE_WINDOW_FROM_CAPTURE

    try:
        while not stop.is_set():
            frame = cam.get_latest_frame()               # blocks until a new frame exists
            if frame is None:
                continue
            now = time.perf_counter()
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            # Resize ONCE; the worker and the renderer both use this exact array (read-only).
            if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H:
                disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
            else:
                disp = frame.copy()
            fid += 1
            frames.put(Frame(fid, now, disp))

            res = results.peek()
            view = disp.copy()
            lag_ms = 0.0
            if res is not None:
                blend = cv2.addWeighted(disp, 1.0 - OVERLAY_ALPHA, res.overlay, OVERLAY_ALPHA, 0)
                np.copyto(view, blend, where=res.mask[..., None])     # tint only masked pixels
                lag_ms = (now - res.t_capture) * 1000.0
                if res.frame_id != last_res_id:
                    last_res_id = res.frame_id
                    for i, v in enumerate((res.pre_ms, res.run_ms, res.post_ms, lag_ms)):
                        perf[i] = v if perf[i] == 0 else 0.9 * perf[i] + 0.1 * v
                det = bev.project(res.boxes)
                draw_detections(view, res.boxes, res.scores, det[2], det[3])
            else:
                det = bev.project(np.zeros((0, 4)))

            ai_ms = perf[0] + perf[1] + perf[2]
            hud = f"Display {disp_fps:3.0f} fps | AI {ai_ms:4.1f} ms (net {perf[1]:4.1f}) | overlay lag {lag_ms:3.0f} ms"
            if recorder.active:
                hud += " | REC"
            cv2.putText(view, hud, (12, 26), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, hud, (12, 26), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            combined = np.hstack((view, bev.render(disp, res, det)))
            cv2.imshow(WINDOW_NAME, combined)

            if need_exclude:                              # window exists only after first imshow
                ok = exclude_window_from_capture(WINDOW_NAME)
                print("[INFO] Window excluded from screen capture." if ok else
                      "[WARN] Could not exclude window from capture (needs Windows 10 2004+). "
                      "Move the window to a 2nd monitor or outside CAPTURE_REGION.")
                need_exclude = False

            if recorder.active and now - t_rec >= 1.0 / REC_FPS:
                recorder.push(combined)
                t_rec = now

            if now - t_log >= PERF_LOG_EVERY_S:
                t_log = now
                print(f"[PERF] display {disp_fps:3.0f} fps | pre {perf[0]:4.1f} | onnx {perf[1]:5.1f} | "
                      f"post {perf[2]:4.1f} ms | lag {perf[3]:4.0f} ms")

            if not handle_key(cv2.waitKey(1) & 0xFF, bev, recorder):
                break
    finally:
        stop.set()
        worker.join(timeout=2.0)
        cam.stop()
        recorder.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
