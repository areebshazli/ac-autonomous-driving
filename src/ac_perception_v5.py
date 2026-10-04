"""
ac_perception_v5.py - perception (v4) + telemetry + planner + virtual gamepad.

    python ac_perception_v5.py --shadow          # SAFE: shows what the autopilot WOULD do; sends NOTHING to the game.
                                                 # Uses your normal AC controller setup - drive by hand as usual.
    python ac_perception_v5.py                   # live: the autopilot can be engaged (needs the virtual pad set up in AC)
    python ac_perception_v5.py --passthrough     # live + your physical Xbox-type pad is forwarded to the virtual pad,
                                                 # so you can drive by hand and take over instantly

First time:  drive onto a straight, flat stretch, stay in your lane, press Ctrl+Alt+C  (calibrates the BEV camera pitch).

GLOBAL hotkeys (work while Assetto Corsa has focus).  Hold Ctrl+Alt and press:
    E  engage/disengage autopilot          X  EMERGENCY STOP (brakes, disengages)
    A  request lane change LEFT              D  request lane change RIGHT   (still safety-checked)
    W  set speed +5 km/h                     S  set speed -5 km/h
    C  auto-calibrate BEV pitch (straight flat road, lane lines visible both sides; not while engaged)
    Z  zero the LATERAL calibration: while driving straight and manually centred in your lane (not engaged),
       hold this for ~1s to remove a constant sideways bias between the hood and the planned path
    Q  quit
Ultimate kill switch: press Esc in AC to pause the game. Only use in single-player

Requires: ac_perception_v4.py, ac_planner.py, ac_telemetry.py, ac_control.py in the same folder.
"""
import argparse
import collections
import ctypes
import json
import threading
import time
from pathlib import Path

import cv2
import dxcam
import numpy as np

import ac_perception_v4 as v4
import ac_planner as P
import ac_telemetry as tel_mod
from ac_telemetry import AC_LIVE, ACTelemetry

STATUS_NAMES = {0: "OFF", 1: "REPLAY", 2: "LIVE", 3: "PAUSE"}
CAL_FILE = Path(__file__).with_name("bev_calibration.json")

# ------------------------------------------------------------------ hotkeys
CTRL, ALT = 0x11, 0x12
HOTKEYS = {"engage": ord("E"), "estop": ord("X"), "lc_left": ord("A"), "lc_right": ord("D"),
           "faster": ord("W"), "slower": ord("S"), "calibrate": ord("C"), "zero_lat": ord("Z"),
           "quit": ord("Q")}   # each needs Ctrl+Alt held


class Hotkeys:
    def __init__(self):
        try:
            self._down = ctypes.windll.user32.GetAsyncKeyState
        except Exception:
            self._down = None                                # non-Windows (tests)
        self._prev = {k: False for k in HOTKEYS}

    def poll(self):
        out = set()
        if self._down is None:
            return out
        mods = bool(self._down(CTRL) & 0x8000) and bool(self._down(ALT) & 0x8000)
        for name, vk in HOTKEYS.items():
            down = mods and bool(self._down(vk) & 0x8000)
            if down and not self._prev[name]:
                out.add(name)
            self._prev[name] = down
        return out


# ------------------------------------------------------------------ BEV calibration
def load_cal(bev):
    try:
        d = json.loads(CAL_FILE.read_text())
        bev.pitch_deg, bev.vfov_deg, bev.height_m = float(d["pitch_deg"]), float(d["vfov_deg"]), float(d["height_m"])
        bev.base_pitch = float(d.get("base_pitch", bev.pitch_deg))     # old cal files predate this field
        bev.x_offset_m = float(d.get("x_offset_m", 0.0))
        bev.rebuild()
        return True
    except Exception:
        return False


def save_cal(bev):
    CAL_FILE.write_text(json.dumps({"pitch_deg": bev.base_pitch, "vfov_deg": bev.vfov_deg, "height_m": bev.height_m,
                                     "base_pitch": bev.base_pitch, "x_offset_m": bev.x_offset_m}, indent=1))
    # NOTE: "pitch_deg" is saved as the BASELINE too, not whatever live/dynamically-corrected value bev.pitch_deg
    # happens to hold at shutdown - otherwise a save mid-acceleration would bake a transient correction in as
    # if it were the calibrated value.


def auto_pitch(overlay, bev, cfg, lo=-14.0, hi=4.0, step=0.5, eval_z_max=20.0):
    """One-frame BEV pitch estimate. On a straight, flat road a lane keeps the SAME width in the BEV at every
    distance; a wrong pitch makes it flare or pinch. Pick the pitch where lane width is most constant.

    IMPORTANT: a shallower pitch sees farther (larger z_hi) than a steeper one, and farther rows are inherently
    less reliable (perspective/lens error grows with range). If each candidate's flatness were scored over
    whatever range IT happens to reach, a shallower-but-less-accurate pitch could win just by having a longer,
    error-averaged fit to be flat over - not because it is actually more correct. That is exactly the failure
    mode reported in the field: calibrating gives a BEV that reaches farther but the car drives worse, because
    the far reach is trusted by the steering lookahead and is less accurate than it looks. So every candidate is
    scored over the SAME fixed near-field window (z_min..eval_z_max) a fair, apples-to-apples comparison that
    does not reward apparent reach. Returns (|width slope| in m per 10 m, pitch_deg, n_rows) or None.
    """
    saved, best = bev.pitch_deg, None
    try:
        for p in np.arange(lo, hi + 1e-6, step):
            bev.pitch_deg = float(p)
            bev.rebuild()
            warped = cv2.warpPerspective(overlay, bev.M, (v4.BEV_W, v4.BEV_H), flags=cv2.INTER_NEAREST)
            road, line = P.bev_masks(warped)
            lt = P.LaneTracker(cfg)
            lt.update(road, line)
            z, w, _ = lt.last_rows
            keep = z <= eval_z_max
            z, w = z[keep], w[keep]
            if len(z) < 15 or z[-1] - z[0] < 12:
                continue
            slope = abs(np.polyfit(z, w, 1)[0]) * 10.0
            if best is None or slope < best[0]:
                best = (float(slope), float(p), len(z))
    finally:
        bev.pitch_deg = saved
        bev.rebuild()
    return best if best and best[0] < 0.3 else None


class ShadowStats:
    """Compares the planner's steering with the driver's (AC's steerAngle). The ratio tells how far the configured
    --lock is from reality: planned = wheel_angle / lock_cfg, driver = wheel_angle / lock_true."""

    def __init__(self, n=600):
        self.buf = collections.deque(maxlen=n)

    def add(self, plan, drv):
        self.buf.append((plan, drv))

    def estimate(self, lock_cfg):
        if len(self.buf) < 150:
            return None
        a = np.array(self.buf)
        sp, sd = a[:, 0].std(), a[:, 1].std()
        if sp < 0.03 or sd < 0.03:
            return None                                        # not enough steering to say anything (straight road)
        r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        slope = float(np.sign(r) * sd / sp)
        return r, slope, (lock_cfg / slope if slope > 0.05 else None)


# ------------------------------------------------------------------ drawing
def draw_plan_on_bev(canvas, ap, ppm=13.8, ego_y=690, w=360):
    lane = getattr(ap, "last_lane", None)
    if lane is None or not lane.valid:
        return
    cx = w // 2
    zs = np.linspace(P.PlannerCfg.z_min, lane.z_hi, 24)
    pts = np.array([[cx + lane.x_at(z) * ppm, ego_y - z * ppm] for z in zs], np.int32)
    cv2.polylines(canvas, [pts], False, (255, 200, 0), 2, cv2.LINE_AA)               # lane centre (cyan-ish)
    if ap.last_L:
        tx, tz = lane.x_at(ap.last_L) + ap.last_r, ap.last_L
        cv2.circle(canvas, (int(cx + tx * ppm), int(ego_y - tz * ppm)), 6, (0, 165, 255), 2, cv2.LINE_AA)


def put(view, text, y, color=(255, 255, 255)):
    cv2.putText(view, text, (12, y), v4.FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, text, (12, y), v4.FONT, 0.6, color, 1, cv2.LINE_AA)


def draw_steer_bar(view, steer, driver=None, x=640, y=700, half=150):
    cv2.rectangle(view, (x - half, y - 8), (x + half, y + 8), (60, 60, 60), 1)
    cv2.line(view, (x, y - 12), (x, y + 12), (200, 200, 200), 1)
    cv2.rectangle(view, (x, y - 6), (int(x + np.clip(steer, -1, 1) * half), y + 6), (0, 200, 255), -1)   # planner
    if driver is not None:                                                                                # driver
        dx = int(x + np.clip(driver, -1, 1) * half)
        cv2.line(view, (dx, y - 14), (dx, y + 14), (255, 255, 255), 3)


# ------------------------------------------------------------------ main
def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--shadow", action="store_true", help="never touch the game: compute + display only")
    a.add_argument("--passthrough", action="store_true", help="forward your physical Xbox-type pad to the virtual pad")
    a.add_argument("--vset", type=float, default=60.0, help="cruise speed km/h")
    a.add_argument("--lock", type=float, default=0.45, help="front-wheel angle (rad) at full steering lock - tune")
    a.add_argument("--prefer-side", choices=["left", "right"], default="right", help="overtaking side")
    a.add_argument("--no-auto-lane-change", action="store_true", help="only manual lane changes")
    return a.parse_args()


def _fix_dpi_scaling():
    """Windows scales non-DPI-aware windows by stretching the rendered bitmap (blurry text, a
    'ghosting'/double-edge look on anything fine - exactly the smeared HUD text in recordings).
    Declaring per-monitor DPI awareness makes Windows render at native pixel resolution instead."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)     # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()       # older Windows fallback
        except Exception:
            pass


def main():
    _fix_dpi_scaling()
    args = parse()
    cfg = P.PlannerCfg(stick_lock_rad=args.lock, prefer_side=1 if args.prefer_side == "right" else -1)
    if args.no_auto_lane_change:
        cfg.min_gain = 1e9
    ap = P.Autopilot(cfg)
    ap.v_set = args.vset / 3.6
    ap.dry_run = True                                          # plans continuously; only acts once engaged

    # A physical pad must be detected BEFORE the virtual one exists, otherwise we could pick the virtual one.
    reader = None
    if args.passthrough and not args.shadow:
        try:
            from ac_control import XInputReader
            phys = XInputReader.connected()
            reader = XInputReader(phys[0]) if phys else None
            print(f"[INFO] pass-through: physical controller in XInput slot {phys[0]}" if reader else
                  "[WARN] --passthrough: no physical XInput controller found (DualShock/DualSense need DS4Windows)")
        except Exception as e:
            print(f"[WARN] pass-through unavailable ({e})")

    pad = None
    if not args.shadow:
        try:
            from ac_control import SafePad
            pad = SafePad()
        except Exception as e:                                 # vgamepad / ViGEmBus missing
            print(f"[WARN] virtual gamepad unavailable ({e}) -> running in SHADOW mode")
    shadow = pad is None
    if shadow:
        reader = None

    tel = ACTelemetry()
    frames, results = v4.LatestSlot(), v4.LatestSlot()
    stop = threading.Event()
    runner = v4.YolopOnnx(v4.DISPLAY_W, v4.DISPLAY_H)
    bev = v4.BirdsEyeView(v4.DISPLAY_W, v4.DISPLAY_H)
    have_cal = load_cal(bev)
    cal0 = (bev.base_pitch, bev.vfov_deg, bev.height_m, bev.x_offset_m)   # NOT pitch_deg - that now fluctuates
                                                                          # every frame with acceleration, and
                                                                          # would look "changed" on every exit
    recorder = v4.Recorder((v4.DISPLAY_W + v4.BEV_W, v4.DISPLAY_H), v4.REC_FPS)
    keys = Hotkeys()
    stats = ShadowStats()

    cam = dxcam.create(output_idx=v4.MONITOR_IDX, output_color="BGR")
    cam.start(region=v4.CAPTURE_REGION, target_fps=v4.CAPTURE_FPS)
    worker = threading.Thread(target=runner.run, args=(frames, results, stop), daemon=True)
    worker.start()

    cv2.namedWindow(v4.WINDOW_NAME, cv2.WINDOW_NORMAL)
    if v4.WINDOW_SIZE:
        cv2.resizeWindow(v4.WINDOW_NAME, *v4.WINDOW_SIZE)
    if v4.WINDOW_POS:
        cv2.moveWindow(v4.WINDOW_NAME, *v4.WINDOW_POS)
    print("[INFO] " + ("SHADOW mode: nothing is sent to the game. Watch the WINDOW (planned steering, lane path, "
                       "lane-change suggestions) - the console only prints a status line every 2 s." if shadow else
                       "LIVE mode. Ctrl+Alt+E engages the autopilot. Ctrl+Alt+X = emergency stop."))
    print(f"[INFO] BEV pitch {bev.pitch_deg:+.1f} deg ({'from bev_calibration.json' if have_cal else 'default'}). "
          "On a straight flat road press Ctrl+Alt+C to auto-calibrate.")
    if not shadow:
        print("[INFO] Before engaging, in AC's assists confirm: Gearbox = Automatic, Auto Clutch = ON, handbrake OFF. "
              "If the car doesn't move despite throttle being sent, also check Controls -> the virtual pad (not your "
              "physical one) is the device AC is actually reading")

    engaged, notice, notice_until = False, "", 0.0
    fid, disp_fps, last_res_id, last_plan_id = 0, 0.0, -1, -1
    perf = [0.0, 0.0, 0.0, 0.0]
    t_prev = t_plan_prev = t_rec = t_status = time.perf_counter()
    cmd, det = P.Command(), (np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0, bool))
    bad = {"tel": None, "lag": None}
    hold_until, hold_brake = 0.0, 0.0            # brief deliberate braking after a fault / emergency stop
    hold_steer, hold_t = 0.0, 0.0                # steering carried into that brake-hold (faults only)
    stall = {"since": None, "v0": 0.0, "warned": False}   # detects commanded throttle with no real acceleration
    lat_cal = {"buf": [], "need": 20}   # ~1s of samples at typical perception rate, for Ctrl+Alt+Z
    accg_ema = 0.0   # smoothed longitudinal G for dynamic pitch compensation - see the main loop below
    need_exclude = v4.EXCLUDE_WINDOW_FROM_CAPTURE

    def say(text, secs=4.0):
        nonlocal notice, notice_until
        notice, notice_until = text, time.perf_counter() + secs
        print("[AP]", text)

    def disengage(reason, brake=0.0, hold_s=0.0, steer0=0.0):
        """Hand control back. brake/hold_s: optional gentle stop (faults) or hard stop (emergency).
        steer0: steering to carry into the hold. Faults pass the last planned steering so the car keeps following
        the bend while it brakes (it fades out slowly, see PlannerCfg.fault_steer_tau_s); an emergency stop or a
        manual hand-off passes 0 = straight, because there the driver is deciding what happens next."""
        nonlocal engaged, hold_until, hold_brake, hold_steer, hold_t
        if engaged:
            engaged = False
            ap.reset()
            ap.dry_run = True
            hold_until, hold_brake = time.perf_counter() + hold_s, brake
            hold_steer, hold_t = steer0, time.perf_counter()
            if pad:
                pad.neutral()
            say(f"AUTOPILOT OFF: {reason}", 5.0)

    try:
        while not stop.is_set():
            frame = cam.get_latest_frame()
            if frame is None:
                continue
            now = time.perf_counter()
            disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            if frame.shape[1] != v4.DISPLAY_W or frame.shape[0] != v4.DISPLAY_H:
                disp = cv2.resize(frame, (v4.DISPLAY_W, v4.DISPLAY_H), interpolation=cv2.INTER_LINEAR)
            else:
                disp = frame.copy()
            fid += 1
            frames.put(v4.Frame(fid, now, disp))

            T = tel.read()
            tel_ok = T.alive and T.status == AC_LIVE          # fresh packets AND session is live (not paused/replay)
            if tel_ok:
                # Raw acc_lon_g is noisy frame-to-frame (engine vibration, road surface, gear shifts) - reacting
                # to it directly made the BEV visibly wobble on every tiny fluctuation, not just real acceleration
                # or braking events. Smooth it first (accg_ema below), and only actually rebuild the BEV once the
                # smoothed value has moved enough to matter - two layers of damping instead of none.
                accg_ema = (1 - v4.PITCH_ACCEL_SMOOTH_ALPHA) * accg_ema + v4.PITCH_ACCEL_SMOOTH_ALPHA * T.acc_lon_g
                # dynamic pitch compensation: under acceleration the nose lifts and the hood-cam looks further
                # UP. In this codebase more-negative pitch_deg IS "looking up more" (see CAM_PITCH_DEG), so extra
                # upward-look degrees must be SUBTRACTED from pitch, not added - GAIN itself stays a positive
                # "degrees of extra upward look per g" for readability. Sign of accG[2] still needs verifying on
                # real hardware - see the note by PITCH_ACCEL_GAIN_DEG_PER_G.
                extra_look_up = float(np.clip(v4.PITCH_ACCEL_GAIN_DEG_PER_G * accg_ema,
                                               -v4.PITCH_ACCEL_MAX_DEG, v4.PITCH_ACCEL_MAX_DEG))
            else:
                extra_look_up = 0.0                            # no fresh telemetry: don't guess, hold the baseline
            new_pitch = bev.base_pitch - extra_look_up
            if abs(new_pitch - bev.pitch_deg) > v4.PITCH_ACCEL_DEADZONE_DEG:
                bev.pitch_deg = new_pitch
                bev.rebuild()
            man = reader.read() if reader else None            # physical pad: (steer, throttle, brake) or None
            res = results.peek()
            view = disp.copy()
            lag_ms = 0.0
            if res is not None:
                blend = cv2.addWeighted(disp, 1.0 - v4.OVERLAY_ALPHA, res.overlay, v4.OVERLAY_ALPHA, 0)
                np.copyto(view, blend, where=res.mask[..., None])
                lag_ms = (now - res.t_capture) * 1000.0
                det = bev.project(res.boxes)
                v4.draw_detections(view, res.boxes, res.scores, det[2], det[3])
                if res.frame_id != last_res_id:
                    last_res_id = res.frame_id
                    for i, v in enumerate((res.pre_ms, res.run_ms, res.post_ms, lag_ms)):
                        perf[i] = v if perf[i] == 0 else 0.9 * perf[i] + 0.1 * v

            # ---- hotkeys (global) ----
            for k in keys.poll():
                if k == "quit":
                    stop.set()
                elif k == "engage":
                    q = cmd.info.get("lane_q", 0.0)
                    if shadow:
                        say("shadow mode: nothing to engage (start without --shadow)", 3)
                    elif engaged:
                        disengage("manual")
                    elif not tel_ok:
                        say(f"cannot engage: no live telemetry (status={STATUS_NAMES.get(T.status, T.status)}, "
                            f"fresh={int(T.alive)}) - are you in the car, session running, not paused?", 6)
                    elif q < cfg.q_min:
                        say(f"cannot engage: lane not detected (quality {q:.2f} < {cfg.q_min}). Be on a road with lane "
                            "lines, check the BEV, press Ctrl+Alt+C to calibrate pitch", 8)
                    else:
                        ap.reset(); ap.dry_run = False; engaged = True
                        say("AUTOPILOT ENGAGED", 3)
                elif k == "estop":
                    if engaged:
                        disengage("EMERGENCY STOP", brake=0.8, hold_s=1.5)
                    elif pad:
                        hold_until, hold_brake = now + 1.5, 0.8
                elif k in ("lc_left", "lc_right") and engaged:
                    ap.request_lane_change(-1 if k == "lc_left" else 1)
                elif k in ("faster", "slower"):
                    ap.v_set = float(np.clip(ap.v_set + (5 if k == "faster" else -5) / 3.6, 20 / 3.6, 120 / 3.6))
                elif k == "calibrate":
                    if engaged:
                        say("disengage before calibrating", 3)
                    elif res is None:
                        say("no perception result yet - wait a second and retry", 3)
                    else:
                        best = auto_pitch(res.overlay, bev, cfg)
                        if best:
                            bev.base_pitch = bev.pitch_deg = best[1]
                            bev.rebuild()
                            save_cal(bev)
                            say(f"BEV pitch set to {best[1]:+.1f} deg (lane-width drift {best[0]:.2f} m per 10 m, "
                                f"{best[2]} rows) - saved", 6)
                        else:
                            say("calibration failed: needs a straight, flat road with lane lines on BOTH sides, "
                                "you centred in your lane", 8)
                elif k == "zero_lat":
                    if engaged:
                        say("disengage before zeroing the lateral calibration", 3)
                    elif abs(T.steer) > 0.05 or T.speed_ms < 8.0:
                        say("zero-lateral needs you driving straight (steering near centre) at a reasonable "
                            "speed, well-centred in your lane", 6)
                    else:
                        lat_cal["buf"].append(cmd.info.get("c0", 0.0))
                        if len(lat_cal["buf"]) >= lat_cal["need"]:
                            bias = float(np.mean(lat_cal["buf"]))
                            bev.x_offset_m -= bias                      # see the derivation in the chat: new = old - measured
                            bev.rebuild()
                            save_cal(bev)
                            say(f"lateral calibration zeroed (removed a {bias:+.2f} m bias) - saved. x_offset now "
                                f"{bev.x_offset_m:+.2f} m", 6)
                            lat_cal["buf"].clear()
                        else:
                            say(f"zeroing lateral calibration - hold Ctrl+Alt+Z, keep driving straight and centred "
                                f"({len(lat_cal['buf'])}/{lat_cal['need']} samples)", 2)

            # ---- driver takes over (pass-through): big steering deflection or brake press ----
            if engaged and man and (abs(man[0]) > 0.4 or man[2] > 0.25):
                disengage("driver override")
            if hold_until and man and (abs(man[0]) > 0.3 or man[2] > 0.2 or man[1] > 0.2):
                hold_until = 0.0                                                     # driver took over during a stop hold

            # ---- pads ----
            if pad and now < hold_until:
                hold_steer = P.steer_bleed(hold_steer, now - hold_t, cfg.fault_steer_tau_s)
                hold_t = now
                pad.send(hold_steer, 0.0, hold_brake)                # keep the bend, brake, released automatically
            elif pad and hold_until and now >= hold_until:
                pad.neutral()
                hold_until = 0.0

            # ---- planner: once per NEW perception result (always runs; only ACTS when engaged) ----
            if res is not None and res.frame_id != last_plan_id:
                last_plan_id = res.frame_id
                warped = cv2.warpPerspective(res.overlay, bev.M, (v4.BEV_W, v4.BEV_H), flags=cv2.INTER_NEAREST)
                road, line = P.bev_masks(warped)
                _, X, Z, rel = det
                X, Z = P.occlude_clamp(X, Z, rel, cfg.cam_to_bumper)
                ok = (Z > 0.1) & (Z < 48.0) & (np.abs(X) < 14.0)
                dets = np.stack([X[ok], Z[ok]], axis=1) if ok.any() else np.zeros((0, 2))
                dt = float(np.clip(now - t_plan_prev, 0.005, 0.25))
                t_plan_prev = now
                ap.dry_run = not engaged
                cmd = ap.step(now, dt, road, line, dets, T.speed_ms, engaged=True)
                if not engaged and tel_ok and T.speed_ms > 8.0 and cmd.info.get("lane_q", 0) >= cfg.q_min:
                    stats.add(cmd.steer, T.steer)

            # ---- diagnostic: throttle is commanded but the car is not actually accelerating ----
            # (skip this while the car is already near its target speed - steady cruise throttle looks
            # identical to "stuck" by the raw thr>0.3/no-speed-rise test, and falsely fired at 80->80 km/h
            # cruise in the field: gear=2 rpm=6357 thr=0.31, nothing wrong, just holding speed)
            below_target = T.speed_ms < cmd.info.get("v0", T.speed_ms) - 3.0
            if engaged and cmd.throttle > 0.3 and cmd.brake < 0.05 and below_target:
                if stall["since"] is None:
                    stall["since"], stall["v0"], stall["warned"] = now, T.speed_ms, False
                elif not stall["warned"] and now - stall["since"] > 2.5 and T.speed_ms - stall["v0"] < 0.8:
                    stall["warned"] = True
                    print(f"[AP] throttle {cmd.throttle:.2f} commanded for 2.5s but speed hasn't risen (gear "
                          f"{tel_mod.gear_label(T.gear)}, rpm {T.rpm}) - check: handbrake, gearbox=Automatic + Auto "
                          "Clutch=ON in AC assists, and that the virtual pad is the device actually bound in AC's "
                          "controls page  [console only, not shown on screen/recording]")
            else:
                stall["since"] = None

            # ---- supervisor + output while engaged ----
            if engaged:
                bad["tel"] = (bad["tel"] or now) if not tel_ok else None
                bad["lag"] = (bad["lag"] or now) if lag_ms > 400 else None
                if bad["tel"] and now - bad["tel"] > 0.3:
                    disengage("telemetry lost / game paused", brake=0.35, hold_s=4.0, steer0=cmd.steer)
                elif bad["lag"] and now - bad["lag"] > 0.5:
                    disengage("perception stalled", brake=0.35, hold_s=4.0, steer0=cmd.steer)
                elif cmd.fault:
                    disengage(cmd.fault, brake=0.35, hold_s=4.0, steer0=cmd.steer)
                elif pad:
                    pad.send(cmd.steer, cmd.throttle, cmd.brake)     # every frame: also feeds the watchdog
            elif pad and reader and man and now >= hold_until:
                pad.send_raw(*man)                                  # pass-through: you drive, AC sees the virtual pad

            # ---- HUD ----
            i = cmd.info
            ai_ms = perf[0] + perf[1] + perf[2]
            put(view, f"Display {disp_fps:3.0f} fps | AI {ai_ms:4.1f} ms (net {perf[1]:4.1f}) | lag {lag_ms:3.0f} ms"
                      + (" | REC" if recorder.active else ""), 26)
            mode = "AUTOPILOT" if engaged else ("SHADOW" if shadow else "MANUAL")
            col = (0, 255, 0) if engaged else (0, 200, 255)
            word = "cmd" if engaged else "plan"
            put(view, f"{mode} {cmd.state if engaged else ''} | set {ap.v_set * 3.6:3.0f} km/h | v {T.speed_kmh:3.0f} km/h"
                      f" | {word} steer {cmd.steer:+.2f} thr {cmd.throttle:.2f} brk {cmd.brake:.2f}", 52, col)
            lead = i.get("lead_z")
            put(view, f"lane q {i.get('lane_q', 0):.2f} w {i.get('lane_w', 0):.1f} m off {i.get('c0', 0):+.2f} m"
                      f" | L/R lane {'Y' if i.get('left') else '-'}/{'Y' if i.get('right') else '-'}"
                      f" | lead {'%.0f m' % lead if lead else '--'} | pitch {bev.pitch_deg:+.1f} xoff {bev.x_offset_m:+.2f}"
                      + ("" if tel_ok else f" | NO TELEMETRY ({STATUS_NAMES.get(T.status, T.status)})"), 78)
            y = 104
            if not engaged:
                if shadow:
                    put(view, f"driver steer {T.steer:+.2f} (AC)  vs  plan {cmd.steer:+.2f}", y); y += 26
                est = stats.estimate(cfg.stick_lock_rad)
                if est and est[2]:
                    put(view, f"shadow: corr {est[0]:+.2f}, driver/plan gain {est[1]:.2f} -> rough --lock hint {est[2]:.2f}", y); y += 26
                sug = i.get("suggest", 0)
                if sug:
                    put(view, f"would change lane {'RIGHT' if sug > 0 else 'LEFT'}", y, (0, 255, 255)); y += 26
            if i.get("lc_reason"):
                put(view, f"lane change waiting: {i['lc_reason']}", y, (0, 200, 255)); y += 26
            if cmd.fault:
                put(view, f"planner: {cmd.fault}", y, (0, 0, 255)); y += 26
            if now < notice_until:
                put(view, notice, y, (0, 0, 255))
            draw_steer_bar(view, cmd.steer, driver=T.steer if (tel_ok and not engaged) else None)

            bev_img = bev.render(disp, res, det)
            draw_plan_on_bev(bev_img, ap)
            combined = np.hstack((view, bev_img))
            cv2.imshow(v4.WINDOW_NAME, combined)

            if now - t_status >= 2.0:
                t_status = now
                print(f"[STATUS] {mode} {cmd.state if engaged else '':11s} | telemetry={STATUS_NAMES.get(T.status, T.status)} "
                      f"fresh={int(T.alive)} v={T.speed_kmh:3.0f}->{i.get('v0', 0) * 3.6:3.0f} km/h gear={tel_mod.gear_label(T.gear)} "
                      f"rpm={T.rpm:5d} | lane q={i.get('lane_q', 0):.2f} w={i.get('lane_w', 0):.1f} "
                      f"L/R={'Y' if i.get('left') else '-'}/{'Y' if i.get('right') else '-'} kappa={i.get('kappa', 0)*1000:+.2f}e-3 | "
                      f"steer={cmd.steer:+.2f} thr={cmd.throttle:.2f} brk={cmd.brake:.2f} a_des={i.get('a_des', 0):+.2f} | "
                      f"lead={('%.0fm' % i['lead_z']) if i.get('lead_z') else '--'}"
                      + (f" | lc: {i['lc_reason']}" if i.get("lc_reason") else "")
                      + f" | pitch={bev.pitch_deg:+.1f}(base{bev.base_pitch:+.1f}) xoff={bev.x_offset_m:+.2f}"
                      + f" | lag={lag_ms:.0f} ms")

            if need_exclude:
                ok_ex = v4.exclude_window_from_capture(v4.WINDOW_NAME)
                print("[INFO] Window excluded from screen capture." if ok_ex else
                      "[WARN] Could not exclude window from capture; use a 2nd monitor or CAPTURE_REGION.")
                need_exclude = False
            if recorder.active and now - t_rec >= 1.0 / v4.REC_FPS:
                recorder.push(combined)
                t_rec = now
            if not v4.handle_key(cv2.waitKey(1) & 0xFF, bev, recorder):
                break
    finally:
        stop.set()
        try:
            if (bev.base_pitch, bev.vfov_deg, bev.height_m, bev.x_offset_m) != cal0:
                save_cal(bev)                                   # keeps changes made with the [ ] - = , . keys or Z
        except Exception:
            pass
        if pad:
            pad.close()
        worker.join(timeout=2.0)
        cam.stop()
        recorder.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
