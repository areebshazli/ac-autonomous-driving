"""
Send steering / throttle / brake to Assetto Corsa through a virtual Xbox 360 controller (vgamepad + ViGEmBus).

    steering  -> left stick X   (-1 = full left, +1 = full right)
    throttle  -> right trigger  (0..1)
    brake     -> left trigger   (0..1)

One-time setup
    1. pip install vgamepad          (Windows will ask to install the ViGEmBus driver, just accept, then reboot
                                      if asked. If it does not, install ViGEmBus from its GitHub releases.)
    2. Start AC.  Options > Controls: pick the Xbox 360 controller. You'll need content manager for assigning controller I believe.
       Run   python ac_control.py --wiggle   and use AC's "assign" buttons so that
           steering = left stick X axis, throttle = right trigger, brake = left trigger.
       If AC shows ONE shared axis for both triggers instead of two separate ones, assign throttle to its
       positive half and brake to its negative half (AC combines them on some controller profiles).
    3. In the same menu set the steering filters to the minimum (speed sensitivity 0/no steering
       assist, dead-zone 0, gamma 1) so stick position maps predictably to wheel angle.
       In the assists menu turn on AUTOMATIC gearbox (and leave ABS/TC/stability on while testing).
    4. python ac_control.py --calibrate     (car parked) measures how AC reports steering vs stick.

"""
import argparse
import ctypes
import json
import math
import time
from pathlib import Path


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------------------------
# Steering linearisation.  AC applies a response curve to the stick (steering gamma etc.), so
# stick 0.25 may only give 6% steering.  The autopilot works in *normalised steering* (what AC
# reports as steerAngle, -1..1, linear in wheel angle) and this table converts it to a stick value.
# `python ac_control.py --calibrate` measures the table for your setup and saves steer_curve.json.
# ---------------------------------------------------------------------------------------------
CURVE_FILE = Path(__file__).with_name("steer_curve.json")
# stick -> steerAngle, measured on the author's default AC controller settings (Spectre, 2026-09-21)
DEFAULT_CURVE = [(0.0, 0.0), (0.25, 0.057), (0.5, 0.258), (0.75, 0.602), (1.0, 1.0)]


def load_curve():
    try:
        pts = json.loads(CURVE_FILE.read_text())["curve"]
        return [(float(a), float(b)) for a, b in pts]
    except Exception:
        return list(DEFAULT_CURVE)


def build_curve(rows):
    """rows: [(stick, steerAngle), ...] with stick >= 0 -> clean, strictly increasing curve starting at (0, 0)."""
    pts = {0.0: 0.0}
    for s, v in rows:
        if s > 0:
            pts[round(float(s), 3)] = max(float(v), 0.0)
    xs = sorted(pts)
    out, top = [], 0.0
    for x in xs:                                    # enforce monotonic increase, normalise to full stick = 1.0
        top = max(top, pts[x])
        out.append((x, top))
    full = out[-1][1]
    if full > 1e-3:
        out = [(x, y / full) for x, y in out]
    return out


def stick_for(target, curve):
    """Inverse of the curve: which stick value produces normalised steering `target` (-1..1)?"""
    t = _clamp(float(target), -1.0, 1.0)
    a = abs(t)
    xs, ys = [p[0] for p in curve], [p[1] for p in curve]
    if a >= ys[-1]:
        x = xs[-1]
    else:
        x = xs[0]
        for i in range(1, len(ys)):
            if a <= ys[i]:
                span = ys[i] - ys[i - 1]
                x = xs[i - 1] + (xs[i] - xs[i - 1]) * ((a - ys[i - 1]) / span if span > 1e-9 else 0.0)
                break
    return math.copysign(x, t) if t else 0.0


class VirtualPad:
    def __init__(self, curve=None):
        import vgamepad as vg                      # imported here so other tools work without it
        self._vg = vg
        self.curve = curve or load_curve()
        self.pad = vg.VX360Gamepad()
        # Some games only "see" the pad after a first button event.
        self.pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.pad.update()
        time.sleep(0.1)
        self.pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.pad.update()
        self.neutral()

    def send(self, steer, throttle, brake):
        """steer = NORMALISED steering target (-1..1, linear); it is converted to a stick value with the curve."""
        self.send_raw(stick_for(steer, self.curve), throttle, brake)

    def send_raw(self, stick, throttle, brake):
        """Raw pad values (used for pass-through of a physical controller: no curve applied)."""
        self.pad.left_joystick_float(x_value_float=_clamp(float(stick), -1.0, 1.0), y_value_float=0.0)
        self.pad.right_trigger_float(value_float=_clamp(float(throttle), 0.0, 1.0))
        self.pad.left_trigger_float(value_float=_clamp(float(brake), 0.0, 1.0))
        self.pad.update()

    def neutral(self):
        self.pad.reset()
        self.pad.update()


class SafePad(VirtualPad):
    """VirtualPad + dead-man watchdog.

    The virtual stick keeps its last position forever. If the control loop stalls (a blocked capture call,
    a crash inside a callback, a frozen window...) the car would keep steering/accelerating on its own.
    A background thread therefore releases steering + throttle and applies `timeout_brake` whenever
    send() has not been called for `timeout` seconds. Normal sending resumes automatically.
    The watchdog only guards ACTIVE control: neutral() (a deliberate hand-off) disarms it.
    """

    def __init__(self, timeout=0.25, timeout_brake=0.3, curve=None):
        import threading
        super().__init__(curve)
        self.timeout, self.timeout_brake = timeout, timeout_brake
        self.tripped = False
        self._armed = False                     # armed by the first send(); disarmed by neutral()
        self._last = time.monotonic()
        self._lock = threading.RLock()
        self._alive = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def send_raw(self, stick, throttle, brake):
        with self._lock:
            self._last = time.monotonic()
            self._armed, self.tripped = True, False
            super().send_raw(stick, throttle, brake)

    def neutral(self):
        """Deliberate release (hand-off): disarms the watchdog so it does not read the silence as a stall."""
        self._armed, self.tripped = False, False
        super().neutral()

    def _watch(self):
        while self._alive:
            time.sleep(0.05)
            with self._lock:
                if self._armed and not self.tripped and time.monotonic() - self._last > self.timeout:
                    VirtualPad.send_raw(self, 0.0, 0.0, self.timeout_brake)
                    self.tripped = True

    def close(self):
        self._alive = False
        with self._lock:
            self.neutral()


# ---------------------------------------------------------------------------------------------------
# Optional: read a PHYSICAL Xbox-compatible controller (XInput) you can only use one input at a time. 
# while AC only ever sees the virtual pad.  Windows only. (DualShock/DualSense pads need DS4Windows.)
# ---------------------------------------------------------------------------------------------------
class _XIState(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("buttons", ctypes.c_uint16), ("lt", ctypes.c_ubyte),
                ("rt", ctypes.c_ubyte), ("lx", ctypes.c_short), ("ly", ctypes.c_short),
                ("rx", ctypes.c_short), ("ry", ctypes.c_short)]


class XInputReader:
    def __init__(self, index):
        self.index = index
        self._get = _load_xinput()

    @staticmethod
    def connected():
        get = _load_xinput()
        if get is None:
            return []
        st = _XIState()
        return [i for i in range(4) if get(i, ctypes.byref(st)) == 0]

    def read(self):
        """-> (steer -1..1, throttle 0..1, brake 0..1) or None if the pad is unplugged."""
        st = _XIState()
        if self._get is None or self._get(self.index, ctypes.byref(st)) != 0:
            return None
        lx = st.lx / 32767.0
        lx = 0.0 if abs(lx) < 0.08 else lx
        return _clamp(lx, -1.0, 1.0), st.rt / 255.0, st.lt / 255.0


def _load_xinput():
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            return getattr(ctypes.windll, name).XInputGetState
        except Exception:
            continue
    return None


def wiggle():
    """Sweep every input slowly so AC's control-assignment screen can detect them."""
    pad = VirtualPad()
    print("Wiggling steering, then throttle, then brake (repeats). Ctrl+C to stop.")
    try:
        while True:
            for name in ("steer", "throttle", "brake"):
                print(f"  -> {name}")
                t0 = time.time()
                while time.time() - t0 < 4.0:
                    x = math.sin((time.time() - t0) * math.pi)           # slow -1..1..-1 sweep
                    if name == "steer":
                        pad.send(x, 0, 0)
                    elif name == "throttle":
                        pad.send(0, abs(x), 0)
                    else:
                        pad.send(0, 0, abs(x))
                    time.sleep(0.01)
                pad.neutral()
                time.sleep(0.7)
    except KeyboardInterrupt:
        pass
    finally:
        pad.neutral()


def calibrate():
    """Park the car (handbrake on / neutral), be in the car in a LIVE session. Steps the stick from 0 to full,
    records what AC reports as steerAngle and saves steer_curve.json (used automatically from then on)."""
    from ac_telemetry import ACTelemetry, AC_LIVE
    tel = ACTelemetry()
    t = tel.read()
    if not t.alive or t.status != AC_LIVE:
        print("AC is not live. Start a session and sit in the car, then rerun.")
        return
    pad = VirtualPad(curve=[(0.0, 0.0), (1.0, 1.0)])          # raw sticks here, no correction while measuring
    print("Stick -> AC steerAngle (car should stay parked; the wheel will visibly turn)")
    rows = []
    try:
        for s in [0.0] + [round(0.1 * k, 1) for k in range(1, 11)]:
            pad.send_raw(s, 0, 0)
            time.sleep(1.3)
            v = tel.read().steer
            rows.append((s, v))
            print(f"  stick {s:+.1f}  ->  steerAngle {v:+.3f}")
        for s in (-0.5, -1.0):                                # symmetry check
            pad.send_raw(s, 0, 0)
            time.sleep(1.3)
            print(f"  stick {s:+.1f}  ->  steerAngle {tel.read().steer:+.3f}   (left side check)")
    finally:
        pad.neutral()
    curve = build_curve(rows)
    CURVE_FILE.write_text(json.dumps({"curve": curve, "note": "stick -> normalised steerAngle, measured parked"}, indent=1))
    mid = [(x, y) for x, y in curve if 0.15 < x < 0.95 and y > 0.01]
    if mid:
        gam = sum(math.log(y) / math.log(x) for x, y in mid) / len(mid)
        print(f"\nResponse is roughly stick^{gam:.2f} (1.0 = linear).")
    print(f"Saved {CURVE_FILE.name}: the autopilot now converts its steering target to the stick value that gives it.")
    print("(Speed sensitivity in AC's controller settings also changes steering at speed - keep it at 0.)")


def pad_test():
    """Shows which XInput slots have a controller and prints live values - for the pass-through option."""
    idx = XInputReader.connected()
    print("XInput controllers found in slots:", idx or "none")
    if not idx:
        return
    r = XInputReader(idx[0])
    print(f"Reading slot {idx[0]}. Move the left stick / press triggers. Ctrl+C to stop.")
    try:
        while True:
            print("\r  steer %+.2f  throttle %.2f  brake %.2f   " % (r.read() or (0, 0, 0)), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiggle", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pad-test", action="store_true", help="show the physical XInput controller values")
    a = ap.parse_args()
    if a.wiggle:
        wiggle()
    elif a.calibrate:
        calibrate()
    elif a.pad_test:
        pad_test()
    else:
        ap.print_help()
