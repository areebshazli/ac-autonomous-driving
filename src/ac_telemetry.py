"""
Assetto Corsa telemetry via shared memory (Windows).

AC publishes three named memory blocks while a session is running:
    acpmf_physics   ~333 Hz   speed, pedals, steering, gear, G-forces, ...
    acpmf_graphics  per frame status (OFF/REPLAY/LIVE/PAUSE), lap info, ...
    acpmf_static    once      car / track / player names


Run this file directly for a live 10 Hz printout.
"""
import ctypes
import mmap
import sys
import time
from dataclasses import dataclass

c_int, c_float, c_u16 = ctypes.c_int32, ctypes.c_float, ctypes.c_uint16   # c_u16 = wchar_t on Windows

AC_OFF, AC_REPLAY, AC_LIVE, AC_PAUSE = 0, 1, 2, 3
STATUS_NAMES = {AC_OFF: "OFF", AC_REPLAY: "REPLAY", AC_LIVE: "LIVE", AC_PAUSE: "PAUSE"}


class SPageFilePhysics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int), ("gas", c_float), ("brake", c_float), ("fuel", c_float),
        ("gear", c_int), ("rpms", c_int), ("steerAngle", c_float), ("speedKmh", c_float),
        ("velocity", c_float * 3), ("accG", c_float * 3),
        ("wheelSlip", c_float * 4), ("wheelLoad", c_float * 4), ("wheelsPressure", c_float * 4),
        ("wheelAngularSpeed", c_float * 4), ("tyreWear", c_float * 4), ("tyreDirtyLevel", c_float * 4),
        ("tyreCoreTemperature", c_float * 4), ("camberRAD", c_float * 4), ("suspensionTravel", c_float * 4),
        ("drs", c_float), ("tc", c_float), ("heading", c_float), ("pitch", c_float), ("roll", c_float),
    ]


class SPageFileGraphic(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("packetId", c_int), ("status", c_int), ("session", c_int)]


class SPageFileStatic(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("smVersion", c_u16 * 15), ("acVersion", c_u16 * 15),
        ("numberOfSessions", c_int), ("numCars", c_int),
        ("carModel", c_u16 * 33), ("track", c_u16 * 33),
    ]


def _wstr(arr):
    chars = []
    for c in arr:
        if c == 0:
            break
        chars.append(chr(c))
    return "".join(chars)


def _open_map(name, size):
    if sys.platform == "win32":
        return mmap.mmap(-1, size, tagname=name)      # opens AC's existing mapping if it exists
    return mmap.mmap(-1, size)                        # non-Windows: dummy (tests only)


def gear_label(g):
    """AC reports 0 = reverse, 1 = neutral, 2 = 1st gear ... -> 'R', 'N', '1', '2' ..."""
    return "R" if g == 0 else "N" if g == 1 else str(g - 1)


@dataclass
class Telemetry:
    alive: bool            # physics packets are arriving
    status: int            # AC_OFF / AC_REPLAY / AC_LIVE / AC_PAUSE
    speed_kmh: float
    speed_ms: float
    gas: float             # what AC currently sees as throttle (0..1)
    brake: float
    steer: float           # raw steerAngle from AC (units: verify with `ac_control.py --calibrate`)
    gear: int              # 0 = R, 1 = N, 2 = 1st ... (AC convention)
    rpm: int
    acc_lat_g: float       # accG[0]
    acc_lon_g: float       # accG[2]
    heading: float         # radians
    packet_id: int


class ACTelemetry:
    def __init__(self):
        self._phys_map = _open_map("acpmf_physics", ctypes.sizeof(SPageFilePhysics))
        self._gfx_map = _open_map("acpmf_graphics", ctypes.sizeof(SPageFileGraphic))
        self._stat_map = _open_map("acpmf_static", ctypes.sizeof(SPageFileStatic))
        self._last_pid = -1
        self._last_change = 0.0

    def read(self):
        p = SPageFilePhysics.from_buffer_copy(self._phys_map)      # snapshot copy = consistent values
        g = SPageFileGraphic.from_buffer_copy(self._gfx_map)
        now = time.monotonic()
        if p.packetId != self._last_pid:
            self._last_pid, self._last_change = p.packetId, now
        alive = (now - self._last_change) < 0.4 and p.packetId != 0
        v = float(p.speedKmh)
        return Telemetry(alive, int(g.status), v, v / 3.6, float(p.gas), float(p.brake),
                         float(p.steerAngle), int(p.gear), int(p.rpms),
                         float(p.accG[0]), float(p.accG[2]), float(p.heading), int(p.packetId))

    def session_info(self):
        s = SPageFileStatic.from_buffer_copy(self._stat_map)
        return _wstr(s.carModel), _wstr(s.track)

    def close(self):
        for m in (self._phys_map, self._gfx_map, self._stat_map):
            try:
                m.close()
            except Exception:
                pass


if __name__ == "__main__":
    tel = ACTelemetry()
    print("Waiting for Assetto Corsa (start a session, get in the car)...  Ctrl+C to quit")
    named = False
    try:
        while True:
            t = tel.read()
            if t.alive and not named:
                car, track = tel.session_info()
                print(f"[INFO] Connected: car='{car}' track='{track}'")
                named = True
            if not t.alive:
                print("\r[WAIT] no telemetry (is a session running?)          ", end="")
            else:
                print(f"\r{STATUS_NAMES.get(t.status, t.status):6s} | {t.speed_kmh:6.1f} km/h | "
                      f"gas {t.gas:4.2f} brk {t.brake:4.2f} | steer {t.steer:8.3f} | gear {gear_label(t.gear)} | "
                      f"rpm {t.rpm:5d} | latG {t.acc_lat_g:+5.2f} lonG {t.acc_lon_g:+5.2f}   ", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        tel.close()
