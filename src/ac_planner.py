"""
ac_planner.py - driving logic that sits on top of the perception BEV.

    BEV masks (road, line) --LaneTracker--> lane model  x_c(z) = a0 + a1*z + a2*z^2   (metres, ego frame)
    detections (X, Z)      --VehicleTracker--> tracks with relative speed (+ blind-zone memory)
    lane model + tracks    --Autopilot--> steering / throttle / brake + lane-change decisions

Coordinates everywhere: ego frame, x = metres to the RIGHT, z = metres FORWARD from the camera.
Steering: +1 = full right (XInput left-stick X).

Pure numpy - no game, GPU or gamepad needed, so it can be tested offline (see sim_test.py).
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

KEEP, LANE_CHANGE = "KEEP", "LANE_CHANGE"


@dataclass
class PlannerCfg:
    # --- BEV geometry (must match BirdsEyeView: 360x720 canvas, 50 m forward) ---
    bev_w: int = 360
    bev_h: int = 720
    ppm: float = 13.8
    ego_y: int = 690
    # --- lane extraction ---
    z_min: float = 4.0
    z_max: float = 32.0
    z_step: float = 0.75
    z_ref: float = 5.0              # where lane offset is read out
    lane_w_nom: float = 3.5
    lane_w_min: float = 2.4
    lane_w_max: float = 4.8
    merge_m: float = 0.6
    chain_gate_m: float = 1.2
    min_rows: int = 8
    coeff_alpha: float = 0.55       # temporal smoothing of the lane polynomial
    lock_on_max_offset_m: float = 6.0   # first lock-on picks the CLOSEST lane segment within this far off to
                                        # either side, rather than requiring the car to already be inside one -
                                        # lets it find a lane from the shoulder, a tilted start, etc.
    q_min: float = 0.35             # below this the lane model is "unhealthy"
    health_enter_s: float = 0.15    # consecutive GOOD time needed to (re)gain control - debounces flicker
    health_exit_s: float = 0.30     # consecutive BAD time needed to lose control - debounces flicker
    lane_lost_s: float = 1.0        # unhealthy this long -> fault (independent of the debounce above)
    fault_steer_tau_s: float = 10.0 # after a fault, steering decays toward straight with this time constant
                                    # instead of snapping straight (0 = snap). On a curve, snapping straight
                                    # drives the car off the outside of the bend while it brakes.
    width_jump_penalty: float = 0.6 # quality multiplier when width jumps > width_jump_m from its running average
    width_jump_m: float = 0.9       # a bigger single-frame width change than this usually means a bad row
    near_consistency_m: float = 0.5     # if the fitted curve disagrees with the NEAREST actual row by more than
                                        # this, the far field is probably dragging the fit somewhere it shouldn't
                                        # be (a barrier/guardrail/shadow far ahead) - the near field is the most
                                        # reliable data we have (least perspective distortion, least likely
                                        # occluded), so a fit that doesn't match it gets penalised hard
    near_consistency_penalty: float = 0.4
    ego_confirm_s: float = 0.35     # a car must sit in OUR corridor this long before it counts as a lead
    ego_release_s: float = 0.15     # ...and this long absent before it stops counting (faster to forgive)
    # --- car ---
    wheelbase: float = 2.87         # CLS 63 AMG (W218)
    stick_lock_rad: float = 0.45    # front-wheel angle (rad) at normalised steering +-1 = full lock - calibrate
                                    # per vehicle (a CLS at full lock is ~0.5-0.6 rad); the controller tolerates
                                    # 0.7x-1.4x error in this value without becoming unstable (see sim_test.py S3)
    steer_slew: float = 3.0         # max stick change per second
    cam_to_bumper: float = 1.5      # camera is ~1.5 m behind the front bumper
    # --- lateral: pure pursuit ---
    la_base: float = 6.0            # lookahead L = base + time * speed  (long enough to stay damped
    la_time: float = 1.0            # despite steering lag + camera latency; found by sweeping in sim)
    la_min: float = 8.0
    la_max: float = 30.0
    # --- longitudinal: IDM + PI ---
    a_max: float = 1.5
    b_comf: float = 2.5
    b_max: float = 6.0
    s0: float = 5.0
    time_gap: float = 1.6
    a_lat_max: float = 4.0          # cornering speed limit (m/s^2)
    kappa_anticip_s: float = 0.7    # a tightening bend's curvature estimate lags the true value while the lane
                                    # fit is still catching up (see sim_test.py S11); extrapolating the curvature
                                    # forward by this many seconds (using its recent rate of change) makes the
                                    # cornering speed limit below brake for the bend it is ENTERING, not the
                                    # shallower one the fit measured half a second ago. 0 = no anticipation.
    kappa_rate_tau_s: float = 0.3   # smooths the curvature-rate estimate itself (raw frame-to-frame curvature
                                    # noise would otherwise make the anticipated value jumpier than the real one)
    kp: float = 0.25
    ki: float = 0.05
    tau: float = 1.0
    # --- tracking ---
    tr_alpha: float = 0.3           # position correction gain - can stay fast, position noise is forgiving
    tr_beta: float = 0.01           # velocity correction gain (standard g-h filter: v += (beta/dt)*residual).
                                    # Was 0.05: at 25Hz that's an effective gain of 1.25 per metre of position
                                    # noise - a single noisy detection could swing the estimated lead speed by
                                    # several m/s, which the IDM's closing-speed term amplifies further, producing
                                    # full panic-brake commands for a car sitting at a rock-steady, safe gap.
                                    # 0.01 eliminates that in testing while reacting to a genuine sudden hard
                                    # brake from a lead just as fast (~0.16s) as the old value did.
    coast_z: float = 14.0           # vehicles lost closer than this while we close in -> blind-zone memory
    # --- lane change ---
    lc_time: float = 3.5
    lc_min_speed: float = 7.0       # m/s (~25 km/h) - must be low enough to overtake slow traffic
    lc_max_kappa: float = 1 / 150.0
    lc_min_quality: float = 0.6
    block_dv: float = 2.0           # lead must be this much slower than v_set to count as blocking
    block_confirm_s: float = 0.6
    nb_confirm_s: float = 0.5
    min_gain: float = 1.5           # m/s speed potential gain needed to change lane
    prefer_side: int = 1            # +1 = overtake on the right (Japan: keep left, pass right)
    side_bias: float = 0.5
    side_back: float = 8.0          # target lane must be free from -side_back ...
    side_front: float = 6.0         # ... to +side_front (blind zone / alongside)
    coast_margin_back: float = 10.0 # extra exclusion margin cap for a long-unseen (dead-reckoned) car - was 6.0:
    coast_margin_front: float = 5.0 # a 9.5s-old estimate can drift several metres, so give it more room
    coast_unsafe_s: float = 7.0     # a coasting track older than this is too stale to clear FOR A LANE CHANGE -
                                    # a growing margin against a hard boundary can still be beaten by bad luck at
                                    # the exact edge, so past this age we just refuse to use that side at all
    gap_base: float = 8.0
    gap_time: float = 1.0
    ttc_min: float = 5.0
    cooldown: float = 6.0
    request_timeout_s: float = 5.0  # a manual request gives up after this long of never being safe/possible


@dataclass
class Command:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    state: str = "OFF"
    fault: str = ""
    info: dict = field(default_factory=dict)


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def steer_bleed(steer, dt, tau_s):
    """Fault fallback: fade the last steering command toward straight with time constant tau_s rather than
    snapping it to zero. Shared by the planner and by v5's post-fault brake-hold so both behave the same."""
    return 0.0 if tau_s <= 0 else steer * math.exp(-dt / tau_s)


def quintic(s):
    return s * s * s * (10 + s * (-15 + 6 * s))      # 0->1 with zero velocity/accel at both ends


# =============================== BEV -> masks ===============================
def occlude_clamp(X, Z, reliable, cam_to_bumper):
    """Detections whose distance is only a floor (occluded by the hood, see v4.hood_floor_z) get clamped to
    'basically touching' rather than trusted, so the planner brakes hard instead of thinking the gap is safe.

    But that ONLY applies when the raw projected Z was positive - meaning the box's bottom edge really did land
    on the ground plane in front of the car, just too close to trust precisely. A box whose bottom sits at or
    above the horizon (a false detection on a distant building, sign, or the skyline - not a real nearby car)
    inverts through the same ground-plane maths to a NEGATIVE Z, not a small positive one. Clamping that to
    'basically touching' turns a false detection of something 50+ metres away, or not on the road at all, into
    a phantom car sitting on the bumper - a real observed bug (full panic-brake with nothing in front of the
    car). Those are left alone here and get discarded by the normal Z > 0.1 sanity filter downstream instead.
    """
    Z = Z.copy()
    occluded_near = ~reliable & (Z > 0)       # genuinely close, but the hood cuts off the true bottom edge
    Z[occluded_near] = cam_to_bumper * 0.5
    return X, Z


def bev_masks(warped_bgr, close_px=9):
    """Warped (nearest-neighbour) overlay -> boolean road / lane-line masks.
    Overlay colours: green = drivable, red = lane line (BGR). Small holes in the road are closed."""
    line = warped_bgr[..., 2] > 0
    road = (warped_bgr[..., 1] > 0) | line
    if close_px:
        k = np.ones((close_px, close_px), np.uint8)
        road = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    return road, line


# ============================== lane extraction =============================
def _runs(m):
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def row_segments(road_row, line_row, cfg):
    """Split one BEV row into lane segments [(x_left_m, x_right_m), ...] (metres, +x right)."""
    rs, re = _runs(road_row)
    if rs.size == 0:
        return []
    ls, le = _runs(line_row)
    pts = sorted(list(rs) + list(re) + [(s + e) / 2 for s, e in zip(ls, le)])
    merge_px = cfg.merge_m * cfg.ppm
    groups = []
    for p in pts:                                    # merge boundaries closer than merge_m
        if groups and p - groups[-1][-1] < merge_px:
            groups[-1].append(p)
        else:
            groups.append([p])
    bs = [float(np.mean(g)) for g in groups]
    cx = cfg.bev_w / 2
    segs = []
    for a, b in zip(bs[:-1], bs[1:]):
        w = (b - a) / cfg.ppm
        if w < cfg.lane_w_min or w > cfg.lane_w_max:
            continue
        if road_row[int(a):int(b)].mean() < 0.75:    # gap between two road runs, not a lane
            continue
        segs.append(((a - cx) / cfg.ppm, (b - cx) / cfg.ppm))
    return segs


@dataclass
class LaneModel:
    valid: bool = False
    coeffs: np.ndarray = None       # a0, a1, a2  (x = a0 + a1 z + a2 z^2)
    width: float = 3.5
    quality: float = 0.0
    left_ok: bool = False           # drivable lane exists to the left / right
    right_ok: bool = False
    c0: float = 0.0                 # lane-centre lateral position at z_ref
    flip: int = 0                   # +1/-1 when the ego lane identity jumped one lane right/left
    z_hi: float = 0.0

    def x_at(self, z):
        a0, a1, a2 = self.coeffs
        if z <= self.z_hi:
            return a0 + a1 * z + a2 * z * z
        return (a0 + a1 * self.z_hi + a2 * self.z_hi ** 2) + (a1 + 2 * a2 * self.z_hi) * (z - self.z_hi)

    @property
    def curvature(self):
        if self.coeffs is None:
            return 0.0
        a1 = self.coeffs[1]
        return 2 * self.coeffs[2] / (1 + a1 * a1) ** 1.5


class LaneTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.z_rows = np.arange(cfg.z_min, cfg.z_max, cfg.z_step)
        self.prev_coeffs = None
        self.prev_c0 = None
        self.prev_z_hi = None
        self.width = cfg.lane_w_nom
        self.miss = 0
        self.last_rows = (np.zeros(0), np.zeros(0), np.zeros(0))

    def _extract(self, road, line):
        cfg = self.cfg
        zs, cs, ws, lo, ro = [], [], [], [], []
        prev_c = None
        # prev_c above is LOCAL to this one call (chains rows within a single frame, nearest to farthest).
        # cold_start reflects the TRACKER's cross-frame history instead - whether we have ever locked onto a
        # lane before at all. The relaxed nearest-segment search below must only apply on a true cold start
        # (e.g. right after startup, or re-acquiring after a long loss) - not on every frame's nearest row,
        # which needs to stay anchored strictly at the ego position or it can mis-anchor onto a neighbouring
        # lane under ordinary noise/curvature and quietly corrupt otherwise-normal tracking.
        cold_start = self.prev_c0 is None
        for z in self.z_rows:
            y = int(round(cfg.ego_y - z * cfg.ppm))
            if y < 1 or y >= road.shape[0] - 1:
                continue
            rr = road[y - 1:y + 2].sum(0) >= 2
            ll = line[y - 1:y + 2].any(0)
            segs = row_segments(rr, ll, cfg)
            if not segs:
                if prev_c is None and z > 8.0:
                    return None
                continue
            if prev_c is None:
                if cold_start:
                    # True cold start (no lane history at all): don't require the car to already be centred
                    # in a lane - pick the closest segment instead, so a car parked on the shoulder, tilted,
                    # or not yet aligned can still find and steer toward a lane ahead, not just one it's
                    # already sitting in. Capped so it won't lock onto something implausibly far to the side.
                    if segs:
                        centers = [(a + b) / 2 for a, b in segs]
                        i = int(np.argmin([abs(c) for c in centers]))
                        if abs(centers[i]) > cfg.lock_on_max_offset_m:
                            i = None
                    else:
                        i = None
                else:
                    # Normal frame-to-frame tracking's nearest row: keep the strict anchor at the ego
                    # position. We already know roughly where we are (self.prev_c0) - a nearest-segment
                    # search here could mis-anchor onto an adjacent lane under ordinary noise or curvature.
                    cand = [i for i, (a, b) in enumerate(segs) if a <= 0.0 <= b]
                    i = cand[0] if cand else None
                if i is None:
                    if z > 8.0:
                        return None
                    continue
            else:
                d = [abs((a + b) / 2 - prev_c) for a, b in segs]
                i = int(np.argmin(d))
                if d[i] > cfg.chain_gate_m:
                    continue
            a, b = segs[i]
            prev_c = (a + b) / 2
            zs.append(z); cs.append(prev_c); ws.append(b - a)
            lo.append(i > 0 and abs(segs[i - 1][1] - a) < 0.9)
            ro.append(i < len(segs) - 1 and abs(segs[i + 1][0] - b) < 0.9)
        self.last_rows = (np.array(zs), np.array(ws), np.array(cs))     # per-row lane data (used by BEV auto-calibration)
        if len(zs) < cfg.min_rows or (zs[-1] - zs[0]) < 8.0:
            return None
        Z, C = np.array(zs), np.array(cs)
        w = 1.0 / (1.0 + Z / 15.0)
        p = np.polyfit(Z, C, 2, w=w)
        res = C - np.polyval(p, Z)
        keep = np.abs(res) < max(0.4, 2.5 * res.std())
        if keep.sum() >= cfg.min_rows and not keep.all():
            p = np.polyfit(Z[keep], C[keep], 2, w=w[keep])
            res = C[keep] - np.polyval(p, Z[keep])
        quality = (len(zs) / len(self.z_rows)) * max(0.0, 1.0 - float(np.sqrt((res ** 2).mean())) / 0.6)
        if abs(float(np.median(ws)) - self.width) > cfg.width_jump_m:      # a sudden width jump = a contaminated
            quality *= cfg.width_jump_penalty                             # row (text, shadow, guardrail...)
        near_resid = abs(np.polyval(p, Z[0]) - C[0])                      # does the fit still agree with the
        if near_resid > cfg.near_consistency_m:                          # single most-trustworthy (nearest) row?
            quality *= cfg.near_consistency_penalty
        mid = [(z, l, r) for z, l, r in zip(zs, lo, ro) if 6.0 <= z <= 24.0]
        left_ok = bool(mid) and np.mean([m[1] for m in mid]) >= 0.5
        right_ok = bool(mid) and np.mean([m[2] for m in mid]) >= 0.5
        return p[::-1], float(np.median(ws)), quality, left_ok, right_ok, float(Z[-1])

    def update(self, road, line):
        cfg = self.cfg
        raw = self._extract(road, line)
        if raw is None:
            self.miss += 1
            if self.miss > 10:
                self.prev_coeffs = self.prev_c0 = self.prev_z_hi = None
            if self.prev_coeffs is not None:
                # Hold over the last known lane shape. valid=False (this frame found nothing - used for the
                # quality/health-debounce logic), but coeffs/width/z_hi are NOT None, so a caller relying on the
                # debounced 'healthy' flag (which can stay true for a short grace period through a miss) still
                # gets a usable, if stale, geometry instead of crashing on lane.x_at(None).
                return LaneModel(False, self.prev_coeffs, self.width, 0.0, False, False,
                                  self.prev_c0 or 0.0, 0, self.prev_z_hi)
            return LaneModel(valid=False, width=self.width)
        self.miss = 0
        coeffs, width, quality, left_ok, right_ok, z_hi = raw
        self.width = 0.9 * self.width + 0.1 * clip(width, cfg.lane_w_min, cfg.lane_w_max)
        c0_raw = coeffs[0] + coeffs[1] * cfg.z_ref + coeffs[2] * cfg.z_ref ** 2
        flip = 0
        if self.prev_c0 is not None:
            jump = c0_raw - self.prev_c0
            if abs(jump) > 0.6 * self.width:
                flip = 1 if jump > 0 else -1          # centre jumped right => we entered the right lane
        if self.prev_coeffs is None or flip:
            sm = coeffs
        else:
            sm = cfg.coeff_alpha * coeffs + (1 - cfg.coeff_alpha) * self.prev_coeffs
        self.prev_coeffs, self.prev_c0, self.prev_z_hi = sm, c0_raw, z_hi
        c0 = sm[0] + sm[1] * cfg.z_ref + sm[2] * cfg.z_ref ** 2
        return LaneModel(True, sm, self.width, quality, left_ok, right_ok, c0, flip, z_hi)


# ============================== vehicle tracking ============================
@dataclass
class Track:
    id: int
    x: float
    z: float
    vx: float = 0.0
    vz: float = 0.0                 # relative longitudinal speed = va - v_ego (recomputed every update)
    va: float = 0.0                 # ABSOLUTE longitudinal speed - other cars keep this while we brake/accelerate
    hits: int = 1
    since_seen: float = 0.0
    coasting: bool = False

    @property
    def confirmed(self):
        return self.hits >= 3 or self.coasting


class VehicleTracker:
    """Alpha-beta filter per vehicle. vz = relative longitudinal speed (lead speed = v_ego + vz).
    Vehicles that vanish close to the car while we are closing in are remembered ('coasting')
    because the hood camera cannot see cars that are alongside us."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.next_id = 1
        self.t_prev = None

    @staticmethod
    def _memory_s(tr):
        """How long to remember a car that vanished beside us: as long as it should still be there.
        Slow relative speed => it stays alongside for a long time (capped at 30 s)."""
        return clip(25.0 / max(abs(tr.vz), 0.5), 12.0, 30.0)

    def update(self, t, dets, v_ego):
        cfg = self.cfg
        dt = 0.05 if self.t_prev is None else clip(t - self.t_prev, 1e-3, 0.25)
        self.t_prev = t
        for tr in self.tracks:
            if not tr.coasting:
                tr.x += tr.vx * dt                    # remembered cars: keep lateral position frozen
            tr.z += (tr.va - v_ego) * dt              # model: other car holds its own speed
            tr.since_seen += dt
        dets = np.asarray(dets, float).reshape(-1, 2)
        pairs = []
        for i, tr in enumerate(self.tracks):
            gz = max(4.0, 0.2 * max(tr.z, 0.0))
            for j, (x, z) in enumerate(dets):
                c = ((x - tr.x) / 2.0) ** 2 + ((z - tr.z) / gz) ** 2
                if c < 1.0:
                    pairs.append((c, i, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            tr, (x, z) = self.tracks[i], dets[j]
            rx, rz = x - tr.x, z - tr.z
            tr.x += cfg.tr_alpha * rx
            tr.z += cfg.tr_alpha * rz
            tr.vx += cfg.tr_beta / dt * rx
            tr.va += cfg.tr_beta / dt * rz
            tr.hits += 1
            tr.since_seen = 0.0
            tr.coasting = False
        for j, (x, z) in enumerate(dets):
            if j not in used_d:
                self.tracks.append(Track(self.next_id, float(x), float(z), va=v_ego))
                self.next_id += 1
        keep = []
        for tr in self.tracks:
            if tr.since_seen == 0.0:
                keep.append(tr)
            elif tr.hits >= 3 and tr.z < cfg.coast_z + 2 and tr.since_seen < self._memory_s(tr) \
                    and tr.z > -cfg.side_back - 1.0 - min(0.7 * tr.since_seen, cfg.coast_margin_back):
                tr.coasting = True          # vanished near the car: probably alongside, in the camera's blind zone
                keep.append(tr)
            elif tr.since_seen < (0.6 if tr.hits >= 3 else 0.25):
                keep.append(tr)
        self.tracks = keep
        for tr in self.tracks:
            tr.vz = tr.va - v_ego
        return [tr for tr in self.tracks if tr.confirmed]


# =========================== longitudinal controller ========================
class LongController:
    """IDM (car-following) gives a desired acceleration; a PI loop on speed turns it into pedals."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.i = 0.0

    def reset(self):
        self.i = 0.0

    def step(self, dt, v, v0, gap, v_lead):
        c = self.cfg
        v0 = max(v0, 0.5)
        a = c.a_max * (1.0 - (v / v0) ** 4)
        if gap is not None:
            s = max(gap, 0.5)
            s_star = c.s0 + max(0.0, v * c.time_gap + v * (v - v_lead) / (2 * math.sqrt(c.a_max * c.b_comf)))
            a -= c.a_max * (s_star / s) ** 2
        a = clip(a, -c.b_max, c.a_max)
        err = clip(v + a * c.tau, 0.0, max(v0, v)) - v
        u = c.kp * err + self.i
        if -0.6 < u < 1.0:
            self.i = clip(self.i + c.ki * err * dt, -0.3, 0.6)
        if u >= -0.05:
            return clip(u, 0.0, 1.0), 0.0, a
        return 0.0, clip(-u * 1.5, 0.0, 1.0), a


# ================================== autopilot ================================
class Autopilot:
    def __init__(self, cfg=None):
        self.cfg = cfg or PlannerCfg()
        self.lanes = LaneTracker(self.cfg)
        self.veh = VehicleTracker(self.cfg)
        self.long = LongController(self.cfg)
        self.v_set = 60 / 3.6
        self.dry_run = False            # True: compute everything but only *suggest* lane changes (shadow mode)
        self.suggest = 0                # dry-run: direction (-1/+1) the planner would change lane to right now
        self.reset()

    def reset(self):
        self.state = KEEP
        self.dir = 0
        self.s_lin = 0.0
        self.crossed = 0
        self.lc_t = 0.0
        self.abort = False
        self.cooldown = 0.0
        self.blocked_t = 0.0
        self.nb_t = {-1: 0.0, 1: 0.0}
        self.bad_t = 0.0
        self.good_t = 0.0
        self.healthy_state = False
        self.ego_conf = {}
        self.steer = 0.0
        self.request = 0
        self.request_age = 0.0
        self.lc_reason = ""
        self._prev_kappa = None         # for curvature-rate anticipation (see kappa_anticip_s)
        self._kappa_rate = 0.0
        self.long.reset()

    def request_lane_change(self, d):
        """Manual request: -1 = left, +1 = right. Still goes through the safety check, and now PERSISTS
        (rather than being silently dropped after one failed frame) until it succeeds or times out - see
        cfg via request_timeout_s below and self.lc_reason for why it's waiting."""
        self.request, self.request_age = d, 0.0

    # ---------- helpers ----------
    def _in_corridor(self, tr, lane, off):
        xc = lane.x_at(clip(tr.z, self.cfg.z_min, lane.z_hi)) + off
        return abs(tr.x - xc) < lane.width / 2 + 0.35

    def _lead(self, lane, tracks, off):
        c = [t for t in tracks if t.z > 0.5 and self._in_corridor(t, lane, off)]
        return min(c, key=lambda t: t.z) if c else None

    def _lane_clear(self, k_off, lane, tracks, v):
        """Is the lane at lateral offset k_off (metres from our lane centre) safe to enter?"""
        c = self.cfg
        lead = None
        for t in tracks:
            if not self._in_corridor(t, lane, k_off):
                continue
            if t.coasting and t.since_seen > c.coast_unsafe_s:
                return False, t                                   # too long since we actually saw it - don't trust it
            grow = t.since_seen if t.coasting else 0.0            # remembered cars: position uncertainty grows
            if -(c.side_back + min(0.7 * grow, c.coast_margin_back)) <= t.z < c.side_front + min(0.3 * grow, c.coast_margin_front):
                return False, t                                   # alongside / blind zone
            if t.z >= c.side_front and (lead is None or t.z < lead.z):
                lead = t
        if lead is not None:
            gap = lead.z - c.cam_to_bumper
            v_t = max(0.0, v + lead.vz)
            need = c.gap_base + c.gap_time * v + 2.0 * max(v - v_t, 0.0)
            if gap < need:
                return False, lead
            if v > v_t and gap / (v - v_t) < c.ttc_min:
                return False, lead
        return True, lead

    def _potential(self, lead, v, trigger):
        if lead is not None and lead.z < trigger:
            return min(self.v_set, max(0.0, v + lead.vz))
        return self.v_set

    # ---------- lane-change decision ----------
    def _maybe_start_lc(self, dt, lane, tracks, v):
        c = self.cfg
        manual = self.request
        self.lc_reason = ""

        # --- manual request: keep it alive across frames instead of dropping it after one bad frame ---
        if manual:
            self.request_age += dt
            if self.cooldown > 0:
                self.lc_reason = f"cooldown {self.cooldown:.1f}s"
            elif v < c.lc_min_speed:
                self.lc_reason = f"too slow ({v * 3.6:.0f} < {c.lc_min_speed * 3.6:.0f} km/h)"
            elif lane.quality < c.lc_min_quality:
                self.lc_reason = f"lane not confident enough (q {lane.quality:.2f} < {c.lc_min_quality})"
            elif abs(lane.curvature) > c.lc_max_kappa:
                self.lc_reason = "curve too tight"
            elif self.nb_t[manual] < c.nb_confirm_s:
                self.lc_reason = "confirming there is a lane there"
            else:
                clear, _ = self._lane_clear(manual * lane.width, lane, tracks, v)
                if not clear:
                    self.lc_reason = "not clear (car in the way)"
            if self.lc_reason:
                if self.request_age > c.request_timeout_s:
                    self.lc_reason = f"gave up: {self.lc_reason}"
                    self.request, self.request_age = 0, 0.0
                return                                          # still pending (or just gave up) - try again next frame

        if self.cooldown > 0 or v < c.lc_min_speed or abs(lane.curvature) > c.lc_max_kappa                 or lane.quality < c.lc_min_quality:
            self.blocked_t = max(0.0, self.blocked_t - 2 * dt)
            if not manual:
                return
        lead = self._lead(lane, tracks, 0.0)
        trigger = clip(2.2 * max(v, 0.8 * self.v_set) + 8, 20, 45)
        blocked = lead is not None and lead.z < trigger and (v + lead.vz) < self.v_set - c.block_dv
        self.blocked_t = self.blocked_t + dt if blocked else max(0.0, self.blocked_t - 2 * dt)
        if not manual and self.blocked_t < c.block_confirm_s:
            return
        pot_cur = self._potential(lead, v, trigger)
        cands = []
        for d in (-1, 1):
            if manual and d != manual:
                continue
            if self.nb_t[d] < c.nb_confirm_s:
                continue
            clear, lead_t = self._lane_clear(d * lane.width, lane, tracks, v)
            if not clear:
                continue
            gain = self._potential(lead_t, v, trigger) - pot_cur + (c.side_bias if d == c.prefer_side else 0.0)
            if manual or gain >= c.min_gain:
                cands.append((gain, d))
        if cands:
            best = max(cands)[1]
            if self.dry_run:                                   # shadow mode: report, never start the manoeuvre
                self.suggest = best
                return
            self.dir = best
            self.state, self.s_lin, self.crossed, self.lc_t, self.abort = LANE_CHANGE, 0.0, 0, 0.0, False
            self.blocked_t = 0.0
            self.request, self.request_age = 0, 0.0

    def _advance_lc(self, dt, lane, tracks, v):
        c = self.cfg
        W = lane.width
        self.lc_t += dt
        if lane.valid and lane.flip:
            self.crossed += lane.flip
        if not self.abort and self.s_lin < 0.5 and self.crossed == 0:
            clear, _ = self._lane_clear(self.dir * W, lane, tracks, v)
            self.nb_lost = 0.0 if (self.dir < 0 and lane.left_ok or self.dir > 0 and lane.right_ok) \
                else getattr(self, "nb_lost", 0.0) + dt
            if not clear or self.nb_lost > 0.3:
                self.abort = True
        self.s_lin = clip(self.s_lin + (-1.0 if self.abort else 1.0) * dt / c.lc_time, 0.0, 1.0)
        r = quintic(self.s_lin) * self.dir * W - self.crossed * W
        if self.s_lin >= 1.0 and not self.abort:
            r = 0.0                                          # settle: just centre in whatever lane we perceive
            if abs(lane.c0) < 0.6 or self.lc_t > c.lc_time + 2.0:
                self.state, self.cooldown = KEEP, c.cooldown
        if self.abort and self.s_lin <= 0.0:
            self.state, self.cooldown = KEEP, c.cooldown
            r = 0.0
        return clip(r, -1.05 * W, 1.05 * W)

    # ---------- main step ----------
    def step(self, t, dt, road, line, dets, v, engaged=True):
        c = self.cfg
        self.suggest = 0
        lane = self.lanes.update(road, line)
        self.last_lane, self.last_r, self.last_L = lane, 0.0, 0.0      # exposed for the HUD / BEV overlay
        tracks = self.veh.update(t, dets, v)
        raw_ok = lane.valid and lane.quality >= c.q_min
        self.bad_t = 0.0 if raw_ok else self.bad_t + dt
        self.good_t = self.good_t + dt if raw_ok else 0.0
        if not self.healthy_state and self.good_t >= c.health_enter_s:
            self.healthy_state = True
        elif self.healthy_state and self.bad_t >= c.health_exit_s:
            self.healthy_state = False
        healthy = self.healthy_state              # debounced: a single bad frame no longer yanks the brake
        info = dict(lane_q=lane.quality, lane_w=lane.width, c0=lane.c0, left=lane.left_ok,
                    right=lane.right_ok, n_veh=len(tracks), kappa=lane.curvature, lc_reason=self.lc_reason)
        if not engaged:
            self.reset()
            return Command(state="OFF", info=info)
        if lane.valid:
            for k in (-1, 1):
                ok = lane.left_ok if k < 0 else lane.right_ok
                self.nb_t[k] = self.nb_t[k] + dt if ok else 0.0
        fault = "LANE LOST" if self.bad_t > c.lane_lost_s else ""

        r_eff = 0.0
        if healthy:
            if self.state == KEEP:
                self.cooldown = max(0.0, self.cooldown - dt)
                self._maybe_start_lc(dt, lane, tracks, v)
            if self.state == LANE_CHANGE:
                r_eff = self._advance_lc(dt, lane, tracks, v)
        elif self.state == LANE_CHANGE:
            self.abort = True

        # lateral: pure pursuit on the (possibly shifted) lane centre. Uses whatever geometry is available -
        # a fresh fit, or (during the brief health-debounce grace window after a missed frame) the held-over
        # shape from LaneTracker - either way lane.coeffs is guaranteed non-None here, never crashes.
        if lane.coeffs is not None:
            L = min(clip(c.la_base + c.la_time * v, c.la_min, c.la_max), lane.z_hi)
            x_t = lane.x_at(L) + r_eff
            self.last_r, self.last_L = r_eff, L
            kappa = 2 * x_t / (x_t * x_t + L * L)
            stick = clip(math.atan(c.wheelbase * kappa) / c.stick_lock_rad, -1.0, 1.0)
            m = c.steer_slew * dt
            self.steer += clip(stick - self.steer, -m, m)
        # else: hold the last steering value while perception recovers

        # longitudinal: while cruising (KEEP), require a car to sit in OUR corridor for a little while before
        # it counts as a lead - a single noisy frame (bad lane geometry, a momentary bad detection) no longer
        # brakes for a car that isn't really in front of us. During an active lane change, use the instant
        # test instead (r_eff already points at the target lane, and safety there should not be smoothed).
        if lane.valid and self.state == KEEP:
            seen = set()
            for tr in tracks:
                seen.add(tr.id)
                cur = self.ego_conf.get(tr.id, 0.0)
                cur = min(1.0, cur + dt / c.ego_confirm_s) if self._in_corridor(tr, lane, 0.0) \
                    else max(0.0, cur - dt / c.ego_release_s)
                self.ego_conf[tr.id] = cur
            self.ego_conf = {k: v for k, v in self.ego_conf.items() if k in seen}
            confirmed = [t for t in tracks if t.z > 0.5 and self.ego_conf.get(t.id, 0.0) >= 1.0]
            lead = min(confirmed, key=lambda t: t.z) if confirmed else None
        else:
            lead = self._lead(lane, tracks, r_eff) if lane.valid else None
        kappa_now = lane.curvature if lane.coeffs is not None else 0.0
        if self._prev_kappa is not None and dt > 1e-6:
            raw_rate = (abs(kappa_now) - abs(self._prev_kappa)) / dt
            a = clip(dt / c.kappa_rate_tau_s, 0.0, 1.0) if c.kappa_rate_tau_s > 0 else 1.0
            self._kappa_rate += a * (raw_rate - self._kappa_rate)
        self._prev_kappa = kappa_now
        # A tightening bend's measured curvature lags the true value while the quadratic fit is still catching
        # up (see sim_test.py S11: a sharp enough bend can lose lock entirely before the fit ever reads the true
        # curvature). Extrapolating forward by the recent curvature rate means the cornering speed limit below
        # brakes for the bend the car is ENTERING, not the shallower one that was true half a second ago - so by
        # the time the fit (or a fault) catches up, the car is already slow enough to still make the turn.
        kappa_anticip = abs(kappa_now) + max(0.0, self._kappa_rate) * c.kappa_anticip_s
        v_curve = math.sqrt(c.a_lat_max / max(kappa_anticip, 1e-3)) if lane.coeffs is not None else 12.0
        v0 = min(self.v_set, max(v_curve, 7.0))
        gap = lead.z - c.cam_to_bumper if lead else None
        v_lead = max(0.0, v + lead.vz) if lead else 0.0
        thr, brk, a_des = self.long.step(dt, v, v0, gap, v_lead)
        if not healthy:
            thr, brk = 0.0, 0.15
        if fault:
            thr, brk = 0.0, 0.3
            self.steer = steer_bleed(self.steer, dt, c.fault_steer_tau_s)
        info.update(r_eff=r_eff, lead_z=lead.z if lead else None, lead_v=v_lead if lead else None,
                    v0=v0, a_des=a_des, crossed=self.crossed, suggest=self.suggest, s=self.s_lin, blocked_t=self.blocked_t,
                    cooldown=self.cooldown, lc_reason=self.lc_reason)   # refreshed: _maybe_start_lc runs after the
                                                                        # earlier snapshot above, so re-read it here
        return Command(self.steer, thr, brk, self.state + (" (abort)" if self.abort and self.state == LANE_CHANGE else ""),
                       fault, info)
