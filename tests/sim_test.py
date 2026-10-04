"""
sim_test.py is the main test planner. Kindly test the planner offline: python tests/sim_test.py (kindly run from the repo root...)

A 3-lane road (straight or curved) is rendered into BEV road/lane-line masks along with random holes and
lane-line dropouts, like a real segmentation network, then the SAME planner code that will drive
Assetto Corsa steers a kinematic-bicycle car with steering lag, command latency and perception delay.
"""
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ac_planner import Autopilot, PlannerCfg, LANE_CHANGE, occlude_clamp

CFG = PlannerCfg()
LANE_W = 3.5
LANE_CENTRES = (-LANE_W, 0.0, LANE_W)                        # left, middle, right
BOUNDARIES = (-1.5 * LANE_W, -0.5 * LANE_W, 0.5 * LANE_W, 1.5 * LANE_W)


class Road:
    def __init__(self, kappa=0.0):
        self.k = kappa

    def xy(self, s, b):
        s = np.asarray(s, float)
        if abs(self.k) < 1e-9:
            return b + 0 * s, s
        th = self.k * s
        return (1 - np.cos(th)) / self.k + b * np.cos(th), np.sin(th) / self.k - b * np.sin(th)

    def theta(self, s):
        return self.k * s

    def locate(self, X, Z):
        """world point -> (s, lateral b, tangent heading), via dense nearest-point search"""
        if not hasattr(self, "_g"):
            sg = np.arange(-40, 600, 0.25)
            cx, cz = self.xy(sg, 0.0)
            self._g = (sg, cx, cz)
        sg, cx, cz = self._g
        i = int(np.argmin((cx - X) ** 2 + (cz - Z) ** 2))
        th = self.theta(sg[i])
        b = (X - cx[i]) * math.cos(th) - (Z - cz[i]) * math.sin(th)
        return sg[i], b, th


class StepRoad:
    """Straight for s <= s_bend, then a constant-radius bend, unlike Road's fixed curvature from s=0, this
    tests the moment of ENTERING a sharp bend from a straight (see S11: a tight enough entry can lose lane
    lock before the quadratic fit ever reads the true curvature, which is what kappa_anticip_s guards against)."""
    def __init__(self, kappa, s_bend=60.0):
        self.s_bend, self.k = s_bend, kappa

    def xy(self, s, b):
        s = np.asarray(s, float)
        x, z = np.zeros_like(s), np.zeros_like(s)
        straight = s <= self.s_bend
        x[straight], z[straight] = b, s[straight]
        bend = ~straight
        th = self.k * (s[bend] - self.s_bend)
        x[bend] = (1 - np.cos(th)) / self.k + b * np.cos(th)
        z[bend] = self.s_bend + np.sin(th) / self.k - b * np.sin(th)
        return x, z

    def theta(self, s):
        s = np.asarray(s, float)
        th = np.zeros_like(s)
        bend = s > self.s_bend
        th[bend] = self.k * (s[bend] - self.s_bend)
        return th

    def locate(self, X, Z):
        if not hasattr(self, "_g"):
            sg = np.arange(-40, 600, 0.25)
            self._g = (sg,) + self.xy(sg, 0.0)
        sg, cx, cz = self._g
        i = int(np.argmin((cx - X) ** 2 + (cz - Z) ** 2))
        th = float(self.theta(np.array([sg[i]]))[0])
        b = (X - cx[i]) * math.cos(th) - (Z - cz[i]) * math.sin(th)
        return sg[i], b, th


class Traffic:
    def __init__(self, lane_b, s, v):
        self.b, self.s, self.v = lane_b, s, v


def render_bev(road, pose, rng, holes=True):
    """Ego-frame BEV masks (road, line) from the true road geometry."""
    Xe, Ze, psi = pose
    s_e, _, _ = road.locate(Xe, Ze)
    s = np.arange(s_e - 15, s_e + 75, 0.5)
    cp, sp = math.cos(psi), math.sin(psi)

    def to_px(b):
        X, Z = road.xy(s, b)
        dx, dz = X - Xe, Z - Ze
        xe, ze = dx * cp - dz * sp, dx * sp + dz * cp
        ok = (ze > -8) & (ze < 44)
        return np.stack([CFG.bev_w / 2 + xe * CFG.ppm, CFG.ego_y - ze * CFG.ppm], 1)[ok]

    road_img = np.zeros((CFG.bev_h, CFG.bev_w), np.uint8)
    line_img = np.zeros_like(road_img)
    left, right = to_px(BOUNDARIES[0]), to_px(BOUNDARIES[-1])
    if len(left) > 2 and len(right) > 2:
        cv2.fillPoly(road_img, [np.round(np.vstack([left, right[::-1]])).astype(np.int32)], 255)
    for b in BOUNDARIES:
        pts = to_px(b)
        if len(pts) > 2:
            cv2.polylines(line_img, [np.round(pts).astype(np.int32)], False, 255, 3)
    if holes:                                              # segmentation-network imperfections
        for _ in range(35):
            x, y = rng.integers(0, CFG.bev_w), rng.integers(200, CFG.bev_h)
            w, h = rng.integers(6, 22), rng.integers(6, 22)
            road_img[y:y + h, x:x + w] = 0
        for _ in range(14):
            x, y = rng.integers(0, CFG.bev_w), rng.integers(150, CFG.bev_h)
            line_img[y:y + rng.integers(20, 70), max(x - 6, 0):x + 6] = 0
    line = line_img > 0
    road = (road_img > 0) | line
    road = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    return road, line


def detect(road, pose, traffic, rng):
    """Noisy detections of the traffic's rear-bottom-centre in the ego frame."""
    Xe, Ze, psi = pose
    out = []
    for c in traffic:
        X, Z = road.xy(c.s - 2.25, c.b)
        dx, dz = float(X - Xe), float(Z - Ze)
        x = dx * math.cos(psi) - dz * math.sin(psi)
        z = dx * math.sin(psi) + dz * math.cos(psi)
        if 3.5 < z < 45 and abs(x) < 12 and rng.random() < 0.92:
            out.append((x + rng.normal(0, 0.15 + 0.005 * z), z + rng.normal(0, 0.3 + 0.02 * z)))
    return np.array(out).reshape(-1, 2)


def simulate(kappa=0.0, T=20.0, b0=0.0, yaw0_deg=0.0, v0=15.0, v_set=60 / 3.6, traffic=(), lock_mismatch=1.0,
             manual=None, seed=1, plan_hz=25.0, cfg=None, observe=False, dropout=None, road=None):
    # dropout=(t0, t1): the perception delivers NO road at all in that window (a shadow under a flyover, glare,
    # a segmentation failure) - used to test what the car does when it is briefly blind
    # road: pass a pre-built Road/StepRoad to override the default constant-curvature-from-start road
    rng = np.random.default_rng(seed)
    road = road if road is not None else Road(kappa)
    ap = Autopilot(cfg or PlannerCfg())
    ap.v_set = v_set
    obs = None                                             # optional shadow-mode twin fed the same inputs
    if observe:
        obs = Autopilot(cfg or PlannerCfg()); obs.v_set = v_set; obs.dry_run = True
    traffic = [Traffic(c.b, c.s, c.v) for c in traffic]
    X, Z, psi, v = b0, 0.0, math.radians(yaw0_deg), v0
    delta, cmd_buf = 0.0, []
    pose_hist = []
    L, LAG, LATENCY, PERC_DELAY = 2.87, 0.15, 0.12, 0.10
    dt_phys, dt_plan = 1 / 100.0, 1 / plan_hz
    t, next_plan, cmd = 0.0, 0.0, None
    log = dict(t=[], b=[], v=[], steer=[], state=[], alat=[], collide=False, min_gap=1e9, lc_dir=[], lc_unsafe=[], obs_steer=[], plan_steer=[], obs_suggest=[], fault_t=None)
    thr = brk = steer_c = 0.0
    was_lc = False
    while t < T:
        if t >= next_plan:
            next_plan += dt_plan
            pose_hist.append((t, X, Z, psi))
            past = [p for p in pose_hist if p[0] <= t - PERC_DELAY]
            _, Xp, Zp, psip = past[-1] if past else pose_hist[0]
            road_m, line_m = render_bev(road, (Xp, Zp, psip), rng)
            if dropout and dropout[0] <= t < dropout[1]:
                road_m, line_m = np.zeros_like(road_m), np.zeros_like(line_m)
            dets = detect(road, (Xp, Zp, psip), traffic, rng)
            if manual and t >= manual[0] and not getattr(ap, "_did_manual", False):
                ap.request_lane_change(manual[1]); ap._did_manual = True
            cmd = ap.step(t, dt_plan, road_m, line_m, dets, v, engaged=True)
            if cmd.fault and log["fault_t"] is None:
                log["fault_t"] = t
            cmd_buf.append((t + LATENCY, cmd))
            if obs is not None:
                co = obs.step(t, dt_plan, road_m, line_m, dets, v, engaged=True)
                log["obs_steer"].append(co.steer); log["plan_steer"].append(cmd.steer)
                if co.info.get("suggest"):
                    log["obs_suggest"].append((round(t, 2), co.info["suggest"]))
            in_lc = cmd.state.startswith(LANE_CHANGE)
            if in_lc and not was_lc:                       # a lane change just started: judge it with ground truth
                s_e, b_e, _ = road.locate(X, Z)
                base = min(LANE_CENTRES, key=lambda c: abs(c - b_e))
                target = base + ap.dir * LANE_W
                unsafe = any(abs(c.b - target) < 1.75 and -9.0 <= c.s - s_e < 7.0 for c in traffic)
                log["lc_dir"].append((round(t, 2), ap.dir))
                log["lc_unsafe"].append(unsafe)
            was_lc = in_lc
        while cmd_buf and cmd_buf[0][0] <= t:
            _, c = cmd_buf.pop(0)
            steer_c, thr, brk = c.steer, c.throttle, c.brake
        steer_target = steer_c * CFG.stick_lock_rad * lock_mismatch
        delta += (steer_target - delta) * dt_phys / LAG
        a = 5.0 * thr - 9.0 * brk - (0.0008 * v * v + 0.15)
        v = max(0.0, v + a * dt_phys)
        psi_dot = v / L * math.tan(delta)
        psi += psi_dot * dt_phys
        X += v * math.sin(psi) * dt_phys
        Z += v * math.cos(psi) * dt_phys
        for c in traffic:
            c.s += c.v * dt_phys
        t += dt_phys
        if int(round(t / dt_phys)) % 4 == 0:
            s_e, b_e, th_e = road.locate(X, Z)
            for c in traffic:
                if abs(b_e - c.b) < 1.9 and (c.s - 2.25 < s_e + 1.5) and (c.s + 2.25 > s_e - 3.4):
                    log["collide"] = True
                if abs(b_e - c.b) < 1.9 and c.s > s_e:
                    log["min_gap"] = min(log["min_gap"], c.s - 2.25 - (s_e + 1.5))
            log["t"].append(t); log["b"].append(b_e); log["v"].append(v)
            log["steer"].append(steer_c)
            log["state"].append(cmd.state if cmd else "")
            log["alat"].append(abs(v * psi_dot))
    for k in ("t", "b", "v", "steer", "alat"):
        log[k] = np.array(log[k])
    log["ap"] = ap
    return log


# ------------------------------------------------------------------ scenarios
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def tail(log, seconds):
    return log["t"] > log["t"][-1] - seconds


def main():
    t0 = time.time()
    print("S1  straight road, start 1.2 m off-centre with 3 deg heading error")
    g = simulate(b0=1.2, yaw0_deg=3.0, T=16)
    late = tail(g, 5)
    check("converges to lane centre", np.abs(g["b"][late]).max() < 0.25, f"(|b| last 5 s max = {np.abs(g['b'][late]).max():.2f} m)")
    check("no big overshoot", g["b"].min() > -0.6, f"(min b = {g['b'].min():.2f} m)")
    check("reaches set speed", abs(g["v"][-1] - 60 / 3.6) < 1.0, f"(v = {g['v'][-1] * 3.6:.0f} km/h)")

    print("S2  constant-radius curves")
    for R in (250, 80, 50):
        g = simulate(kappa=1 / R, T=18, v0=14)
        late = tail(g, 6)
        check(f"R={R} m tracked", np.abs(g["b"][late]).max() < 0.5,
              f"(|b| max = {np.abs(g['b'][late]).max():.2f} m, v = {g['v'][-1] * 3.6:.0f} km/h)")
    g = simulate(kappa=-1 / 120, T=16, v0=14)
    check("left curve R=120 tracked", np.abs(g["b"][tail(g, 6)]).max() < 0.5, f"(|b| max = {np.abs(g['b'][tail(g, 6)]).max():.2f} m)")

    print("S3  steering-gain mismatch (real lock 0.7x and 1.4x of configured)")
    for m in (0.7, 1.4):
        g = simulate(b0=1.0, T=18, lock_mismatch=m)
        late = tail(g, 6)
        check(f"mismatch {m}x stable", np.abs(g["b"][late]).max() < 0.3 and g["steer"][late].std() < 0.05,
              f"(|b| max = {np.abs(g['b'][late]).max():.2f}, steer std = {g['steer'][late].std():.3f})")

    print("S4  slow lead in our lane, both neighbour lanes free -> should overtake")
    lead = Traffic(0.0, 50.0, 9.0)
    g = simulate(T=28, traffic=[lead], v0=16.7)
    fin = g["b"][-1]
    check("changes lane", abs(fin) > 2.8, f"(final b = {fin:+.2f} m, decisions {g['lc_dir']})")
    check("prefers right (overtake side)", len(g["lc_dir"]) > 0 and g["lc_dir"][0][1] == 1)
    check("no collision", not g["collide"], f"(min gap {g['min_gap']:.1f} m)")
    check("smooth (lateral accel < 3 m/s^2)", g["alat"].max() < 3.0, f"(max {g['alat'].max():.2f})")
    check("settles in new lane", abs(abs(fin) - 3.5) < 0.4)
    check("no repeated lane changes", len(g["lc_dir"]) == 1, f"({len(g['lc_dir'])} started)")

    print("S5  slow lead ahead + slower cars in BOTH neighbour lanes that we are overtaking (blind zone)")
    tr = [Traffic(0.0, 48.0, 9.0), Traffic(3.5, 14.0, 9.0), Traffic(-3.5, 18.0, 9.0)]
    g = simulate(T=30, traffic=tr, v0=12.0)
    check("no collision", not g["collide"], f"(min gap {g['min_gap']:.1f} m)")
    check("every lane change was truly safe", not any(g["lc_unsafe"]), f"(decisions {g['lc_dir']}, unsafe flags {g['lc_unsafe']})")
    check("never gets closer than 6 m to the lead", g["min_gap"] > 6.0, f"({g['min_gap']:.1f} m)")

    print("S6  right lane occupied by a slower car we are about to pass, left free -> go LEFT")
    tr = [Traffic(0.0, 55.0, 10.0), Traffic(3.5, 28.0, 12.0)]
    g = simulate(T=30, traffic=tr, v0=16.7)
    check("no collision", not g["collide"], f"(min gap {g['min_gap']:.1f} m)")
    check("did not cut into the right car", all(d == -1 for _, d in g["lc_dir"]) and len(g["lc_dir"]) >= 1,
          f"(decisions {g['lc_dir']})")
    check("lane change was truly safe", not any(g["lc_unsafe"]))

    print("S7  manual lane-change requests on a gentle curve (tests lane-identity flip handling)")
    for d in (1, -1):
        g = simulate(kappa=1 / 300, T=16, manual=(3.0, d))
        fin = g["b"][-1]
        check(f"request {'right' if d > 0 else 'left'}", abs(fin - d * 3.5) < 0.4 and g["alat"].max() < 3.0,
              f"(final b = {fin:+.2f}, max lat acc {g['alat'].max():.2f}, crossed={g['ap'].crossed})")

    print("S8  shadow / dry-run twin (same inputs as the real planner, never acts)")
    g = simulate(T=14, observe=True, b0=0.8)
    dmax = max(abs(a - b) for a, b in zip(g["obs_steer"], g["plan_steer"]))
    check("dry-run steering identical to the real planner's", dmax < 1e-9, f"(max diff {dmax:.2e})")
    g = simulate(T=24, traffic=[Traffic(0.0, 50.0, 9.0)], observe=True, v0=16.7)
    t_lc, d_lc = g["lc_dir"][0]
    t_sg, d_sg = g["obs_suggest"][0]
    check("suggests the same lane change at the same moment", d_sg == d_lc and abs(t_sg - t_lc) < 0.25,
          f"(real planner started {d_lc:+d} at {t_lc:.2f} s, twin suggested {d_sg:+d} at {t_sg:.2f} s)")

    print("S9  lane lost (perception blind) at 90 km/h: must not drive off the road while the fault is handled")
    for label, k in (("straight", 0.0), ("R=250 curve", 1 / 250), ("R=120 curve", 1 / 120)):
        g = simulate(kappa=k, T=10.3, v0=25.0, v_set=25.0, dropout=(6.0, 10.0))
        m = (g["t"] >= 6.0) & (g["t"] <= 10.3)
        dev = float(np.abs(g["b"][m]).max())
        check(f"{label}: 4 s blind stays inside the lane lines", dev < 0.8 and g["fault_t"] is not None,
              f"(max deviation {dev:.2f} m, fault raised at {g['fault_t']:.1f} s)" if g["fault_t"] else f"(no fault raised, dev {dev:.2f} m)")
    g = simulate(kappa=1 / 120, T=7.3, v0=25.0, v_set=25.0, dropout=(6.0, 7.0))
    m = (g["t"] >= 6.0) & (g["t"] <= 7.3)
    check("R=120 curve: a 1 s blackout barely moves the car", np.abs(g["b"][m]).max() < 0.25,
          f"(max deviation {np.abs(g['b'][m]).max():.2f} m)")

    print("S10 a false detection on the skyline (box bottom above the horizon, a known real bug) must not")
    print("    become a phantom near-lead and trigger a panic brake")
    # Minimal standalone re-creation of BirdsEyeView.project()'s ground-plane maths (sim_test.py must not import
    # ac_perception_v4 - that pulls in dxcam/onnxruntime, which this offline test suite deliberately has none of).
    def project_like_v4(boxes, height_m=1.15, pitch_deg=-6.0, vfov_deg=60.0, w=1280, h=720):
        f = (h / 2) / math.tan(math.radians(vfov_deg) / 2)
        s_, c_ = math.sin(math.radians(pitch_deg)), math.cos(math.radians(pitch_deg))
        cx_pt = (boxes[:, 0] + boxes[:, 2]) / 2
        y2 = boxes[:, 3]
        # invert: pixel (cx, y2) -> ground (X, Z), same relations BirdsEyeView.rebuild()'s to_px() encodes
        Z = (c_ * height_m - (y2 - h / 2) / f * s_ * 0 - 0)  # placeholder, replaced by direct closed form below
        # closed form: y_px = f*(c*H - s*Z)/(s*H + c*Z) + h/2  =>  solve for Z
        yc = y2 - h / 2
        num = f * c_ * height_m - yc * s_ * height_m
        den = yc * c_ + f * s_
        Z = num / np.where(den == 0, 1e-9, den)
        X = cx_pt * (s_ * height_m + c_ * Z) / f
        hf_num = f * c_ * height_m - (605 - h / 2) * s_ * height_m
        hf_den = (605 - h / 2) * c_ + f * s_
        hood_floor = hf_num / (hf_den if hf_den != 0 else 1e-9)
        reliable = Z > hood_floor * 1.03
        return X, Z, reliable
    sky_box = np.array([[610.0, 395.0, 670.0, 421.0]])          # bottom edge just above the horizon
    Xs, Zs, rels = project_like_v4(sky_box)
    check("sky detection projects to negative Z (not a nearby object)", Zs[0] < 0, f"(Z={Zs[0]:.1f} m)")
    Xc, Zc = occlude_clamp(Xs, Zs.copy(), rels, PlannerCfg().cam_to_bumper)
    check("occlude_clamp discards it rather than treating it as touching", not (0 < Zc[0] < 2.0), f"(Zc={Zc[0]:.2f} m)")
    close_box = np.array([[600.0, 500.0, 700.0, 690.0]])         # a REAL close car, hood-occluded
    Xr, Zr, relr = project_like_v4(close_box)
    Xrc, Zrc = occlude_clamp(Xr, Zr.copy(), relr, PlannerCfg().cam_to_bumper)
    check("a genuinely close (hood-occluded) car is still clamped to 'touching'", abs(Zrc[0] - 0.75) < 1e-6, f"(Zc={Zrc[0]:.2f} m)")

    print("S11 entering a sharp bend (R=15/12 m, a real hairpin, not the gentle S2 sweepers) from a straight at")
    print("    60 km/h: the lane fit's curvature reading lags the true bend while it catches up, and used to")
    print("    lose lock before it caught up at all - car ran wide onto the curb (a known real bug)")
    for R in (15, 12):
        g = simulate(kappa=0.0, T=16.0, v0=60 / 3.6, v_set=60 / 3.6, seed=3, road=StepRoad(1.0 / R))
        worst = float(np.abs(g["b"]).max())
        check(f"R={R} m bend entry: stays on the road (within the lane+shoulder)", worst < 1.75,
              f"(max deviation {worst:.2f} m)")

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks passed in {time.time() - t0:.0f}s")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
