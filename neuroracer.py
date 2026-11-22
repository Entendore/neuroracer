#!/usr/bin/env python
"""
NeuroRacer — neural networks learn to race on procedural (or hand-drawn) tracks.

    pip install PySide6 numpy        # numpy optional but recommended
    python neuroracer.py

Features: neuro-evolution (GA) racing · procedural + curriculum tracks ·
recurrent brains · speciation · multi-track (generalist) fitness · racing-line
optimization · track editor · deterministic scrubbable replays · record ghost ·
manual driving · OpenGL renderer · headless trainer (train.py).

Keys: Space pause · N new track · T track editor · G record gen · O replay ·
R restart · E end gen · F follow cam · M drive (WASD/arrows) · +/- speed ·
wheel zoom · drag pan · double-click reset
"""
import base64
import json
import math
import os
import random
import sys
from array import array
from operator import mul

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (QColor, QFont, QPainter, QPainterPath, QPalette,
                           QPen, QPolygonF)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QSpinBox, QVBoxLayout,
                               QWidget)

# ---------------------------------------------------------------------------
# tuning knobs
# ---------------------------------------------------------------------------
DT = 1.0 / 60.0
RAY_ANGLES = (-90, -55, -30, -12, 0, 12, 30, 55, 90)
RAY_RANGE = 230.0
HIDDEN = 10
CTX = 4                                   # recurrent context units (0 = v1 brain)
N_IN = len(RAY_ANGLES) + 4
N_OUT = 2
OLD_GENOME_SIZE = N_IN * HIDDEN + HIDDEN + HIDDEN * N_OUT + N_OUT
GENOME_SIZE = (HIDDEN * (N_IN + CTX) + HIDDEN + HIDDEN * N_OUT + N_OUT
               + CTX * HIDDEN + CTX)
MAX_SPEED = 340.0
ACCEL = 300.0
BRAKE = 560.0
DRAG = 0.8
TURN_RATE = 3.6
CAR_CLEAR = 7.0
CAR_LEN, CAR_W = 26.0, 13.0
GEN_SECONDS = 35.0
GEN_MAX_STEPS = int(GEN_SECONDS / DT)
CELL = 6.0

CAR_SHAPE = QPolygonF([
    QPointF(CAR_LEN * 0.50, 0.0),
    QPointF(CAR_LEN * 0.10, -CAR_W * 0.50),
    QPointF(-CAR_LEN * 0.50, -CAR_W * 0.40),
    QPointF(-CAR_LEN * 0.50, CAR_W * 0.40),
    QPointF(CAR_LEN * 0.10, CAR_W * 0.50),
])


def migrate_genome(genome):
    """v1 (feedforward) -> v2 (recurrent) by zero-filling context weights.
    Zero context is a fixed point, so old champions behave identically."""
    g = [float(x) for x in genome]
    if len(g) == GENOME_SIZE:
        return g
    if CTX == 0 or len(g) != OLD_GENOME_SIZE:
        raise ValueError(f"genome size {len(g)} unsupported (want {GENOME_SIZE})")
    k = HIDDEN * N_IN
    w1, b1 = g[:k], g[k:k + HIDDEN]
    k += HIDDEN
    w2 = g[k:k + HIDDEN * N_OUT]
    k += HIDDEN * N_OUT
    b2 = g[k:k + N_OUT]
    out = []
    for r in range(HIDDEN):
        out.extend(w1[r * N_IN:(r + 1) * N_IN])
        out.extend([0.0] * CTX)
    out.extend(b1)
    out.extend(w2)
    out.extend(b2)
    out.extend([0.0] * (CTX * HIDDEN + CTX))
    return out


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _catmull(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
               + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
               + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
               + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
               + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
    )


def resample_closed(pts, spacing):
    n = len(pts)
    cum = [0.0]
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cum.append(cum[-1] + math.hypot(bx - ax, by - ay))
    total = cum[-1]
    m = max(16, int(total / spacing))
    step = total / m
    out, j = [], 0
    for k in range(m):
        d = k * step
        while cum[j + 1] < d:
            j += 1
        ax, ay = pts[j % n]
        bx, by = pts[(j + 1) % n]
        t = (d - cum[j]) / max(1e-9, cum[j + 1] - cum[j])
        out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return out


def min_curvature_radius(pts):
    rmin = float("inf")
    n = len(pts)
    for i in range(n):
        ax, ay = pts[(i - 1) % n]
        bx, by = pts[i]
        cx, cy = pts[(i + 1) % n]
        a = math.hypot(bx - ax, by - ay)
        b = math.hypot(cx - bx, cy - by)
        c = math.hypot(ax - cx, ay - cy)
        area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5
        if area > 1e-6:
            rmin = min(rmin, a * b * c / (4.0 * area))
    return rmin


def compute_racing_line(center, half_width, clear=CAR_CLEAR, iters=500, alpha=0.12):
    """Shortest-path smoothing of the centreline, clamped to the corridor."""
    try:
        import numpy as np
    except ImportError:
        return None
    ctr = np.asarray(center, dtype=float)
    pts = ctr.copy()
    max_off = max(4.0, half_width - clear - 8.0)
    for _ in range(iters):
        mid = 0.5 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
        pts += alpha * (mid - pts)
        rel = pts - ctr
        d = np.hypot(rel[:, 0], rel[:, 1])
        over = d > max_off
        if over.any():
            pts[over] = ctr[over] + rel[over] * (max_off / d[over])[:, None]
    return [(float(x), float(y)) for x, y in pts]


def sketch_centerline(points, spacing=14.0, smoothing=3):
    """Freehand loop -> clean, uniformly spaced centreline."""
    pts = resample_closed(list(points), 5.0)
    for _ in range(smoothing):
        n = len(pts)
        pts = [((pts[(i - 1) % n][0] + 2 * pts[i][0] + pts[(i + 1) % n][0]) * 0.25,
                (pts[(i - 1) % n][1] + 2 * pts[i][1] + pts[(i + 1) % n][1]) * 0.25)
               for i in range(n)]
    return resample_closed(pts, spacing)


def _seg_seg_dist2(ax, ay, bx, by, cx, cy, dx, dy):
    def pt_seg(px, py, sx, sy, ex, ey):
        vx, vy = ex - sx, ey - sy
        l2 = vx * vx + vy * vy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / l2))
        qx, qy = sx + vx * t, sy + vy * t
        return (px - qx) ** 2 + (py - qy) ** 2
    return min(pt_seg(ax, ay, cx, cy, dx, dy), pt_seg(bx, by, cx, cy, dx, dy),
               pt_seg(cx, cy, ax, ay, bx, by), pt_seg(dx, dy, ax, ay, bx, by))


def centerline_self_intersects(center, min_dist, skip=14):
    n = len(center)
    thr = min_dist * min_dist
    for i in range(n):
        ax, ay = center[i]
        bx, by = center[(i + 1) % n]
        mix, miy = 0.5 * (ax + bx), 0.5 * (ay + by)
        for j in range(i + skip, n):
            if n - (j - i) < skip:
                break
            cx, cy = center[j]
            dx, dy = center[(j + 1) % n]
            mjx, mjy = 0.5 * (cx + dx), 0.5 * (cy + dy)
            if (mix - mjx) ** 2 + (miy - mjy) ** 2 > 4.0 * thr:
                continue
            if _seg_seg_dist2(ax, ay, bx, by, cx, cy, dx, dy) < thr:
                return True
    return False


def track_from_sketch(points, half_width=46.0, spacing=14.0):
    """Validate a freehand loop. Returns (Track, None) or (None, reason)."""
    if len(points) < 8:
        return None, "draw more of the loop first"
    center = sketch_centerline(points, spacing=spacing)
    if len(center) < 60:
        return None, "track too short — draw a bigger loop"
    rmin = min_curvature_radius(center)
    need = half_width * 1.8
    if rmin < need:
        return None, (f"corner too tight (radius {rmin:.0f}px, need >= {need:.0f}px) — "
                      "smooth the loop or shrink road width with [")
    if centerline_self_intersects(center, 2.0 * half_width + 8.0):
        return None, "track crosses itself — redraw with more separation"
    return Track.from_centerline(center, half_width=half_width, spacing=spacing), None


# ---------------------------------------------------------------------------
# occupancy grid + wall-distance field
# ---------------------------------------------------------------------------
class Grid:
    __slots__ = ("x0", "y0", "gw", "gh", "inv", "cells", "margin", "df")

    def __init__(self, x0, y0, gw, gh, cell, margin):
        self.x0, self.y0, self.gw, self.gh = x0, y0, gw, gh
        self.inv = 1.0 / cell
        self.margin = margin
        self.cells = bytearray(gw * gh)
        self.df = None


def build_grid(center, radius, cell=CELL):
    xs = [p[0] for p in center]
    ys = [p[1] for p in center]
    pad = radius + 24.0
    x0, y0 = min(xs) - pad, min(ys) - pad
    gw = int((max(xs) + pad - x0) / cell) + 1
    gh = int((max(ys) + pad - y0) / cell) + 1
    g = Grid(x0, y0, gw, gh, cell, CAR_CLEAR)
    offs, nc, r2 = [], int(radius / cell), radius * radius
    for gy in range(-nc, nc + 1):
        for gx in range(-nc, nc + 1):
            if (gx * cell) ** 2 + (gy * cell) ** 2 <= r2:
                offs.append((gx, gy))
    ic = 1.0 / cell
    for (x, y) in center:
        cx, cy = int((x - x0) * ic), int((y - y0) * ic)
        for gx, gy in offs:
            X, Y = cx + gx, cy + gy
            if 0 <= X < gw and 0 <= Y < gh:
                g.cells[Y * gw + X] = 1
    # two-pass 3-4 chamfer distance transform (px to nearest wall)
    INF = 1 << 30
    df = [float(INF) if g.cells[i] else 0.0 for i in range(gw * gh)]
    for y in range(gh):
        row = y * gw
        for x in range(gw):
            i = row + x
            v = df[i]
            if v == 0.0:
                continue
            if x > 0 and df[i - 1] + 3.0 < v: v = df[i - 1] + 3.0
            if y > 0:
                if df[i - gw] + 3.0 < v: v = df[i - gw] + 3.0
                if x > 0 and df[i - gw - 1] + 4.0 < v: v = df[i - gw - 1] + 4.0
                if x < gw - 1 and df[i - gw + 1] + 4.0 < v: v = df[i - gw + 1] + 4.0
            df[i] = v
    for y in range(gh - 1, -1, -1):
        row = y * gw
        for x in range(gw - 1, -1, -1):
            i = row + x
            v = df[i]
            if v == 0.0:
                continue
            if x < gw - 1 and df[i + 1] + 3.0 < v: v = df[i + 1] + 3.0
            if y < gh - 1:
                if df[i + gw] + 3.0 < v: v = df[i + gw] + 3.0
                if x < gw - 1 and df[i + gw + 1] + 4.0 < v: v = df[i + gw + 1] + 4.0
                if x > 0 and df[i + gw - 1] + 4.0 < v: v = df[i + gw - 1] + 4.0
            df[i] = v
    g.df = [v * (CELL / 3.0) * 0.85 for v in df]
    return g


# ---------------------------------------------------------------------------
# track
# ---------------------------------------------------------------------------
class Track:
    def __init__(self, seed=None, difficulty=0.0):
        rng = random.Random(seed)
        self.seed = seed
        self.difficulty = difficulty
        d = difficulty
        self.spacing = 14.0
        self.half_width = 46.0 * (1.30 - 0.65 * d)
        self.base_r = rng.uniform(430.0, 540.0)
        center, irreg = None, 0.34 * (0.75 + 0.85 * d)
        for _ in range(90):
            center = self._make_center(rng, irreg)
            if len(center) >= 60 and min_curvature_radius(center) >= self.half_width * 1.8:
                break
            irreg *= 0.93
        self._build(center)

    @classmethod
    def from_centerline(cls, center, half_width=46.0, spacing=14.0):
        self = cls.__new__(cls)
        self.seed = None
        self.difficulty = 0.0
        self.spacing = spacing
        self.half_width = half_width
        self._build(center)
        return self

    def _make_center(self, rng, irreg):
        m = rng.randint(14, 22)
        vals = [rng.uniform(-1.0, 1.0) for _ in range(m)]
        for _ in range(3):
            vals = [(vals[i - 1] + 2 * vals[i] + vals[(i + 1) % m]) * 0.25 for i in range(m)]
        peak = max(abs(v) for v in vals) or 1.0
        pts = []
        for i in range(m):
            a = 2.0 * math.pi * i / m
            r = self.base_r * (1.0 + (vals[i] / peak) * irreg)
            pts.append((math.cos(a) * r, math.sin(a) * r))
        dense = []
        for i in range(m):
            dense.extend(_catmull(pts[(i - 1) % m], pts[i], pts[(i + 1) % m],
                                  pts[(i + 2) % m], j / 24.0)
                         for j in range(24))
        return resample_closed(dense, self.spacing)

    def _build(self, center):
        hw = self.half_width
        self.center = center
        self.n = n = len(center)
        self.length = n * self.spacing
        outer, inner = QPolygonF(), QPolygonF()
        for i in range(n):
            ax, ay = center[(i - 1) % n]
            bx, by = center[(i + 1) % n]
            tx, ty = bx - ax, by - ay
            tl = math.hypot(tx, ty) or 1.0
            nx, ny = -ty / tl * hw, tx / tl * hw
            x, y = center[i]
            outer.append(QPointF(x + nx, y + ny))
            inner.append(QPointF(x - nx, y - ny))
        self.poly_outer, self.poly_inner = outer, inner
        self.center_poly = QPolygonF([QPointF(x, y) for x, y in center])
        self.center_poly.append(QPointF(center[0][0], center[0][1]))
        self.road_path = QPainterPath()
        self.road_path.setFillRule(Qt.FillRule.OddEvenFill)
        self.road_path.addPolygon(outer)
        self.road_path.addPolygon(inner)
        self.grid = build_grid(center, hw - CAR_CLEAR)
        self.rect = QRectF(self.grid.x0, self.grid.y0,
                           self.grid.gw * CELL, self.grid.gh * CELL)
        ax, ay = center[-1]
        bx, by = center[1]
        self.start = (center[0][0], center[0][1], math.atan2(by - ay, bx - ax))
        self.fit = self.center
        self.racing = compute_racing_line(center, hw)
        self.racing_path = None
        if self.racing:
            rp = QPainterPath()
            rp.addPolygon(QPolygonF([QPointF(x, y) for x, y in self.racing]))
            rp.closeSubpath()
            self.racing_path = rp

    def spec(self):
        if self.seed is not None:
            return {"kind": "procedural", "seed": self.seed,
                    "difficulty": getattr(self, "difficulty", 0.0)}
        return {"kind": "custom", "half_width": self.half_width,
                "spacing": self.spacing,
                "center": [(round(x, 2), round(y, 2)) for x, y in self.center]}

    @classmethod
    def from_spec(cls, spec):
        if spec.get("kind") == "custom":
            return cls.from_centerline(
                [(float(x), float(y)) for x, y in spec["center"]],
                half_width=float(spec.get("half_width", 46.0)),
                spacing=float(spec.get("spacing", 14.0)))
        return cls(spec.get("seed"), difficulty=float(spec.get("difficulty", 0.0)))


# ---------------------------------------------------------------------------
# neural network (recurrent)
# ---------------------------------------------------------------------------
class Net:
    __slots__ = ("w1", "b1", "w2", "b2", "wc", "bc", "ctx", "x", "h", "o")

    def __init__(self, genome):
        genome = migrate_genome(genome)
        self.w1 = [genome[i * (N_IN + CTX):(i + 1) * (N_IN + CTX)] for i in range(HIDDEN)]
        s1 = HIDDEN * (N_IN + CTX)
        self.b1 = genome[s1:s1 + HIDDEN]
        s2 = s1 + HIDDEN
        self.w2 = [genome[s2 + i * HIDDEN:s2 + (i + 1) * HIDDEN] for i in range(N_OUT)]
        s3 = s2 + HIDDEN * N_OUT
        self.b2 = genome[s3:s3 + N_OUT]
        s4 = s3 + N_OUT
        self.wc = [genome[s4 + i * HIDDEN:s4 + (i + 1) * HIDDEN] for i in range(CTX)]
        s5 = s4 + CTX * HIDDEN
        self.bc = genome[s5:s5 + CTX]
        self.ctx = [0.0] * CTX
        self.x = self.h = self.o = None

    def reset_state(self):
        for i in range(CTX):
            self.ctx[i] = 0.0

    def forward(self, x):
        self.x = x
        xf = (*x, *self.ctx)
        h = [math.tanh(b + sum(map(mul, xf, row))) for row, b in zip(self.w1, self.b1)]
        o = [math.tanh(b + sum(map(mul, h, row))) for row, b in zip(self.w2, self.b2)]
        self.ctx = [math.tanh(b + sum(map(mul, h, row))) for row, b in zip(self.wc, self.bc)]
        self.h, self.o = h, o
        return o


# ---------------------------------------------------------------------------
# lidar (sphere tracing against the distance field)
# ---------------------------------------------------------------------------
def sense_rays(track, x, y, head):
    grid = track.grid
    gx0, gy0, inv = grid.x0, grid.y0, grid.inv
    gw, gh, cells, df = grid.gw, grid.gh, grid.cells, grid.df
    margin = grid.margin
    cos, sin = math.cos, math.sin
    rays = []
    step_len = 7.0
    for off in RAY_ANGLES:
        ang = head + off * 0.017453292519943295
        dxr, dyr = cos(ang), sin(ang)
        dist = RAY_RANGE
        prev, d = 0.0, step_len
        while d < RAY_RANGE:
            px, py = x + dxr * d, y + dyr * d
            gxi, gyi = int((px - gx0) * inv), int((py - gy0) * inv)
            if gxi < 0 or gyi < 0 or gxi >= gw or gyi >= gh or not cells[gyi * gw + gxi]:
                lo, hi = prev, d
                for _ in range(4):
                    mid = 0.5 * (lo + hi)
                    px, py = x + dxr * mid, y + dyr * mid
                    gxi, gyi = int((px - gx0) * inv), int((py - gy0) * inv)
                    if gxi < 0 or gyi < 0 or gxi >= gw or gyi >= gh or not cells[gyi * gw + gxi]:
                        hi = mid
                    else:
                        lo = mid
                dist = min(RAY_RANGE, 0.5 * (lo + hi) + margin)
                break
            jump = df[gyi * gw + gxi]
            if jump < step_len:
                jump = step_len
            prev, d = d, d + jump
        rays.append(dist / RAY_RANGE)
    return rays


# ---------------------------------------------------------------------------
# car
# ---------------------------------------------------------------------------
class Car:
    __slots__ = ("genome", "net", "x", "y", "head", "v", "ci", "progress",
                 "alive", "stuck", "rays", "lat", "steer", "thr", "hue",
                 "death_x", "death_y", "trail",
                 "age", "lap_start_age", "last_lap", "laps", "li",
                 "manual", "ctrl", "track", "shadow", "fit", "rec_in")

    def __init__(self, genome, track, hue, manual=False):
        self.genome = genome
        self.net = None if genome is None else Net(genome)
        self.hue = hue
        self.manual = manual
        self.ctrl = (0.0, 0.0)
        self.shadow = False
        self.rec_in = None
        self.reset(track)

    def reset(self, track):
        self.track = track
        self.x, self.y, self.head = track.start
        self.v = 0.0
        self.ci = 0
        self.progress = 0
        self.alive = True
        self.stuck = 0.0
        self.rays = [1.0] * len(RAY_ANGLES)
        self.lat = self.steer = self.thr = 0.0
        self.death_x = self.death_y = None
        self.trail = []
        self.age = 0.0
        self.lap_start_age = 0.0
        self.last_lap = None
        self.laps = 0
        self.li = 0
        self.fit = 0.0
        if self.net is not None:
            self.net.reset_state()

    def step(self, world):
        t = self.track
        grid = t.grid
        c = t.fit
        n = t.n

        ci, best, bi = self.ci, 1e30, self.ci
        for k in range(-16, 17):
            i = ci + k
            if i < 0:
                i += n
            elif i >= n:
                i -= n
            pt = c[i]
            dx = pt[0] - self.x
            dy = pt[1] - self.y
            d = dx * dx + dy * dy
            if d < best:
                best, bi = d, i
        delta = bi - ci
        if delta > n // 2:
            delta -= n
        elif delta < -(n // 2):
            delta += n
        self.progress += delta
        self.ci = bi

        li = self.progress // n
        if li > self.li and li >= 1 and (self.age - self.lap_start_age) > 1.0:
            lap_t = self.age - self.lap_start_age
            self.last_lap = lap_t
            self.laps += 1
            self.lap_start_age = self.age
            if self.manual:
                if world.player_best_lap is None or lap_t < world.player_best_lap:
                    world.player_best_lap = lap_t
            elif not self.shadow:
                if world.best_lap is None or lap_t < world.best_lap:
                    world.best_lap = lap_t
                    world.best_lap_gen = world.generation
        self.li = li
        self.age += DT

        pt = c[bi]
        pax, pay = c[bi - 1]
        pbx, pby = c[bi + 1] if bi + 1 < n else c[0]
        tx, ty = pbx - pax, pby - pay
        tl = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tl, ty / tl
        dx, dy = self.x - pt[0], self.y - pt[1]
        self.lat = (tx * dy - ty * dx) / t.half_width

        head = self.head
        self.rays = sense_rays(t, self.x, self.y, head)

        if self.manual:
            steer, thr = self.ctrl
        else:
            chd, shd = math.cos(head), math.sin(head)
            inputs = (*self.rays, self.v / MAX_SPEED,
                      shd * tx - chd * ty, chd * tx + shd * ty, self.lat)
            out = self.net.forward(inputs)
            steer, thr = out[0], out[1]

        v = self.v
        v += (thr * ACCEL if thr > 0 else thr * BRAKE) * DT
        v -= DRAG * v * DT
        v = 0.0 if v < 0.0 else (MAX_SPEED if v > MAX_SPEED else v)
        self.v = v
        self.head += steer * TURN_RATE * (v / (v + 70.0)) * DT
        ch2, sh2 = math.cos(self.head), math.sin(self.head)
        self.x += ch2 * v * DT
        self.y += sh2 * v * DT
        self.trail.append((self.x, self.y))
        if len(self.trail) > 70:
            self.trail.pop(0)

        gx0, gy0, inv = grid.x0, grid.y0, grid.inv
        gw, gh, cells = grid.gw, grid.gh, grid.cells
        gxi, gyi = int((self.x - gx0) * inv), int((self.y - gy0) * inv)
        if gxi < 0 or gyi < 0 or gxi >= gw or gyi >= gh or not cells[gyi * gw + gxi]:
            self.alive = False
            self.death_x, self.death_y = self.x, self.y
            return
        if not self.manual:
            if v < 22.0:
                self.stuck += DT
                if self.stuck > 2.5:
                    self.alive = False
                    self.death_x, self.death_y = self.x, self.y
            else:
                self.stuck = max(0.0, self.stuck - 2.0 * DT)


# ---------------------------------------------------------------------------
# world + evolution (multi-track fitness, speciation, curriculum, recorder)
# ---------------------------------------------------------------------------
class World:
    def __init__(self, cfg, seed=None, difficulty=0.0):
        self.cfg = cfg
        for _k, _v in (("population", 44), ("mut_rate", 0.15), ("mut_strength", 0.30)):
            self.cfg.setdefault(_k, _v)
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.multi = max(1, int(cfg.get("multi", 1)))
        self.eval_mode = cfg.get("eval_mode", "mean")
        self._exam_base = self.rng.randint(0, 10 ** 9)
        self._build_tracks()
        self.track_version = 0
        self.best_ever_genome = None
        self.best_ever_progress = 0
        self.best_ever_fit = 0.0
        self.last_best_progress = 0
        self.history = []
        self.fit_history = []
        self.best_lap = None
        self.best_lap_gen = None
        self.ghost = None
        self.recorder = None
        self.on_replay = None
        self.player = None
        self.player_best_lap = None
        self.respawn_t = 0.0
        self.stall = 0
        self.gen_max_steps = GEN_MAX_STEPS
        self.shadow = None
        self.species_reps = []
        self.species_stall = []
        self.species_best = []
        self.restart_evolution()

    # ---- tracks -------------------------------------------------------------
    def _build_tracks(self):
        base = self._exam_base
        self.tracks = [Track(base, difficulty=self.difficulty)]
        for i in range(1, self.multi):
            s = (base + 0x9E3779B9 * i) % (10 ** 9)
            self.tracks.append(Track(s, difficulty=self.difficulty))
        self.track = self.tracks[0]
        for t in self.tracks:
            self._apply_fit(t)

    def _apply_fit(self, track=None):
        t = track if track is not None else self.track
        t.fit = t.racing if (self.cfg.get("racing_line") and getattr(t, "racing", None)) else t.center
        return t

    def _spawn_shadow(self, genomes):
        if self.multi <= 1:
            self.shadow = None
            return
        P = len(genomes)
        self.shadow = [[Car(list(g), trk, i / max(1, P - 1)) for i, g in enumerate(genomes)]
                       for trk in self.tracks[1:]]
        for col in self.shadow:
            for c in col:
                c.shadow = True

    # ---- population ----------------------------------------------------------
    def restart_evolution(self):
        P = self.cfg["population"]
        rng = self.rng
        self.cars = [Car([rng.gauss(0, 0.5) for _ in range(GENOME_SIZE)],
                         self.track, i / max(1, P - 1)) for i in range(P)]
        self.generation = 1
        self.gen_steps = 0
        self.history = []
        self.fit_history = []
        self.best_ever_fit = 0.0
        self.stall = 0
        self.species_reps = []
        self.species_stall = []
        self.species_best = []
        self.recorder = None
        self._spawn_shadow([c.genome for c in self.cars])
        if self.player is not None:
            self.player.reset(self.track)

    def new_track(self):
        self._exam_base = self.rng.randint(0, 10 ** 9)
        self._build_tracks()
        self.track_version += 1
        self.best_ever_genome = None
        self.best_ever_progress = 0
        self.best_ever_fit = 0.0
        self.last_best_progress = 0
        self.best_lap = None
        self.best_lap_gen = None
        self.ghost = None
        self.player_best_lap = None
        if self.player is not None:
            self.player.reset(self.track)
        self.restart_evolution()

    def load_genome(self, genome, fitness):
        genome = migrate_genome(list(genome))
        rng, P = self.rng, self.cfg["population"]
        self.cars = []
        for i in range(P):
            g = list(genome)
            if i > 0:
                for j in range(GENOME_SIZE):
                    if rng.random() < 0.10:
                        g[j] += rng.gauss(0, 0.10)
            self.cars.append(Car(g, self.track, i / max(1, P - 1)))
        self.generation = 1
        self.gen_steps = 0
        self.history = []
        self.fit_history = []
        self.last_best_progress = 0
        self.best_lap = None
        self.best_lap_gen = None
        self.best_ever_genome = list(genome)
        self.best_ever_progress = fitness
        self.best_ever_fit = fitness / self.track.n
        self.ghost = Car(list(genome), self.track, hue=0.72)
        self.species_reps = []
        self.species_stall = []
        self.species_best = []
        self.recorder = None
        self._spawn_shadow([c.genome for c in self.cars])

    def promote(self):
        champ = self.best_ever_genome
        g0 = self.generation
        self.difficulty = min(0.85, self.difficulty + 0.15)
        self._build_tracks()
        self.track_version += 1
        self.best_ever_genome = None
        self.best_ever_progress = 0
        self.best_ever_fit = 0.0
        self.last_best_progress = 0
        self.best_lap = None
        self.best_lap_gen = None
        self.ghost = None
        self.restart_evolution()
        if champ is not None:
            self.load_genome(champ, 0.0)
            self.generation = g0 + 1

    def install_track(self, track, reseed=True):
        """Editor/file tracks replace the DISPLAY track; exam tracks stay."""
        champ = self.best_ever_genome
        self.tracks[0] = track
        self.track = track
        self._apply_fit(track)
        self.track_version += 1
        self.best_ever_genome = None
        self.best_ever_progress = 0
        self.best_ever_fit = 0.0
        self.last_best_progress = 0
        self.best_lap = None
        self.best_lap_gen = None
        self.ghost = None
        self.restart_evolution()
        if reseed and champ is not None:
            self.load_genome(champ, 0.0)
            self.generation = 1

    # ---- queries ---------------------------------------------------------------
    def alive_count(self):
        return sum(1 for c in self.cars if c.alive)

    def ns(self):
        return len(self.species_reps)

    def leader(self):
        best = None
        for c in self.cars:
            if c.alive and (best is None or c.progress > best.progress):
                best = c
        if best is None:
            for c in self.cars:
                if best is None or c.progress > best.progress:
                    best = c
        return best

    # islands helpers (scalar fallback)
    def get_genome(self, i):
        return list(self.cars[i].genome)

    def top_indices(self, k):
        order = sorted(range(len(self.cars)), key=lambda i: self.cars[i].fit, reverse=True)
        return order[:k]

    def inject(self, genomes):
        m = min(len(genomes), max(1, self.cfg["population"] // 10))
        for k, g in enumerate(genomes[:m]):
            self.cars[len(self.cars) - 1 - k] = Car(migrate_genome(list(g)), self.track, 0.5)

    # ---- stepping ---------------------------------------------------------------
    def step(self):
        for c in self.cars:
            if c.alive:
                c.step(self)
        if self.ghost is not None and self.ghost.alive:
            self.ghost.step(self)
        if self.shadow is not None:
            for col in self.shadow:
                for c in col:
                    if c.alive:
                        c.step(self)
        pl = self.player
        if pl is not None:
            if pl.alive:
                pl.step(self)
            else:
                if self.respawn_t <= 0.0:
                    self.respawn_t = 1.2
                self.respawn_t -= DT
                if self.respawn_t <= 0.0:
                    pl.reset(self.track)
        self.gen_steps += 1
        if self.recorder is not None:
            self.recorder.sample(self)
        if self.alive_count() == 0 or self.gen_steps >= self.gen_max_steps:
            self.evolve()
        rec = self.recorder
        if rec is not None and rec.stop_after == rec.steps:
            self.recorder = None
            if self.on_replay is not None:
                self.on_replay(rec)

    def begin_recording(self):
        if self.recorder is None:
            self.recorder = Recorder(self)
            return True
        return False

    # ---- evolution ------------------------------------------------------------
    def evolve(self):
        if self.recorder is not None:
            self.recorder.stop_after = self.recorder.steps

        t0 = self.track
        for c in self.cars:
            c.fit = c.progress / t0.n
        if self.shadow:
            for col, trk in zip(self.shadow, self.tracks[1:]):
                tn = trk.n
                for i, c in enumerate(col):
                    laps = c.progress / tn
                    if self.eval_mode == "min":
                        if laps < self.cars[i].fit:
                            self.cars[i].fit = laps
                    else:
                        self.cars[i].fit += laps
            if self.eval_mode == "mean":
                k = 1 + len(self.shadow)
                for c in self.cars:
                    c.fit /= k

        cars = sorted(self.cars, key=lambda c: c.fit, reverse=True)
        champ = cars[0]
        disp = max(self.cars, key=lambda c: c.progress)
        self.last_best_progress = disp.progress
        record = self.best_ever_genome is None or champ.fit > self.best_ever_fit
        if record:
            self.best_ever_fit = champ.fit
            self.best_ever_genome = list(champ.genome)
        if champ.progress > self.best_ever_progress:
            self.best_ever_progress = champ.progress
        ps = sorted(c.progress for c in self.cars)
        m = len(ps)
        med = ps[m // 2] if m % 2 else 0.5 * (ps[m // 2 - 1] + ps[m // 2])
        self.history.append((self.generation, disp.progress, med))
        self.fit_history.append((self.generation, champ.fit))
        if record or self.ghost is None:
            self.ghost = Car(list(self.best_ever_genome), self.track, hue=0.72)
        self.stall = 0 if record else self.stall + 1
        meteor = self.stall >= self.cfg.get("meteor", 25)
        if self.cfg.get("curriculum") and self.difficulty < 0.85 \
                and self.best_ever_fit >= 2.0:
            self.promote()
            return

        rng, P = self.rng, self.cfg["population"]
        rate, strength = self.cfg["mut_rate"], self.cfg["mut_strength"]
        genomes = []

        def tourn(pool):
            b = None
            for _ in range(3):
                c = pool[rng.randrange(len(pool))]
                if b is None or c.fit > b.fit:
                    b = c
            return b.genome

        if self.cfg.get("speciate", True):
            reps = self.species_reps
            if not reps:
                reps.append(list(champ.genome))
                self.species_stall.append(0)
                self.species_best.append(0.0)
            thresh = self.cfg.get("spec_dist", 0.32)
            assign = []
            for c in cars:
                gi = c.genome
                bj, bd = -1, thresh
                for j, rep in enumerate(reps):
                    d = sum(abs(a - b) for a, b in zip(gi, rep)) / len(gi)
                    if d < bd:
                        bd, bj = d, j
                if bj < 0:
                    if len(reps) < 24:
                        reps.append(list(gi))
                        self.species_stall.append(0)
                        self.species_best.append(0.0)
                        bj = len(reps) - 1
                    else:
                        bj = 0
                assign.append(bj)
            species = [[] for _ in reps]
            for c, j in zip(cars, assign):
                species[j].append(c)
            stats, best_sp, bb = [], -1, -1.0
            for s, mem in enumerate(species):
                if not mem:
                    stats.append(None)
                    continue
                bf = max(c.fit for c in mem)
                if bf > self.species_best[s]:
                    self.species_best[s] = bf
                    self.species_stall[s] = 0
                    reps[s] = list(max(mem, key=lambda c: c.fit).genome)
                else:
                    self.species_stall[s] += 1
                stats.append((mem, sum(c.fit for c in mem) / len(mem)))
                if bf > bb:
                    bb, best_sp = bf, s
            total = sum(sc for _, sc in stats if sc is not None) or 1.0
            plan = []
            for s, st in enumerate(stats):
                if st is None:
                    plan.append([s, 0])
                elif self.species_stall[s] > 15 and s != best_sp:
                    plan.append([s, 1])
                else:
                    plan.append([s, max(1, int(round(P * st[1] / total)))])
            over = sum(q for _, q in plan) - P
            while over > 0:
                plan.sort(key=lambda e: -e[1])
                if plan[0][1] <= 1:
                    break
                plan[0][1] -= 1
                over -= 1
            while over < 0:
                plan.sort(key=lambda e: -e[1])
                plan[0][1] += 1
                over += 1
            for s, q in plan:
                if q <= 0:
                    continue
                mem = sorted(species[s], key=lambda c: c.fit, reverse=True)
                genomes.append(list(mem[0].genome))
                pool = mem if len(mem) >= 4 else cars[:max(4, P // 10)]
                got = 1
                while got < q:
                    if meteor and rng.random() < 0.30:
                        genomes.append([rng.gauss(0.0, 0.6) for _ in range(GENOME_SIZE)])
                    else:
                        p1, p2 = tourn(pool), tourn(pool)
                        child = [a if rng.random() < 0.5 else b for a, b in zip(p1, p2)]
                        for i in range(GENOME_SIZE):
                            if rng.random() < rate:
                                child[i] += rng.gauss(0.0, strength)
                                if rng.random() < 0.06:
                                    child[i] = rng.gauss(0.0, 0.6)
                        genomes.append(child)
                    got += 1
            keep = [s for s, q in plan if q > 0]
            self.species_reps = [reps[s] for s in keep]
            self.species_stall = [self.species_stall[s] for s in keep]
            self.species_best = [self.species_best[s] for s in keep]
        else:
            genomes = [list(c.genome) for c in cars[:max(2, P // 10)]]
            while len(genomes) < P:
                if meteor and rng.random() < 0.30:
                    genomes.append([rng.gauss(0.0, 0.6) for _ in range(GENOME_SIZE)])
                    continue
                p1, p2 = tourn(cars), tourn(cars)
                child = [a if rng.random() < 0.5 else b for a, b in zip(p1, p2)]
                for i in range(GENOME_SIZE):
                    if rng.random() < rate:
                        child[i] += rng.gauss(0.0, strength)
                        if rng.random() < 0.06:
                            child[i] = rng.gauss(0.0, 0.6)
                genomes.append(child)

        self.cars = [Car(g, self.track, i / max(1, P - 1)) for i, g in enumerate(genomes[:P])]
        self._spawn_shadow([c.genome for c in self.cars])
        self.generation += 1
        self.gen_steps = 0
        if meteor:
            self.stall = 0


# ---------------------------------------------------------------------------
# recorder / replay
# ---------------------------------------------------------------------------
class Recorder:
    """Captures one generation: genomes, track spec, per-step state + net I/O."""

    def __init__(self, world):
        self.track_spec = world.track.spec()
        self.genomes = [list(c.genome) for c in world.cars]
        self.P = len(world.cars)
        self.generation = world.generation
        self.difficulty = getattr(world, "difficulty", 0.0)
        self.species = world.ns()
        self.best_progress = 0
        self.best_lap = None
        self.best_lap_gen = None
        self.states = array("f")
        self.alive = bytearray()
        self.progress = array("i")
        self.io = array("f")
        self.steps = 0
        self.stop_after = None

    def sample(self, world):
        for c in world.cars:
            self.states.extend((c.x, c.y, c.head, c.v))
            self.alive.append(1 if c.alive else 0)
            self.progress.append(int(c.progress))
            if c.net is not None and c.net.x is not None:
                self.io.extend(c.net.x)
                self.io.extend(c.net.o)
            else:
                self.io.extend([0.0] * (N_IN + N_OUT))
            if c.alive and c.progress > self.best_progress:
                self.best_progress = c.progress
        self.best_lap = world.best_lap
        self.best_lap_gen = world.best_lap_gen
        self.steps += 1


REPLAY_MAGIC = b"NRR1"


def save_replay(path, rep):
    hdr = {"format": "neuroracer-replay-v1", "dt": DT,
           "n_in": N_IN, "n_out": N_OUT, "hidden": HIDDEN, "ctx": CTX,
           "population": rep.P, "steps": rep.steps, "generation": rep.generation,
           "difficulty": rep.difficulty, "species": rep.species,
           "best_progress": rep.best_progress, "best_lap": rep.best_lap,
           "best_lap_gen": rep.best_lap_gen, "track": rep.track_spec,
           "genomes": [base64.b64encode(array("f", g).tobytes()).decode()
                       for g in rep.genomes]}
    with open(path, "wb") as fh:
        fh.write(REPLAY_MAGIC)
        fh.write(json.dumps(hdr).encode())
        fh.write(b"\n")
        fh.write(rep.states.tobytes())
        fh.write(bytes(rep.alive))
        fh.write(rep.progress.tobytes())
        fh.write(rep.io.tobytes())


def load_replay(path):
    with open(path, "rb") as fh:
        if fh.read(4) != REPLAY_MAGIC:
            raise ValueError("not a NeuroRacer replay")
        hdr = json.loads(fh.readline())
        P, steps = int(hdr["population"]), int(hdr["steps"])
        n_in, n_out = int(hdr["n_in"]), int(hdr["n_out"])
        states = array("f"); states.frombytes(fh.read(steps * P * 4 * 4))
        alive = bytearray(fh.read(steps * P))
        progress = array("i"); progress.frombytes(fh.read(steps * P * 4))
        io = array("f"); io.frombytes(fh.read(steps * P * (n_in + n_out) * 4))
        genomes = []
        for s in hdr["genomes"]:
            a = array("f"); a.frombytes(base64.b64decode(s))
            genomes.append(list(a))
    if n_in != N_IN or n_out != N_OUT or not genomes or len(genomes[0]) != GENOME_SIZE:
        raise ValueError("replay was recorded with a different brain architecture")
    return Replay(hdr, genomes, states, alive, progress, io)


class Replay:
    """Deterministic scrubbable playback; quacks like a World for the views."""

    player = None
    ghost = None
    player_best_lap = None
    respawn_t = 0.0

    def __init__(self, hdr, genomes, states, alive, progress, io):
        self.hdr = hdr
        self.track_spec = hdr["track"]
        self.track = Track.from_spec(self.track_spec)
        self.genomes = genomes
        self.P = P = int(hdr["population"])
        self.steps = int(hdr["steps"])
        self.generation = int(hdr["generation"])
        self.difficulty = float(hdr.get("difficulty", 0.0))
        self.species = int(hdr.get("species", 1))
        self.best_progress = float(hdr.get("best_progress", 0.0))
        self.best_lap = hdr.get("best_lap")
        self.best_lap_gen = hdr.get("best_lap_gen")
        self.states, self.alive = states, alive
        self.progress, self.io = progress, io
        self.gen_max_steps = self.steps
        self.track_version = 1 << 20
        self.frame = -1
        self.gen_steps = 0
        self.cars = []
        for i, g in enumerate(genomes):
            c = Car(list(g), self.track, i / max(1, P - 1))
            c.trail = []
            self.cars.append(c)
        for i, c in enumerate(self.cars):
            col = bytes(self.alive[i::P])
            fd = col.find(b"\x00")
            if fd < 0:
                c.death_x = c.death_y = None
            else:
                j = (fd * P + i) * 4
                c.death_x, c.death_y = states[j], states[j + 1]
        self._leader = None
        self._alivect = 0
        self.set_frame(0)

    @classmethod
    def from_recorder(cls, rec):
        hdr = {"population": rec.P, "steps": rec.steps, "generation": rec.generation,
               "difficulty": rec.difficulty, "species": rec.species,
               "best_progress": rec.best_progress, "best_lap": rec.best_lap,
               "best_lap_gen": rec.best_lap_gen, "track": rec.track_spec}
        return cls(hdr, rec.genomes, rec.states, rec.alive, rec.progress, rec.io)

    def ns(self):
        return self.species

    def alive_count(self):
        return self._alivect

    def leader(self):
        return self._leader

    def set_frame(self, t):
        t = max(0, min(self.steps - 1, int(t)))
        self.frame = t
        self.gen_steps = t
        P, base = self.P, t * self.P
        st, al, pr, io = self.states, self.alive, self.progress, self.io
        stride = N_IN + N_OUT
        self._leader, bestp, li = None, None, -1
        for i, c in enumerate(self.cars):
            j = (base + i) * 4
            c.x, c.y, c.head, c.v = st[j], st[j + 1], st[j + 2], st[j + 3]
            c.alive = bool(al[base + i])
            c.progress = pr[base + i]
            k = (base + i) * stride
            c.rec_in = io[k:k + N_IN]
            if c.alive and (bestp is None or c.progress > bestp):
                bestp, self._leader, li = c.progress, c, i
        if self._leader is None:
            bestp = None
            for i, c in enumerate(self.cars):
                if bestp is None or c.progress > bestp:
                    bestp, self._leader, li = c.progress, c, i
        lead = self._leader
        if lead is not None:
            lead.rays = sense_rays(self.track, lead.x, lead.y, lead.head)
            if lead.rec_in is not None:
                lead.net.forward(lead.rec_in)
            frames, f = [], t
            col = self.alive[li::P]
            while f >= 0 and col[f] and len(frames) < 70:
                frames.append(f)
                f -= 1
            frames.reverse()
            lead.trail = []
            for tf in frames:
                j = (tf * P + li) * 4
                lead.trail.append((st[j], st[j + 1]))
        self._alivect = sum(1 for i in range(base, base + P) if al[i])


# ---------------------------------------------------------------------------
# camera mixin (raster + GL views, editor + replay aware)
# ---------------------------------------------------------------------------
class CameraMixin:
    def _init_camera(self, world):
        self.world = world
        self.zoom = 1.0
        r = world.track.rect
        self.camx, self.camy = r.center().x(), r.center().y()
        self._drag = None
        self.follow = False
        self.follow_zoom = True
        self._zoom_touched = False
        self._last_gen = None
        self.follow_changed = None
        self.replay = None
        self._tv = -1
        self.editor_mode = False
        self.epoints = []
        self.estroke = False
        self.epreview = None
        self.eclosed = False
        self.ehw = 46.0

    def _scale(self):
        r = self.world.track.rect
        return min(self.width() / r.width(), self.height() / r.height()) * self.zoom

    def reset_view(self):
        self.zoom = 1.0
        r = self.world.track.rect
        self.camx, self.camy = r.center().x(), r.center().y()
        self.update()

    def _to_world(self, pos):
        s = self._scale()
        return ((pos.x() - self.width() / 2) / s + self.camx,
                (pos.y() - self.height() / 2) / s + self.camy)

    def wheelEvent(self, e):
        wx, wy = self._to_world(e.position())
        self.zoom = max(0.35, min(12.0, self.zoom * 1.002 ** e.angleDelta().y()))
        self._zoom_touched = True
        s = self._scale()
        self.camx = wx - (e.position().x() - self.width() / 2) / s
        self.camy = wy - (e.position().y() - self.height() / 2) / s
        self.update()

    def mousePressEvent(self, e):
        if self.editor_mode:
            if e.button() == Qt.MouseButton.LeftButton:
                self.estroke = True
                wx, wy = self._to_world(e.position())
                self._editor_append(wx, wy, force=True)
                self.update()
            elif e.button() == Qt.MouseButton.RightButton:
                self._drag = e.position()
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.position()
            if self.follow:
                self.follow = False
                if self.follow_changed:
                    self.follow_changed(False)

    def mouseMoveEvent(self, e):
        if self.editor_mode:
            if self.estroke and (e.buttons() & Qt.MouseButton.LeftButton):
                wx, wy = self._to_world(e.position())
                self._editor_append(wx, wy)
                self.update()
            elif self._drag is not None:
                s = self._scale()
                d = e.position() - self._drag
                self.camx -= d.x() / s
                self.camy -= d.y() / s
                self._drag = e.position()
                self.update()
            return
        if self._drag is not None:
            s = self._scale()
            d = e.position() - self._drag
            self.camx -= d.x() / s
            self.camy -= d.y() / s
            self._drag = e.position()
            self.update()

    def mouseReleaseEvent(self, e):
        if self.editor_mode:
            if e.button() == Qt.MouseButton.LeftButton and self.estroke:
                self.estroke = False
                self._editor_preview()
            self._drag = None
            return
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        if self.editor_mode:
            return
        self.reset_view()

    def set_replay(self, rep):
        self.replay = rep
        self._last_gen = None
        self._tv = -1
        self.update()

    # ---- editor ---------------------------------------------------------------
    def editor_start(self):
        self.editor_mode = True
        self.epoints = []
        self.estroke = False
        self.epreview = None
        self.eclosed = False
        self.update()

    def editor_cancel(self):
        self.editor_mode = False
        self.epoints = []
        self.epreview = None
        self.update()

    def editor_width(self, d):
        self.ehw = max(22.0, min(80.0, self.ehw + d))
        self._editor_update_closed()
        self._editor_preview()

    def editor_undo(self):
        if self.estroke:
            self.estroke = False
        if self.epoints:
            self.epoints.pop()
        self._editor_update_closed()
        self._editor_preview()

    def editor_build(self):
        if len(self.epoints) < 8:
            return None, "draw more of the loop first"
        if not self.eclosed:
            return None, "close the loop (finish near the starting point)"
        return track_from_sketch(self.epoints, self.ehw)

    def _editor_update_closed(self):
        pts = self.epoints
        self.eclosed = (len(pts) > 8
                        and math.dist(pts[0], pts[-1]) < max(80.0, self.ehw * 1.6))

    def _editor_append(self, wx, wy, force=False):
        pts = self.epoints
        if not force and pts:
            lx, ly = pts[-1]
            if (wx - lx) ** 2 + (wy - ly) ** 2 < 100.0:
                return
        pts.append((wx, wy))
        self._editor_update_closed()
        if not self.estroke:
            self._editor_preview()

    def _editor_preview(self):
        if len(self.epoints) < 8 or not self.eclosed:
            self.epreview = None
        else:
            try:
                self.epreview = sketch_centerline(self.epoints)
            except Exception:
                self.epreview = None
        self.update()

    def _paint_editor_world(self, p, s):
        x0 = self.camx - self.width() / (2 * s)
        x1 = self.camx + self.width() / (2 * s)
        y0 = self.camy - self.height() / (2 * s)
        y1 = self.camy + self.height() / (2 * s)
        p.setPen(QPen(QColor(255, 255, 255, 14), 1.0 / s))
        g = 100.0
        x = math.floor(x0 / g) * g
        while x <= x1:
            p.drawLine(QPointF(x, y0), QPointF(x, y1))
            x += g
        y = math.floor(y0 / g) * g
        while y <= y1:
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            y += g
        pts = self.epoints
        if self.epreview:
            p.setPen(QPen(QColor(255, 255, 255, 150), 1.6 / s, Qt.PenStyle.DashLine))
            poly = QPolygonF([QPointF(x, y) for x, y in self.epreview])
            poly.append(QPointF(self.epreview[0][0], self.epreview[0][1]))
            p.drawPolyline(poly)
        if pts:
            p.setPen(QPen(QColor(90, 220, 255), 2.2 / s))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
            if not self.eclosed and len(pts) > 1:
                p.setPen(QPen(QColor(90, 220, 255, 90), 1.2 / s, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(pts[-1][0], pts[-1][1]), QPointF(pts[0][0], pts[0][1]))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(90, 220, 255))
            r = 3.5 / s
            for x, y in pts:
                p.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
            p.setBrush(QColor(255, 210, 80))
            p.drawEllipse(QPointF(pts[0][0], pts[0][1]), 5.0 / s, 5.0 / s)

    def _paint_editor_hud(self, p):
        est, pts = 0.0, self.epoints
        for i in range(1, len(pts)):
            est += math.dist(pts[i - 1], pts[i])
        if self.eclosed and len(pts) > 2:
            est += math.dist(pts[-1], pts[0])
        f = self.font()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        lines = [
            "TRACK EDITOR — draw a closed loop with the LEFT button "
            "(draw in the direction cars should drive)",
            "right-drag pan · wheel zoom · Enter build · Backspace undo · "
            "[ ] road width · Esc cancel",
            f"points {len(pts)} · length ~{est:,.0f}px · road width ±{self.ehw:.0f}px"
            + ("  ·  loop CLOSED" if self.eclosed else "  ·  open"),
        ]
        fm = p.fontMetrics()
        h = fm.height() * len(lines) + 16
        wmax = max(fm.horizontalAdvance(t) for t in lines) + 20
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRect(QRectF(10, self.height() - h - 10, wmax, h))
        p.setPen(QColor(140, 230, 255))
        y = self.height() - h - 10 + fm.ascent() + 6
        for t in lines:
            p.drawText(QPointF(20, y))
            y += fm.height()

    # ---- shared camera / HUD -------------------------------------------------
    def update_camera(self):
        w = self.replay if self.replay is not None else self.world
        pl0, lead0 = w.player, w.leader()
        tgt = (pl0 if pl0 is not None else lead0) if self.follow else None
        if w.generation != self._last_gen:
            self._last_gen = w.generation
            if tgt is not None:
                self.camx, self.camy = tgt.x, tgt.y
        if w.track_version != self._tv:
            self._tv = w.track_version
            r = w.track.rect
            self.zoom = 1.0
            self._zoom_touched = False
            self.camx, self.camy = r.center().x(), r.center().y()
        if tgt is not None:
            k = 0.16
            self.camx += (tgt.x - self.camx) * k
            self.camy += (tgt.y - self.camy) * k
            if self.follow_zoom and not self._zoom_touched:
                target = 2.4 - 1.3 * (tgt.v / MAX_SPEED)
                self.zoom += (target - self.zoom) * 0.04
        return pl0, lead0

    @staticmethod
    def _hud_texts(w, lead0, pl0):
        txt = (f"GEN {w.generation}    ALIVE {w.alive_count()}/{len(w.cars)}"
               f"    NEXT GEN IN {max(0.0, w.gen_max_steps * DT - w.gen_steps * DT):3.0f}s"
               f"    DIFF {w.difficulty:.2f}    SPECIES {w.ns()}")
        if pl0 is not None:
            if pl0.alive:
                extra = f"YOU {pl0.v * 0.55:3.0f} km/h"
                if pl0.last_lap is not None:
                    extra += f"    last lap {pl0.last_lap:6.2f}s"
                if pl0.laps:
                    extra += f"    laps {pl0.laps}"
                if w.player_best_lap is not None:
                    extra += f"    your best {w.player_best_lap:6.2f}s"
            else:
                extra = f"CRASHED — respawn {max(0.0, w.respawn_t):.1f}s"
        elif lead0 is not None:
            extra = f"{lead0.v * 0.55:3.0f} km/h"
            if lead0.last_lap is not None:
                extra += f"    last lap {lead0.last_lap:6.2f}s"
            if lead0.laps:
                extra += f"    laps {lead0.laps}"
        else:
            extra = ""
        if w.best_lap is not None:
            extra += f"    AI best lap {w.best_lap:6.2f}s (gen {w.best_lap_gen})"
        return txt, extra


# ---------------------------------------------------------------------------
# raster track view
# ---------------------------------------------------------------------------
class TrackWidget(CameraMixin, QWidget):
    def __init__(self, world, parent=None):
        super().__init__(parent)
        self._init_camera(world)
        self.show_rays = True
        self.show_racing = True
        self.hud_font = QFont()
        self.hud_font.setPointSize(11)
        self.hud_font.setBold(True)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, _evt):
        w = self.replay if self.replay is not None else self.world
        t = w.track
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(15, 17, 21))
        pl0, lead0 = self.update_camera()
        s = self._scale()
        p.save()
        p.translate(self.width() / 2, self.height() / 2)
        p.scale(s, s)
        p.translate(-self.camx, -self.camy)
        lw = 1.5 / s

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(53, 57, 66))
        p.drawPath(t.road_path)

        dash = QPen(QColor(235, 200, 90, 95), lw)
        dash.setDashPattern([3.0, 5.0])
        p.setPen(dash)
        p.drawPolyline(t.center_poly)
        chev = QPen(QColor(255, 255, 255, 34), 2.0 / s)
        chev.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(chev)
        for i in range(0, t.n, max(8, t.n // 16)):
            px, py = t.center[i]
            ax, ay = t.center[(i - 3) % t.n]
            bx, by = t.center[(i + 3) % t.n]
            ang = math.atan2(by - ay, bx - ax)
            ca, sa, k = math.cos(ang), math.sin(ang), 9.0
            p.drawLine(QPointF(px - ca * k - sa * k * 0.6, py - sa * k + ca * k * 0.6), QPointF(px, py))
            p.drawLine(QPointF(px, py), QPointF(px - ca * k + sa * k * 0.6, py - sa * k - ca * k * 0.6))

        if self.show_racing and getattr(t, "racing_path", None):
            p.setPen(QPen(QColor(90, 230, 140, 110), lw * 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(t.racing_path)

        wall = QPen(QColor(222, 226, 235), lw * 1.7)
        wall.setCapStyle(Qt.PenCapStyle.RoundCap)
        wall.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(wall)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(t.poly_outer)
        p.drawPolygon(t.poly_inner)

        sx, sy, sang = t.start
        p.save()
        p.translate(sx, sy)
        p.rotate(math.degrees(sang))
        hw = t.half_width
        sq = (2.0 * hw) / 8.0
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(8):
            p.setBrush(QColor(238, 238, 240) if i % 2 == 0 else QColor(24, 24, 27))
            p.drawRect(QRectF(-3.0, -hw + i * sq, 6.0, sq))
        p.restore()

        p.setPen(Qt.PenStyle.NoPen)
        for c in w.cars:
            if not c.alive and c.death_x is not None:
                p.save()
                p.translate(c.death_x, c.death_y)
                p.rotate(math.degrees(c.head))
                p.setBrush(QColor(225, 80, 80, 38))
                p.drawPolygon(CAR_SHAPE)
                p.restore()

        for c in w.cars:
            if c.alive and c is not lead0:
                self._draw_car(p, c, s, False)
        if lead0 is not None and lead0.alive:
            if self.show_rays:
                self._draw_rays(p, lead0, s)
            if len(lead0.trail) > 1:
                col = QColor.fromHsvF((0.52 + 0.92 * lead0.hue) % 1.0, 0.5, 1.0, 110)
                p.setPen(QPen(col, 2.0 / s))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in lead0.trail]))
            self._draw_car(p, lead0, s, True)

        if w.ghost is not None:
            g = w.ghost
            if len(g.trail) > 1:
                p.setPen(QPen(QColor(255, 255, 255, 46), 1.6 / s))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in g.trail]))
            p.save()
            p.translate(g.x, g.y)
            p.rotate(math.degrees(g.head))
            p.setPen(QPen(QColor(255, 255, 255, 140), 1.4 / s))
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawPolygon(CAR_SHAPE)
            p.restore()

        pl = w.player
        if pl is not None:
            if pl.alive:
                if self.show_rays:
                    for off, r in zip(RAY_ANGLES, pl.rays):
                        ang = pl.head + math.radians(off)
                        d = r * RAY_RANGE
                        p.setPen(QPen(QColor(140, 200, 255, 60), 1.1 / s))
                        p.drawLine(QPointF(pl.x, pl.y),
                                   QPointF(pl.x + math.cos(ang) * d, pl.y + math.sin(ang) * d))
                if len(pl.trail) > 1:
                    p.setPen(QPen(QColor(255, 150, 60, 150), 2.0 / s))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pl.trail]))
                p.save()
                p.translate(pl.x, pl.y)
                p.rotate(math.degrees(pl.head))
                p.setPen(QPen(QColor(255, 255, 255, 240), 2.2 / s))
                p.setBrush(QColor(255, 120, 40))
                p.drawPolygon(CAR_SHAPE)
                p.restore()
            elif pl.death_x is not None:
                p.save()
                p.translate(pl.death_x, pl.death_y)
                p.rotate(math.degrees(pl.head))
                p.setBrush(QColor(225, 80, 80, 90))
                p.drawPolygon(CAR_SHAPE)
                p.restore()

        if self.editor_mode:
            self._paint_editor_world(p, s)
        p.restore()

        if self.editor_mode:
            self._paint_editor_hud(p)
            p.end()
            return

        p.setFont(self.hud_font)
        txt, extra = self._hud_texts(w, lead0, pl0)
        fm = p.fontMetrics()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 130))
        p.drawRect(QRectF(10, 10, fm.horizontalAdvance(txt) + 20, 28))
        p.drawRect(QRectF(10, 42, max(10.0, fm.horizontalAdvance(extra) + 20.0), 22))
        p.setPen(QColor(232, 235, 242))
        p.drawText(QPointF(20, 29), txt)
        p.setPen(QColor(255, 190, 120) if pl0 is not None else QColor(170, 176, 190))
        p.drawText(QPointF(20, 58), extra)
        p.end()

    @staticmethod
    def _draw_car(p, car, s, is_leader):
        p.save()
        p.translate(car.x, car.y)
        p.rotate(math.degrees(car.head))
        if is_leader:
            p.setPen(QPen(QColor(255, 255, 255, 235), 2.4 / s))
        else:
            p.setPen(QPen(QColor(0, 0, 0, 100), 1.0 / s))
        p.setBrush(QColor.fromHsvF((0.52 + 0.92 * car.hue) % 1.0, 0.60, 0.95))
        p.drawPolygon(CAR_SHAPE)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(18, 22, 28, 170))
        p.drawEllipse(QPointF(CAR_LEN * 0.10, 0.0), CAR_W * 0.26, CAR_W * 0.26)
        p.restore()

    @staticmethod
    def _draw_rays(p, car, s):
        for off, r in zip(RAY_ANGLES, car.rays):
            ang = car.head + math.radians(off)
            d = r * RAY_RANGE
            p.setPen(QPen(QColor(int(240 - 150 * r), int(70 + 175 * r), 70, 80), 1.2 / s))
            p.drawLine(QPointF(car.x, car.y),
                       QPointF(car.x + math.cos(ang) * d, car.y + math.sin(ang) * d))


# ---------------------------------------------------------------------------
# neural-network visualiser
# ---------------------------------------------------------------------------
class NNWidget(QWidget):
    IN_LABELS = tuple(f"r{i}" for i in range(len(RAY_ANGLES))) + ("spd", "sD", "cD", "lat")

    def __init__(self, world, parent=None):
        super().__init__(parent)
        self.world = world

    def paintEvent(self, _evt):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(23, 25, 30))
        car = self.world.leader()
        if car is None or car.net is None or car.net.x is None:
            p.setPen(QColor(110, 115, 128))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "waiting for first step…")
            p.end()
            return
        net = car.net
        layers = (net.x, net.h, net.o)
        W, H = float(self.width()), float(self.height())
        xs = (36.0, W * 0.52, W - 54.0)
        maxlen = max(len(l) for l in layers)
        r = max(3.0, min(7.0, (H - 50.0) / (2.4 * maxlen)))
        span = H - 48.0
        pos = [[(xs[li], 24.0 + span * (i + 0.5) / len(layer)) for i in range(len(layer))]
               for li, layer in enumerate(layers)]
        f = self.font()
        f.setPointSize(7)
        p.setFont(f)
        for j, row in enumerate(net.w1):
            for i, wgt in enumerate(row):
                self._conn(p, pos[0][i], pos[1][j], wgt, r)
        for j, row in enumerate(net.w2):
            for i, wgt in enumerate(row):
                self._conn(p, pos[1][i], pos[2][j], wgt, r)
        for li, layer in enumerate(layers):
            for i, a in enumerate(layer):
                self._node(p, pos[li][i][0], pos[li][i][1], a, r)
        p.setPen(QColor(130, 136, 150))
        for i, lab in enumerate(self.IN_LABELS):
            p.drawText(QPointF(3.0, pos[0][i][1] + 3.0), lab)
        p.drawText(QPointF(xs[2] + r + 4.0, pos[2][0][1] + 3.0), "steer")
        p.drawText(QPointF(xs[2] + r + 4.0, pos[2][0][1] + 16.0), f"{net.o[0]:+.2f}")
        p.drawText(QPointF(xs[2] + r + 4.0, pos[2][1][1] + 3.0), "gas/brk")
        p.drawText(QPointF(xs[2] + r + 4.0, pos[2][1][1] + 16.0), f"{net.o[1]:+.2f}")
        for i, a in enumerate(net.ctx):
            self._node(p, 40.0 + i * 16.0, H - 20.0, a, 4.5)
        p.drawText(QPointF(44.0 + len(net.ctx) * 16.0, H - 16.0), "memory")
        p.drawText(QPointF(8.0, H - 34.0),
                   f"gen {self.world.generation} · leader fitness "
                   f"{car.progress * self.world.track.spacing:,.0f} px")
        p.end()

    @staticmethod
    def _conn(p, a, b, wgt, r):
        alpha = min(200, 10 + int(abs(wgt) * 150))
        col = QColor(96, 220, 130, alpha) if wgt >= 0 else QColor(235, 92, 92, alpha)
        p.setPen(QPen(col, 1.0))
        p.drawLine(QPointF(a[0] + r, a[1]), QPointF(b[0] - r, b[1]))

    @staticmethod
    def _node(p, x, y, act, r):
        a = max(-1.0, min(1.0, act))
        if a >= 0:
            col = QColor(int(45 + 210 * a), int(48 + 142 * a), int(55 + 15 * a))
        else:
            t = -a
            col = QColor(int(45 + 35 * t), int(48 + 102 * t), int(55 + 200 * t))
        p.setPen(QPen(QColor(10, 11, 14), 1.0))
        p.setBrush(col)
        p.drawEllipse(QPointF(x, y), r, r)


# ---------------------------------------------------------------------------
# fitness history graph
# ---------------------------------------------------------------------------
class HistoryWidget(QWidget):
    def __init__(self, world, parent=None):
        super().__init__(parent)
        self.world = world
        self.setMinimumHeight(110)

    def paintEvent(self, _evt):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(23, 25, 30))
        hist = self.world.history
        fits = getattr(self.world, "fit_history", [])
        multi = len(fits) == len(hist) and len(fits) > 1
        W, H = self.width(), self.height()
        padl, padr, padt, padb = 34, 8, 12, 16
        if len(hist) < 2:
            p.setPen(QColor(110, 115, 128))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "fitness history — need at least 2 generations")
            p.end()
            return
        tn = self.world.track.n
        gens = [g for g, _, _ in hist]
        g0, g1 = gens[0], gens[-1]
        laps = [b / tn for _, b, _ in hist]
        flaps = [f for _, f in fits] if multi else []
        ytop = max(1.0, max(laps + flaps) * 1.08)

        def X(g):
            return padl + (W - padl - padr) * ((g - g0) / max(1, g1 - g0))

        def Y(l):
            return (H - padb) - (H - padb - padt) * (l / ytop)

        f = self.font()
        f.setPointSize(7)
        p.setFont(f)
        step = 0.5 if ytop <= 6 else (1.0 if ytop <= 12 else 2.0)
        v = step
        while v < ytop:
            y = int(Y(v))
            p.setPen(QColor(45, 48, 58))
            p.drawLine(padl, y, W - padr, y)
            p.setPen(QColor(120, 125, 140))
            p.drawText(QPointF(2, y + 3), f"{v:g}")
            v += step
        p.setPen(QPen(QColor(120, 126, 140, 120), 1.2))
        p.drawPolyline(QPolygonF([QPointF(X(g), Y(md / tn)) for g, _, md in hist]))
        p.setPen(QPen(QColor(86, 156, 255), 1.8))
        p.drawPolyline(QPolygonF([QPointF(X(g), Y(b / tn)) for g, b, _ in hist]))
        rec, run = 0.0, []
        for g, b, _ in hist:
            rec = max(rec, b)
            run.append(QPointF(X(g), Y(rec / tn)))
        p.setPen(QPen(QColor(255, 200, 80), 1.4, Qt.PenStyle.DashLine))
        p.drawPolyline(QPolygonF(run))
        if multi:
            p.setPen(QPen(QColor(90, 220, 255), 1.4, Qt.PenStyle.DashLine))
            p.drawPolyline(QPolygonF([QPointF(X(g), Y(fl)) for (g, _), fl in zip(fits, flaps)]))
        last = hist[-1]
        p.setPen(QColor(232, 235, 242))
        p.drawText(QPointF(W - padr - 92, padt + 2), f"best {last[1] / tn:.2f} laps")
        p.setPen(QColor(255, 200, 80))
        p.drawText(QPointF(padl + 4, padt + 2), f"record {rec / tn:.2f}")
        if multi:
            p.drawText(QPointF(padl + 4, padt + 12), f"fit {flaps[-1]:.2f}")
        p.setPen(QColor(120, 125, 140))
        p.drawText(QPointF(padl, H - 4), f"gen {g0}")
        p.drawText(QPointF(W - padr - 42, H - 4), f"gen {g1}")
        p.end()


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------
class NoWheelSlider(QSlider):
    def wheelEvent(self, e):
        e.ignore()


class NoWheelSpin(QSpinBox):
    def wheelEvent(self, e):
        e.ignore()


class MainWindow(QMainWindow):
    _KEYMAP = {
        Qt.Key.Key_Left: "left",  Qt.Key.Key_A: "left",
        Qt.Key.Key_Right: "right", Qt.Key.Key_D: "right",
        Qt.Key.Key_Up: "gas",     Qt.Key.Key_W: "gas",
        Qt.Key.Key_Down: "brake", Qt.Key.Key_S: "brake",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroRacer — cars evolving to race")
        self.resize(1380, 880)

        self.cfg = {"population": 44, "mut_rate": 0.15, "mut_strength": 0.30,
                    "multi": 1, "speciate": True, "curriculum": False}
        self.world = World(self.cfg)
        self.running = True
        self.speed = 4
        self.keys = set()
        self.replay = None
        self.replay_playing = False
        self._last_replay = None
        self._last_saved_gen = 0
        self.world.on_replay = self._replay_ready

        # view: GL if available, raster fallback
        self.view = None
        if os.environ.get("NEURORACER_GL", "1") != "0":
            try:
                from glview import GLTrackWidget
                self.view = GLTrackWidget(self.world)
            except Exception as exc:
                print(f"GL view unavailable ({exc}); using raster renderer")
        if self.view is None:
            self.view = TrackWidget(self.world)
        self.nn = NNWidget(self.world)
        self.hist = HistoryWidget(self.world)

        panel = QWidget()
        panel.setFixedWidth(340)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(10, 10, 10, 10)
        pv.setSpacing(8)

        title = QLabel("NeuroRacer")
        tf = title.font()
        tf.setPointSize(17)
        tf.setBold(True)
        title.setFont(tf)
        sub = QLabel("neuro-evolution on procedural & drawn tracks")
        sub.setStyleSheet("color:#8a91a3;")
        pv.addWidget(title)
        pv.addWidget(sub)

        gb = QGroupBox("status")
        form = QFormLayout(gb)
        self.lbl_gen, self.lbl_alive = QLabel("1"), QLabel("-")
        self.lbl_spec, self.lbl_diff = QLabel("1"), QLabel("0.00")
        self.lbl_lead, self.lbl_best, self.lbl_last = QLabel("-"), QLabel("-"), QLabel("-")
        self.lbl_stall = QLabel("0")
        self.gen_bar = QProgressBar()
        self.gen_bar.setRange(0, 100)
        self.gen_bar.setFormat("gen time %p%")
        form.addRow("generation", self.lbl_gen)
        form.addRow("cars alive", self.lbl_alive)
        form.addRow("species", self.lbl_spec)
        form.addRow("difficulty", self.lbl_diff)
        form.addRow("leader", self.lbl_lead)
        form.addRow("last gen best", self.lbl_last)
        form.addRow("gens since record", self.lbl_stall)
        form.addRow("all-time best", self.lbl_best)
        form.addRow(self.gen_bar)
        pv.addWidget(gb)

        gb2 = QGroupBox("simulation")
        v2 = QVBoxLayout(gb2)
        row = QHBoxLayout()
        self.btn_pause = self._nf(QPushButton("Pause"))
        self.btn_pause.clicked.connect(self._toggle_pause)
        row.addWidget(self.btn_pause)
        row.addWidget(QLabel("speed ×"))
        self.sl_speed = self._nf(NoWheelSlider(Qt.Orientation.Horizontal))
        self.sl_speed.setRange(1, 40)
        self.sl_speed.setValue(self.speed)
        self.sl_speed.valueChanged.connect(self._speed_changed)
        row.addWidget(self.sl_speed, 1)
        self.lbl_speed = QLabel(str(self.speed))
        row.addWidget(self.lbl_speed)
        v2.addLayout(row)
        self.chk_rays = self._nf(QCheckBox("draw best car's sensor rays"))
        self.chk_rays.setChecked(True)
        self.chk_rays.toggled.connect(lambda on: setattr(self.view, "show_rays", on))
        v2.addWidget(self.chk_rays)
        self.chk_racing = self._nf(QCheckBox("show racing line"))
        self.chk_racing.setChecked(True)
        self.chk_racing.toggled.connect(lambda on: setattr(self.view, "show_racing", on))
        v2.addWidget(self.chk_racing)
        self.chk_fit = self._nf(QCheckBox("fitness on racing line (restart to apply)"))
        self.chk_fit.toggled.connect(lambda on: self.cfg.__setitem__("racing_line", on))
        v2.addWidget(self.chk_fit)
        self.chk_follow = self._nf(QCheckBox("follow camera (F)"))
        self.chk_follow.toggled.connect(self._follow_toggled)
        v2.addWidget(self.chk_follow)
        self.btn_endgen = self._nf(QPushButton("End generation now (E)"))
        self.btn_endgen.clicked.connect(self._end_generation)
        v2.addWidget(self.btn_endgen)
        self.btn_drive = self._nf(QPushButton("Drive yourself (M)"))
        self.btn_drive.clicked.connect(self._toggle_drive)
        v2.addWidget(self.btn_drive)
        self.btn_csv = self._nf(QPushButton("Export fitness CSV"))
        self.btn_csv.clicked.connect(self._export_csv)
        v2.addWidget(self.btn_csv)
        pv.addWidget(gb2)

        gbtr = QGroupBox("track & replay")
        gt = QGridLayout(gbtr)
        self.btn_editor = self._nf(QPushButton("Track editor (T)"))
        self.btn_editor.clicked.connect(self._toggle_editor)
        self.btn_record = self._nf(QPushButton("Record this gen (G)"))
        self.btn_record.clicked.connect(self._record_gen)
        self._record_label = self.btn_record.text()
        gt.addWidget(self.btn_editor, 0, 0)
        gt.addWidget(self.btn_record, 0, 1)
        b_st = self._nf(QPushButton("Save track"))
        b_st.clicked.connect(self._save_track)
        b_lt = self._nf(QPushButton("Load track"))
        b_lt.clicked.connect(self._load_track)
        gt.addWidget(b_st, 1, 0)
        gt.addWidget(b_lt, 1, 1)
        b_sr = self._nf(QPushButton("Save replay"))
        b_sr.clicked.connect(self._save_replay)
        b_or = self._nf(QPushButton("Open replay (O)"))
        b_or.clicked.connect(self._open_replay)
        gt.addWidget(b_sr, 2, 0)
        gt.addWidget(b_or, 2, 1)
        b_wm = self._nf(QPushButton("Width −"))
        b_wm.clicked.connect(lambda: self.view.editor_width(-4))
        b_wp = self._nf(QPushButton("Width +"))
        b_wp.clicked.connect(lambda: self.view.editor_width(4))
        gt.addWidget(b_wm, 3, 0)
        gt.addWidget(b_wp, 3, 1)
        pv.addWidget(gbtr)

        gb3 = QGroupBox("evolution")
        g3 = QGridLayout(gb3)
        g3.addWidget(QLabel("population"), 0, 0)
        self.spin_pop = self._nf(NoWheelSpin())
        self.spin_pop.setRange(12, 120)
        self.spin_pop.setSingleStep(4)
        self.spin_pop.setValue(self.cfg["population"])
        self.spin_pop.setKeyboardTracking(False)
        self.spin_pop.setToolTip("Applied from the next generation on")
        self.spin_pop.valueChanged.connect(lambda v: self.cfg.__setitem__("population", v))
        g3.addWidget(self.spin_pop, 0, 1, 1, 2)
        g3.addWidget(QLabel("mutation rate"), 1, 0)
        self.sl_rate = self._nf(NoWheelSlider(Qt.Orientation.Horizontal))
        self.sl_rate.setRange(1, 60)
        self.sl_rate.setValue(15)
        self.sl_rate.valueChanged.connect(self._rate_changed)
        self.lbl_rate = QLabel("0.15")
        g3.addWidget(self.sl_rate, 1, 1)
        g3.addWidget(self.lbl_rate, 1, 2)
        g3.addWidget(QLabel("mutation strength"), 2, 0)
        self.sl_str = self._nf(NoWheelSlider(Qt.Orientation.Horizontal))
        self.sl_str.setRange(5, 100)
        self.sl_str.setValue(30)
        self.sl_str.valueChanged.connect(self._strength_changed)
        self.lbl_str = QLabel("0.30")
        g3.addWidget(self.sl_str, 2, 1)
        g3.addWidget(self.lbl_str, 2, 2)
        b1 = QHBoxLayout()
        b_restart = self._nf(QPushButton("Restart evolution"))
        b_restart.clicked.connect(self._restart)
        b_track = self._nf(QPushButton("New track"))
        b_track.clicked.connect(self._new_track)
        b1.addWidget(b_restart)
        b1.addWidget(b_track)
        g3.addLayout(b1, 3, 0, 1, 3)
        b2 = QHBoxLayout()
        b_save = self._nf(QPushButton("Save best brain"))
        b_save.clicked.connect(self._save_brain)
        b_load = self._nf(QPushButton("Load brain"))
        b_load.clicked.connect(self._load_brain)
        b2.addWidget(b_save)
        b2.addWidget(b_load)
        g3.addLayout(b2, 4, 0, 1, 3)
        g3.addWidget(QLabel("gen length (s)"), 5, 0)
        self.spin_glen = self._nf(NoWheelSpin())
        self.spin_glen.setRange(10, 180)
        self.spin_glen.setValue(GEN_SECONDS)
        self.spin_glen.setKeyboardTracking(False)
        self.spin_glen.setToolTip("Longer generations let cars complete full laps")
        self.spin_glen.valueChanged.connect(self._glen_changed)
        g3.addWidget(self.spin_glen, 5, 1, 1, 2)
        b3 = QHBoxLayout()
        b_savepop = self._nf(QPushButton("Save population"))
        b_savepop.clicked.connect(self._save_population)
        b_loadpop = self._nf(QPushButton("Load population"))
        b_loadpop.clicked.connect(self._load_population)
        b3.addWidget(b_savepop)
        b3.addWidget(b_loadpop)
        g3.addLayout(b3, 6, 0, 1, 3)
        self.chk_spec = self._nf(QCheckBox("speciation (fitness sharing)"))
        self.chk_spec.setChecked(True)
        self.chk_spec.toggled.connect(lambda on: self.cfg.__setitem__("speciate", on))
        g3.addWidget(self.chk_spec, 7, 0, 1, 3)
        self.chk_curr = self._nf(QCheckBox("auto curriculum (next tier at 2 laps)"))
        self.chk_curr.toggled.connect(lambda on: self.cfg.__setitem__("curriculum", on))
        g3.addWidget(self.chk_curr, 8, 0, 1, 2)
        b_diff = self._nf(QPushButton("Diff +"))
        b_diff.clicked.connect(lambda: self.world.promote())
        g3.addWidget(b_diff, 8, 2)
        g3.addWidget(QLabel("exam tracks"), 9, 0)
        self.spin_multi = self._nf(NoWheelSpin())
        self.spin_multi.setRange(1, 6)
        self.spin_multi.setValue(1)
        self.spin_multi.setKeyboardTracking(False)
        self.spin_multi.setToolTip(
            "1 = one circuit (specialists). 3+ = every genome scored on N circuits "
            "per generation; selection favours generalists. Applied on Restart (R).")
        self.spin_multi.valueChanged.connect(lambda v: self.cfg.__setitem__("multi", v))
        g3.addWidget(self.spin_multi, 9, 1)
        g3.addWidget(QLabel("aggregation"), 10, 0)
        self.cmb_mode = self._nf(QComboBox())
        self.cmb_mode.addItems(["mean laps", "min laps"])
        self.cmb_mode.setToolTip(
            "'mean' = average laps over the exam tracks. 'min' = worst track — "
            "harshest generalist pressure; switch once cars reliably finish laps. "
            "Applied on Restart (R).")
        self.cmb_mode.currentIndexChanged.connect(
            lambda i: self.cfg.__setitem__("eval_mode", ("mean", "min")[i]))
        g3.addWidget(self.cmb_mode, 10, 1, 1, 2)
        pv.addWidget(gb3)

        pv.addWidget(QLabel("fitness per generation"))
        pv.addWidget(self.hist)
        pv.addWidget(QLabel("best car's brain (live)"))
        self.nn.setMinimumHeight(180)
        pv.addWidget(self.nn, 1)

        # replay scrub bar
        self.replay_bar = QWidget()
        rb = QHBoxLayout(self.replay_bar)
        rb.setContentsMargins(8, 4, 8, 4)
        self.btn_rlive = self._nf(QPushButton("✕ live"))
        self.btn_rlive.clicked.connect(self._exit_replay)
        self.btn_rplay = self._nf(QPushButton("⏸"))
        self.btn_rplay.clicked.connect(self._toggle_replay_play)
        self.scrub = self._nf(NoWheelSlider(Qt.Orientation.Horizontal))
        self.scrub.setRange(0, 0)
        self.scrub.valueChanged.connect(self._scrub_to)
        self.lbl_frames = QLabel("0 / 0")
        self.lbl_frames.setMinimumWidth(110)
        rb.addWidget(self.btn_rlive)
        rb.addWidget(self.btn_rplay)
        rb.addWidget(self.scrub, 1)
        rb.addWidget(self.lbl_frames)
        self.replay_bar.hide()

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(self.view, 1)
        lv.addWidget(self.replay_bar)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(left, 1)
        lay.addWidget(panel)
        self.setCentralWidget(central)

        self.statusBar().showMessage(
            "Space pause · N new track · T track editor · G record gen · O replay · "
            "R restart · E end gen · F follow · M drive · +/− speed · wheel zoom · "
            "drag pan · dbl-click reset")

        # resume champion if present
        champ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion.json")
        if os.path.exists(champ):
            try:
                with open(champ) as fh:
                    data = json.load(fh)
                genome = migrate_genome(data["genome"])
                if int(data.get("hidden", -1)) == HIDDEN and len(genome) == GENOME_SIZE:
                    self.world.load_genome(genome, float(data.get("best_progress_points", 0.0)))
            except Exception:
                pass

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    @staticmethod
    def _nf(w):
        w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return w

    # ---- sim controls ---------------------------------------------------------
    def _toggle_pause(self):
        self.running = not self.running
        self.btn_pause.setText("Play" if not self.running else "Pause")

    def _speed_changed(self, v):
        self.speed = v
        self.lbl_speed.setText(str(v))

    def _rate_changed(self, v):
        self.cfg["mut_rate"] = v / 100.0
        self.lbl_rate.setText(f"{v / 100.0:.2f}")

    def _strength_changed(self, v):
        self.cfg["mut_strength"] = v / 100.0
        self.lbl_str.setText(f"{v / 100.0:.2f}")

    def _follow_toggled(self, on):
        self.view.follow = on
        self.view._zoom_touched = False

    def _end_generation(self):
        self.world.evolve()

    def _glen_changed(self, v):
        self.world.gen_max_steps = max(1, int(v / DT))

    def _restart(self):
        self.view.editor_cancel()
        w = self.world
        if w.multi != self.cfg["multi"]:
            w.multi = max(1, int(self.cfg["multi"]))
            w._build_tracks()
        w.eval_mode = self.cfg.get("eval_mode", "mean")
        w._apply_fit()
        w.restart_evolution()
        self.view.update()

    def _new_track(self):
        self.view.editor_cancel()
        self._exit_replay()
        self.world.new_track()
        self.view.reset_view()

    def _toggle_drive(self):
        self.view.editor_cancel()
        on = self.world.toggle_player()
        self.btn_drive.setText("Stop driving (M)" if on else "Drive yourself (M)")
        self.statusBar().showMessage(
            "drive: WASD / arrows · F follow cam · M stop driving", 5000)

    # ---- persistence ------------------------------------------------------------
    def _autosave(self):
        w = self.world
        if w.best_ever_genome is None:
            return
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion.json")
        try:
            with open(path, "w") as fh:
                json.dump({"format": "neuroracer-brain-v2", "n_in": N_IN, "hidden": HIDDEN,
                           "n_out": N_OUT, "ctx": CTX,
                           "genome": [float(g) for g in w.best_ever_genome],
                           "best_progress_points": float(w.best_ever_progress),
                           "saved_at_generation": w.generation}, fh)
            self.statusBar().showMessage(
                f"champion auto-saved (gen {w.generation})", 4000)
        except OSError:
            pass

    def _save_brain(self):
        w = self.world
        genome = w.best_ever_genome
        if genome is None and w.leader() is not None:
            genome = w.leader().genome
        if genome is None:
            QMessageBox.information(self, "NeuroRacer", "Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save best brain", "best_brain.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w") as fh:
            json.dump({"format": "neuroracer-brain-v2", "n_in": N_IN, "hidden": HIDDEN,
                       "n_out": N_OUT, "ctx": CTX,
                       "genome": [float(g) for g in genome],
                       "best_progress_points": float(w.best_ever_progress)}, fh)

    def _load_brain(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load brain", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            genome = migrate_genome(data["genome"])
            if int(data.get("hidden", -1)) != HIDDEN or len(genome) != GENOME_SIZE:
                raise ValueError("brain architecture mismatch")
            self.world.load_genome(genome, float(data.get("best_progress_points", 0.0)))
        except Exception as exc:
            QMessageBox.warning(self, "NeuroRacer", f"Could not load brain:\n{exc}")

    def _save_population(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save population",
                                              "population.json", "JSON (*.json)")
        if path:
            try:
                self.world.save_population(path)
            except OSError as exc:
                QMessageBox.warning(self, "NeuroRacer", f"Could not save:\n{exc}")

    def _load_population(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load population", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.world.load_population(path)
            self.spin_pop.setValue(self.cfg["population"])
            self.view.reset_view()
        except Exception as exc:
            QMessageBox.warning(self, "NeuroRacer", f"Could not load:\n{exc}")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export fitness history",
                                              "fitness.csv", "CSV (*.csv)")
        if not path:
            return
        t = self.world.track
        with open(path, "w") as fh:
            fh.write("generation,best_points,median_points,best_laps,median_laps\n")
            for g, b, m in self.world.history:
                fh.write(f"{g},{b:.1f},{m:.1f},{b / t.n:.4f},{m / t.n:.4f}\n")

    # ---- track editor -----------------------------------------------------------
    def _toggle_editor(self):
        if self.view.editor_mode:
            self.view.editor_cancel()
            self.statusBar().showMessage("editor closed", 2000)
        else:
            self._exit_replay()
            self.view.editor_start()
            self.statusBar().showMessage(
                "track editor: draw a closed loop · Enter builds and trains on it", 6000)

    def _editor_commit(self):
        track, err = self.view.editor_build()
        if track is None:
            self.statusBar().showMessage(f"editor: {err}", 5000)
            return
        self._exit_replay()
        self.world.install_track(track, reseed=True)
        self.view.editor_cancel()
        self.view.reset_view()
        self._sync()
        self.statusBar().showMessage(
            f"custom track installed: {track.n} samples · {track.length:,.0f}px · "
            f"population re-seeded from the champion", 7000)

    def _save_track(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save track", "my_track.json",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w") as fh:
                json.dump({"format": "neuroracer-track-v1",
                           "track": self.world.track.spec()}, fh)
        except OSError as exc:
            QMessageBox.warning(self, "NeuroRacer", f"Could not save track:\n{exc}")

    def _load_track(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load track", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            track = Track.from_spec(data["track"])
            self._exit_replay()
            self.world.install_track(track, reseed=True)
            self.view.editor_cancel()
            self.view.reset_view()
        except Exception as exc:
            QMessageBox.warning(self, "NeuroRacer", f"Could not load track:\n{exc}")

    # ---- recording & replay -------------------------------------------------------
    def _record_gen(self):
        if self.world.begin_recording():
            self.statusBar().showMessage(
                "recording — the scrubber opens when this generation ends", 4000)

    def _replay_ready(self, rec):
        self._enter_replay(Replay.from_recorder(rec))
        self.statusBar().showMessage(
            f"recorded gen {rec.generation} ({rec.steps} frames) — Space play/pause · "
            "←/→ step · drag slider · Esc back to live", 8000)

    def _enter_replay(self, rep):
        if self.view.editor_mode:
            self.view.editor_cancel()
        self.replay = rep
        self._last_replay = rep
        self.replay_playing = True
        self.view.set_replay(rep)
        self.nn.world = rep
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, max(0, rep.steps - 1))
        self.scrub.setValue(0)
        self.scrub.blockSignals(False)
        self.lbl_frames.setText(f"0 / {rep.steps - 1}")
        self.btn_rplay.setText("⏸")
        self.replay_bar.show()
        self.view.update()
        self.nn.update()

    def _exit_replay(self):
        if self.replay is None:
            return
        self.replay = None
        self.replay_playing = False
        self.view.set_replay(None)
        self.nn.world = self.world
        self.replay_bar.hide()
        self.view.update()
        self.nn.update()
        self.hist.update()

    def _toggle_replay_play(self):
        if self.replay is None:
            return
        if not self.replay_playing and self.replay.frame >= self.replay.steps - 1:
            self.replay.set_frame(0)
        self.replay_playing = not self.replay_playing
        self.btn_rplay.setText("⏸" if self.replay_playing else "▶")

    def _scrub_to(self, v):
        if self.replay is None:
            return
        self.replay.set_frame(v)
        self.lbl_frames.setText(f"{v} / {self.replay.steps - 1}")
        self.view.update()
        self.nn.update()

    def _scrub(self, d):
        if self.replay is None:
            return
        self.replay_playing = False
        self.btn_rplay.setText("▶")
        self.scrub.setValue(self.replay.frame + d)

    def _save_replay(self):
        rep = self.replay if self.replay is not None else self._last_replay
        if rep is None:
            QMessageBox.information(self, "NeuroRacer",
                                    "Nothing recorded yet — press G during a generation.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save replay", f"generation_{rep.generation:04d}.nrr", "Replay (*.nrr)")
        if path:
            try:
                save_replay(path, rep)
            except OSError as exc:
                QMessageBox.warning(self, "NeuroRacer", f"Could not save:\n{exc}")

    def _open_replay(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open replay", "", "Replay (*.nrr)")
        if not path:
            return
        try:
            rep = load_replay(path)
        except Exception as exc:
            QMessageBox.warning(self, "NeuroRacer", f"Could not load replay:\n{exc}")
            return
        self._enter_replay(rep)

    # ---- main loop ---------------------------------------------------------------
    def _tick(self):
        if self.replay is not None:
            if self.replay_playing:
                nxt = self.replay.frame + 1
                if nxt >= self.replay.steps:
                    self.replay_playing = False
                    self.btn_rplay.setText("▶")
                else:
                    self.replay.set_frame(nxt)
                    self.scrub.blockSignals(True)
                    self.scrub.setValue(nxt)
                    self.scrub.blockSignals(False)
                    self.lbl_frames.setText(f"{nxt} / {self.replay.steps - 1}")
            self.view.update()
            self.nn.update()
            return
        if self.view.editor_mode:
            self.view.update()
            return
        if self.running:
            for _ in range(self.speed):
                self.world.step()
                if self.replay is not None:
                    break
        if self.world.recorder is not None:
            self.btn_record.setText(f"recording gen {self.world.generation}…")
        elif self.btn_record.text() != self._record_label:
            self.btn_record.setText(self._record_label)
        if self.world.generation != self._last_saved_gen:
            self._last_saved_gen = self.world.generation
            if self._last_saved_gen % 10 == 0:
                self._autosave()
        self._sync()
        self.view.update()
        self.nn.update()
        self.hist.update()

    def _sync(self):
        w, t = self.world, self.world.track
        self.lbl_gen.setText(str(w.generation))
        self.lbl_alive.setText(f"{w.alive_count()} / {len(w.cars)}")
        self.lbl_spec.setText(str(w.ns()))
        self.lbl_diff.setText(f"{w.difficulty:.2f}")
        self.lbl_stall.setText(str(w.stall))
        self.gen_bar.setValue(int(100 * w.gen_steps / w.gen_max_steps))
        lead = w.leader()
        self.lbl_lead.setText("-" if lead is None else f"{lead.progress / t.n:5.2f} laps")
        self.lbl_last.setText(f"{w.last_best_progress / t.n:5.2f} laps")
        if w.multi > 1:
            self.lbl_best.setText(f"{w.best_ever_fit:5.2f} avg laps")
        else:
            self.lbl_best.setText(f"{w.best_ever_progress / t.n:5.2f} laps")

    # ---- keys ---------------------------------------------------------------------
    def keyPressEvent(self, e):
        k = e.key()
        if self.replay is not None:
            if k == Qt.Key.Key_Escape:
                self._exit_replay()
            elif k == Qt.Key.Key_Space:
                self._toggle_replay_play()
            elif k == Qt.Key.Key_Left:
                self._scrub(-1)
            elif k == Qt.Key.Key_Right:
                self._scrub(1)
            return
        if self.view.editor_mode:
            if k == Qt.Key.Key_Escape:
                self._toggle_editor()
            elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._editor_commit()
            elif k == Qt.Key.Key_Backspace:
                self.view.editor_undo()
            elif k == Qt.Key.Key_BracketLeft:
                self.view.editor_width(-4)
            elif k == Qt.Key.Key_BracketRight:
                self.view.editor_width(4)
            return
        if k in self._KEYMAP:
            self.keys.add(self._KEYMAP[k])
            return
        if k == Qt.Key.Key_Space:
            self._toggle_pause()
        elif k == Qt.Key.Key_N:
            self._new_track()
        elif k == Qt.Key.Key_R:
            self._restart()
        elif k == Qt.Key.Key_F:
            self.chk_follow.setChecked(not self.chk_follow.isChecked())
        elif k == Qt.Key.Key_E:
            self._end_generation()
        elif k == Qt.Key.Key_M:
            self._toggle_drive()
        elif k == Qt.Key.Key_T:
            self._toggle_editor()
        elif k == Qt.Key.Key_G:
            self._record_gen()
        elif k == Qt.Key.Key_O:
            self._open_replay()
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.sl_speed.setValue(min(40, self.sl_speed.value() + 1))
        elif k == Qt.Key.Key_Minus:
            self.sl_speed.setValue(max(1, self.sl_speed.value() - 1))
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() in self._KEYMAP:
            self.keys.discard(self._KEYMAP[e.key()])
        else:
            super().keyReleaseEvent(e)


def _apply_dark_palette(app):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(30, 32, 38))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(221, 224, 231))
    pal.setColor(QPalette.ColorRole.Base, QColor(24, 26, 31))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(34, 37, 44))
    pal.setColor(QPalette.ColorRole.Text, QColor(221, 224, 231))
    pal.setColor(QPalette.ColorRole.Button, QColor(44, 47, 55))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(221, 224, 231))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(74, 130, 250))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(48, 51, 60))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(228, 231, 238))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 124, 132))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(120, 124, 132))
    app.setPalette(pal)


def main():
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
