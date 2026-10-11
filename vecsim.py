#!/usr/bin/env python
"""Vectorised (numpy) NeuroRacer core: recurrence, speciation, curriculum,
multi-track fitness. Genomes are interchangeable with neuroracer.py."""
import math

import numpy as np

import neuroracer as nr

_OFF = np.arange(-16, 17, dtype=np.int32)
_RAY_OFF = np.radians(np.array(nr.RAY_ANGLES, np.float32))
_STEP = 7.0
_NSTEPS = int(math.ceil(nr.RAY_RANGE / _STEP))
_NRAY = len(nr.RAY_ANGLES)
_NR = nr.N_IN + nr.CTX
_O1 = nr.HIDDEN * _NR
_O2 = _O1 + nr.HIDDEN
_O3 = _O2 + nr.HIDDEN * nr.N_OUT
_O4 = _O3 + nr.N_OUT
_O5 = _O4 + nr.CTX * nr.HIDDEN


class VecWorld:
    def __init__(self, cfg, track=None, track_seed=None, evo_seed=None, difficulty=0.0):
        self.cfg = cfg
        for _k, _v in (("population", 44), ("mut_rate", 0.15), ("mut_strength", 0.30)):
            self.cfg.setdefault(_k, _v)
        self.rng = np.random.default_rng(evo_seed)
        self.difficulty = difficulty
        self.multi = max(1, int(cfg.get("multi", 1)))
        self.eval_mode = cfg.get("eval_mode", "mean")
        self._track_arg = track
        self._exam_base = (track_seed if track_seed is not None
                           else int(self.rng.integers(1, 10 ** 9)))
        self.extra = []
        if self.multi > 1:
            self.extra = [nr.Track((self._exam_base + 0x9E3779B9 * i) % (10 ** 9),
                                   difficulty=difficulty)
                          for i in range(1, self.multi)]
        self.track = track if track is not None else nr.Track(self._exam_base,
                                                              difficulty=difficulty)
        self._apply_fit()
        self._prepare_track()
        self._no_lap = False
        self.best_ever_genome = None
        self.best_ever_progress = 0.0
        self.best_ever_fit = 0.0
        self.last_best_progress = 0
        self.best_lap = None
        self.gen_max_steps = nr.GEN_MAX_STEPS
        self.species_reps = []
        self.species_stall = []
        self.species_best = []
        self.restart_evolution()

    # ---- track -> arrays -----------------------------------------------------
    def _apply_fit(self, track=None):
        t = track if track is not None else self.track
        t.fit = t.racing if (self.cfg.get("racing_line") and getattr(t, "racing", None)) else t.center

    def _prepare_track(self, track=None):
        t = track if track is not None else self.track
        self.n = len(t.fit)
        self.cx = np.array([p[0] for p in t.fit], np.float32)
        self.cy = np.array([p[1] for p in t.fit], np.float32)
        ax, bx = np.roll(self.cx, 1), np.roll(self.cx, -1)
        ay, by = np.roll(self.cy, 1), np.roll(self.cy, -1)
        tx, ty = bx - ax, by - ay
        L = np.hypot(tx, ty)
        L[L == 0] = 1.0
        self.tx, self.ty = (tx / L).astype(np.float32), (ty / L).astype(np.float32)
        g = t.grid
        self.gx0, self.gy0, self.gw, self.gh = g.x0, g.y0, g.gw, g.gh
        self.inv = np.float32(g.inv)
        self.margin = np.float32(g.margin)
        self.cells = np.frombuffer(bytes(g.cells), np.uint8).reshape(g.gh, g.gw)
        self.df = np.asarray(g.df, np.float32).reshape(g.gh, g.gw)
        self._flat = self.cells.ravel()
        self._dfflat = self.df.ravel()

    # ---- population ------------------------------------------------------------
    def restart_evolution(self):
        P = self.cfg["population"]
        self.P = P
        self.genomes = (self.rng.standard_normal((P, nr.GENOME_SIZE)) * 0.5).astype(np.float32)
        self._new_gen_arrays()
        self.generation = 1
        self.gen_steps = 0
        self.history = []
        self.fit_history = []
        self.stall = 0
        self.best_ever_fit = 0.0
        self.species_reps = []
        self.species_stall = []
        self.species_best = []

    def _new_gen_arrays(self):
        P = self.P
        sx, sy, sh = self.track.start
        z = np.zeros(P, np.float32)
        self.x = np.full(P, sx, np.float32)
        self.y = np.full(P, sy, np.float32)
        self.head = np.full(P, sh, np.float32)
        self.v = z.copy()
        self.ci = np.zeros(P, np.int32)
        self.progress = np.zeros(P, np.int64)
        self.alive = np.ones(P, bool)
        self.stuck = z.copy()
        self.age = z.copy()
        self.lap_start = z.copy()
        self.li = np.zeros(P, np.int64)
        self.ctx = np.zeros((P, nr.CTX), np.float32)
        self.W1 = self.genomes[:, :_O1].reshape(P, nr.HIDDEN, _NR)
        self.B1 = self.genomes[:, _O1:_O2]
        self.W2 = self.genomes[:, _O2:_O3].reshape(P, nr.N_OUT, nr.HIDDEN)
        self.B2 = self.genomes[:, _O3:_O4]
        self.WC = self.genomes[:, _O4:_O5].reshape(P, nr.CTX, nr.HIDDEN)
        self.BC = self.genomes[:, _O5:]

    def alive_count(self):
        return int(self.alive.sum())

    def ns(self):
        return len(self.species_reps)

    def get_genome(self, i):
        return [float(v) for v in self.genomes[i]]

    def top_indices(self, k):
        return np.argsort(-self.progress, kind="stable")[:k].tolist()

    def inject(self, genomes):
        m = min(len(genomes), max(1, self.P // 10))
        for k, g in enumerate(genomes[:m]):
            self.genomes[self.P - 1 - k] = np.asarray(nr.migrate_genome(list(g)), np.float32)
        self._new_gen_arrays()

    def load_genome(self, genome, fitness):
        genome = nr.migrate_genome(list(genome))
        P = self.cfg["population"]
        base = np.asarray(genome, np.float32)
        arr = np.repeat(base[None, :], P, axis=0)
        jit = self.rng.random((P, nr.GENOME_SIZE)) < 0.10
        jit[0] = False
        arr[jit] += (self.rng.standard_normal(int(jit.sum())) * 0.10).astype(np.float32)
        self.genomes = arr
        self.best_ever_genome = [float(v) for v in genome]
        self.best_ever_progress = float(fitness)
        self.best_ever_fit = float(fitness) / self.n
        self.generation = 1
        self.gen_steps = 0
        self.history = []
        self.fit_history = []
        self.stall = 0
        self.last_best_progress = 0
        self.best_lap = None
        self.species_reps = []
        self.species_stall = []
        self.species_best = []
        self._new_gen_arrays()

    # ---- exam evaluation ---------------------------------------------------------
    def _eval_on(self, track):
        saved = (self.track, self.n, self.cx, self.cy, self.tx, self.ty,
                 self.gx0, self.gy0, self.gw, self.gh, self.inv, self.margin,
                 self.cells, self.df, self._flat, self._dfflat)
        self._no_lap = True
        self.track = track
        self._apply_fit(track)
        self._prepare_track(track)
        self._new_gen_arrays()
        for _ in range(self.gen_max_steps):
            if not self.alive.any():
                break
            self._physics()
        laps = (self.progress / np.float64(self.n)).astype(np.float64)
        (self.track, self.n, self.cx, self.cy, self.tx, self.ty,
         self.gx0, self.gy0, self.gw, self.gh, self.inv, self.margin,
         self.cells, self.df, self._flat, self._dfflat) = saved
        self._no_lap = False
        return laps

    # ---- stepping ------------------------------------------------------------------
    def step(self):
        if self.alive.any():
            self._physics()
        self.gen_steps += 1
        if not self.alive.any() or self.gen_steps >= self.gen_max_steps:
            self.evolve()

    def _physics(self):
        P, n = self.P, self.n
        idx = (self.ci[:, None] + _OFF[None, :]) % n
        dx = self.cx[idx] - self.x[:, None]
        dy = self.cy[idx] - self.y[:, None]
        bi = idx[np.arange(P), np.argmin(dx * dx + dy * dy, axis=1)]
        delta = bi - self.ci
        delta = np.where(delta > n // 2, delta - n, delta)
        delta = np.where(delta < -(n // 2), delta + n, delta)
        self.progress += delta
        self.ci = bi.astype(np.int32)

        li = self.progress // n
        lap = (li > self.li) & (li >= 1) & ((self.age - self.lap_start) > 1.0)
        if lap.any():
            if not self._no_lap:
                lt = float((self.age[lap] - self.lap_start[lap]).min())
                if self.best_lap is None or lt < self.best_lap:
                    self.best_lap = lt
            self.lap_start[lap] = self.age[lap]
        self.li = li
        self.age += nr.DT

        tx, ty = self.tx[bi], self.ty[bi]
        ddx = self.x - self.cx[bi]
        ddy = self.y - self.cy[bi]
        lat = (tx * ddy - ty * ddx) / np.float32(self.track.half_width)

        ang = self.head[:, None] + _RAY_OFF[None, :]
        rdx, rdy = np.cos(ang), np.sin(ang)
        ds = np.arange(1, _NSTEPS + 1, dtype=np.float32) * _STEP
        px = self.x[:, None, None] + rdx[:, :, None] * ds[None, None, :]
        py = self.y[:, None, None] + rdy[:, :, None] * ds[None, None, :]
        gxi = ((px - self.gx0) * self.inv).astype(np.int32)
        gyi = ((py - self.gy0) * self.inv).astype(np.int32)
        valid = (gxi >= 0) & (gxi < self.gw) & (gyi >= 0) & (gyi < self.gh)
        blocked = ~(valid & self._flat[np.clip(gyi, 0, self.gh - 1) * self.gw
                                       + np.clip(gxi, 0, self.gw - 1)].astype(bool))
        hit = blocked.any(axis=2)
        khit = blocked.argmax(axis=2)
        dprev = khit.astype(np.float32) * _STEP
        fx = np.clip(((self.x[:, None] + rdx * dprev - self.gx0) * self.inv).astype(np.int32),
                     0, self.gw - 1)
        fy = np.clip(((self.y[:, None] + rdy * dprev - self.gy0) * self.inv).astype(np.int32),
                     0, self.gh - 1)
        est = dprev + self._dfflat[fy * self.gw + fx] * np.float32(0.85)
        est = np.maximum(est, np.float32(_STEP * 0.5))
        dist = np.where(hit, np.clip(est + self.margin, 0.0, nr.RAY_RANGE),
                        np.float32(nr.RAY_RANGE))

        chd, shd = np.cos(self.head), np.sin(self.head)
        Xf = np.empty((P, _NR), np.float32)
        Xf[:, :_NRAY] = (dist / np.float32(nr.RAY_RANGE)).T
        Xf[:, _NRAY] = self.v / np.float32(nr.MAX_SPEED)
        Xf[:, _NRAY + 1] = shd * tx - chd * ty
        Xf[:, _NRAY + 2] = chd * tx + shd * ty
        Xf[:, _NRAY + 3] = lat
        Xf[:, nr.N_IN:] = self.ctx
        h1 = np.tanh(np.einsum("phn,pn->ph", self.W1, Xf, optimize=True) + self.B1)
        o = np.tanh(np.einsum("pon,ph->po", self.W2, h1, optimize=True) + self.B2)
        self.ctx = np.tanh(np.einsum("pcn,pn->pc", self.WC, h1, optimize=True) + self.BC)
        steer, thr = o[:, 0], o[:, 1]

        v = self.v + np.where(thr > 0, thr * nr.ACCEL, thr * nr.BRAKE) * nr.DT \
            - nr.DRAG * self.v * nr.DT
        np.clip(v, 0.0, nr.MAX_SPEED, out=v)
        self.v = v
        self.head += (steer * nr.TURN_RATE * (v / (v + 70.0)) * nr.DT).astype(np.float32)
        self.x += np.cos(self.head) * v * nr.DT
        self.y += np.sin(self.head) * v * nr.DT

        cgx = ((self.x - self.gx0) * self.inv).astype(np.int32)
        cgy = ((self.y - self.gy0) * self.inv).astype(np.int32)
        on = ((cgx >= 0) & (cgx < self.gw) & (cgy >= 0) & (cgy < self.gh)
              & self._flat[np.clip(cgy, 0, self.gh - 1) * self.gw
                           + np.clip(cgx, 0, self.gw - 1)].astype(bool))
        self.stuck = np.where(v < 22.0, self.stuck + nr.DT,
                              np.maximum(self.stuck - 2.0 * nr.DT, 0.0))
        self.alive &= on & ~(self.stuck > 2.5)

    # ---- evolution -----------------------------------------------------------------
    def evolve(self):
        live_prog = self.progress.copy()
        live_laps = live_prog / np.float64(self.n)
        if self.extra:
            cols = [live_laps] + [self._eval_on(t) for t in self.extra]
            stack = np.stack(cols)
            fit = stack.min(axis=0) if self.eval_mode == "min" else stack.mean(axis=0)
        else:
            fit = live_laps

        order = np.argsort(-fit, kind="stable")
        best, best_i = float(fit[order[0]]), int(order[0])
        self.last_best_progress = float(live_prog.max())
        record = self.best_ever_genome is None or best > self.best_ever_fit
        if record:
            self.best_ever_fit = best
            self.best_ever_genome = [float(g) for g in self.genomes[best_i]]
        if float(live_prog[best_i]) > self.best_ever_progress:
            self.best_ever_progress = float(live_prog[best_i])
        self.history.append((self.generation, float(live_prog.max()),
                             float(np.median(live_prog))))
        self.fit_history.append((self.generation, best))
        self.stall = 0 if record else self.stall + 1
        meteor = self.stall >= self.cfg.get("meteor", 25)

        if self.cfg.get("curriculum") and self.difficulty < 0.85 \
                and self.best_ever_fit >= 2.0:
            self.promote()
            return

        rng, P, G = self.rng, self.P, nr.GENOME_SIZE
        rate, strength = float(self.cfg["mut_rate"]), float(self.cfg["mut_strength"])
        children = []

        if self.cfg.get("speciate", True):
            reps = self.species_reps
            if not reps:
                reps.append(self.genomes[int(order[0])].copy())
                self.species_stall.append(0)
                self.species_best.append(0.0)
            thresh = float(self.cfg.get("spec_dist", 0.32))
            D = np.abs(self.genomes[:, None, :] - np.stack(reps)[None, :, :]).mean(axis=2)
            js, dmin = D.argmin(axis=1), D.min(axis=1)
            for i in np.where(dmin >= thresh)[0]:
                if len(reps) < 24:
                    reps.append(self.genomes[i].copy())
                    self.species_stall.append(0)
                    self.species_best.append(0.0)
                    js[i] = len(reps) - 1
            stats, best_sp, bb = [], -1, -1.0
            for s in range(len(reps)):
                mem = np.where(js == s)[0]
                if len(mem) == 0:
                    stats.append(None)
                    continue
                bfi = int(mem[np.argmax(fit[mem])])
                bf = float(fit[bfi])
                if bf > self.species_best[s]:
                    self.species_best[s] = bf
                    self.species_stall[s] = 0
                    reps[s] = self.genomes[bfi].copy()
                else:
                    self.species_stall[s] += 1
                stats.append((mem, float(fit[mem].sum()) / len(mem)))
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
                mem, _ = stats[s]
                idx = mem[np.argsort(-fit[mem], kind="stable")]
                children.append(self.genomes[int(idx[0])].copy())
                if len(idx) >= 4:
                    pool, pfit = self.genomes[idx], fit[idx]
                else:
                    k = max(4, P // 10)
                    pool, pfit = self.genomes[order[:k]], fit[order[:k]]
                got = 1
                while got < q:
                    if meteor and rng.random() < 0.30:
                        children.append((rng.standard_normal(G) * 0.6).astype(np.float32))
                    else:
                        c1 = rng.integers(0, len(pool), 3)
                        p1 = pool[c1[int(np.argmax(pfit[c1]))]]
                        c2 = rng.integers(0, len(pool), 3)
                        p2 = pool[c2[int(np.argmax(pfit[c2]))]]
                        child = np.where(rng.random(G) < 0.5, p1, p2)
                        msk = rng.random(G) < rate
                        child = child + msk * rng.standard_normal(G).astype(np.float32) * strength
                        rs = msk & (rng.random(G) < 0.06)
                        if rs.any():
                            child[rs] = (rng.standard_normal(int(rs.sum())) * 0.6).astype(np.float32)
                        children.append(child.astype(np.float32))
                    got += 1
            keep = [s for s, q in plan if q > 0]
            self.species_reps = [reps[s] for s in keep]
            self.species_stall = [self.species_stall[s] for s in keep]
            self.species_best = [self.species_best[s] for s in keep]
        else:
            children = [self.genomes[i].copy() for i in order[:max(2, P // 10)]]
            while len(children) < P:
                if meteor and rng.random() < 0.30:
                    children.append((rng.standard_normal(G) * 0.6).astype(np.float32))
                    continue
                c1 = rng.integers(0, P, 3)
                c2 = rng.integers(0, P, 3)
                p1 = self.genomes[c1[np.argmax(fit[c1])]]
                p2 = self.genomes[c2[np.argmax(fit[c2])]]
                child = np.where(rng.random(G) < 0.5, p1, p2)
                msk = rng.random(G) < rate
                child = child + msk * rng.standard_normal(G).astype(np.float32) * strength
                rs = msk & (rng.random(G) < 0.06)
                if rs.any():
                    child[rs] = (rng.standard_normal(int(rs.sum())) * 0.6).astype(np.float32)
                children.append(child.astype(np.float32))

        self.genomes = np.stack(children[:P])
        self._new_gen_arrays()
        self.generation += 1
        self.gen_steps = 0
        if meteor:
            self.stall = 0

    def promote(self):
        champ = self.best_ever_genome
        g0 = self.generation
        self.difficulty = min(0.85, self.difficulty + 0.15)
        if self._track_arg is None:
            self.track = nr.Track(self._exam_base, difficulty=self.difficulty)
        for i in range(len(self.extra)):
            self.extra[i] = nr.Track((self._exam_base + 0x9E3779B9 * (i + 1)) % (10 ** 9),
                                     difficulty=self.difficulty)
        self._apply_fit()
        self._prepare_track()
        self.best_ever_genome = None
        self.best_ever_progress = 0.0
        self.best_ever_fit = 0.0
        self.last_best_progress = 0
        self.best_lap = None
        self.restart_evolution()
        if champ is not None:
            self.load_genome(champ, 0.0)
            self.generation = g0 + 1
