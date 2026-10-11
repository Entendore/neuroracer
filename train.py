#!/usr/bin/env python
"""Headless NeuroRacer trainer — vectorised, optional island parallelism.

    python train.py --gens 300 --pop 96 --racing-line --eval 5
    python train.py --multi 3 --eval-mode min --gens 400        # generalists
    python train.py --islands 4 --gens 400 --pop 96              # one per core
    python train.py --track-json my_track.json --gens 300        # drawn track
    python train.py --gens 200 --resume --track-seed 42
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Process, Queue
from queue import Empty

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import neuroracer as nr


def save_champion(path, genome, progress, best_lap):
    with open(path, "w") as fh:
        json.dump({"format": "neuroracer-brain-v2", "n_in": nr.N_IN,
                   "hidden": nr.HIDDEN, "n_out": nr.N_OUT, "ctx": nr.CTX,
                   "genome": [float(g) for g in genome],
                   "best_progress_points": float(progress),
                   "best_lap": best_lap}, fh)


def make_world(args, track_seed, evo_seed):
    cfg = {"population": args.pop, "mut_rate": args.mut_rate,
           "mut_strength": args.mut_strength, "meteor": args.meteor,
           "racing_line": args.racing_line, "multi": args.multi,
           "eval_mode": args.eval_mode, "curriculum": args.curriculum}
    track = None
    if args.track_json:
        with open(args.track_json) as fh:
            track = nr.Track.from_spec(json.load(fh)["track"])
    try:
        import numpy  # noqa: F401
        import vecsim
        return vecsim.VecWorld(cfg, track=track, track_seed=track_seed,
                               evo_seed=evo_seed, difficulty=args.difficulty), "numpy"
    except ImportError:
        w = nr.World(cfg, seed=track_seed, difficulty=args.difficulty)
        random.seed(evo_seed)
        if track is not None:
            w.install_track(track, reseed=False)
        return w, "python"


def _worker(wid, n_islands, args, track_seed, evo_seed, elite_q, res_q):
    world, backend = make_world(args, track_seed, evo_seed)
    world.gen_max_steps = max(1, int(args.gen_length / nr.DT))
    if args.resume and wid == 0 and os.path.exists(args.out):
        with open(args.out) as fh:
            data = json.load(fh)
        if len(data.get("genome", [])) in (nr.GENOME_SIZE, nr.OLD_GENOME_SIZE):
            world.load_genome(nr.migrate_genome(data["genome"]),
                              float(data.get("best_progress_points", 0.0)))
            print(f"[{wid}] resumed from {args.out}")
    target = world.generation + args.gens
    gt = time.perf_counter()
    while world.generation < target:
        world.step()
        if world.gen_steps == 0 and world.history:
            g, b, m = world.history[-1]
            n = world.track.n
            wall, gt = time.perf_counter() - gt, time.perf_counter()
            fit_s = f"  fit {world.fit_history[-1][1]:5.2f}" if args.multi > 1 else ""
            print(f"[{wid}/{backend}] gen {g:4d}  best {b / n:6.2f}  med {m / n:6.2f}"
                  f"  rec {world.best_ever_progress / n:6.2f}{fit_s}"
                  f"  stall {world.stall:3d}  [{wall:5.1f}s/gen]", flush=True)
            if g % 25 == 0 and wid == 0:
                save_champion(args.out, world.best_ever_genome,
                              world.best_ever_progress, world.best_lap)
            if n_islands > 1 and g % args.migrate == 0:
                elite_q.put(("elite", wid, [world.get_genome(i)
                                            for i in world.top_indices(3)]))
                got = []
                while True:
                    try:
                        msg = elite_q.get_nowait()
                    except Empty:
                        break
                    if msg[0] == "elite" and msg[1] != wid:
                        got.extend(msg[2])
                if got:
                    world.inject(got[:3])
    res_q.put((wid, world.best_ever_genome, world.best_ever_progress,
               getattr(world, "best_ever_fit", 0.0), world.history, world.best_lap))


def evaluate(genome, seconds=120.0):
    """Honest metric: champion on a FRESH track, reference (scalar) sim."""
    probe = nr.World({"population": 2, "mut_rate": 0.15, "mut_strength": 0.30})
    car = nr.Car(nr.migrate_genome(list(genome)), probe.track, hue=0.5)
    for _ in range(int(seconds / nr.DT)):
        if not car.alive:
            break
        car.step(probe)
    return car, probe


def main():
    ap = argparse.ArgumentParser(description="Headless neuro-evolution trainer")
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gen-length", type=float, default=45.0)
    ap.add_argument("--mut-rate", type=float, default=0.15)
    ap.add_argument("--mut-strength", type=float, default=0.30)
    ap.add_argument("--meteor", type=int, default=25)
    ap.add_argument("--track-seed", type=int, default=None)
    ap.add_argument("--track-json", default=None,
                    help="train on a saved custom track (editor's 'Save track')")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--islands", type=int, default=1)
    ap.add_argument("--migrate", type=int, default=10,
                    help="generations between elite exchange")
    ap.add_argument("--racing-line", action="store_true",
                    help="measure fitness on the optimised racing line")
    ap.add_argument("--multi", type=int, default=1,
                    help="evaluate every genome on N circuits per generation")
    ap.add_argument("--eval-mode", choices=("mean", "min"), default="mean")
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--eval", type=int, default=0, metavar="N")
    ap.add_argument("--out", default="champion.json")
    ap.add_argument("--history", default="history.csv")
    args = ap.parse_args()

    track_seed = args.track_seed if args.track_seed is not None \
        else random.randrange(1, 10 ** 9)
    print(f"track seed {track_seed} · islands {args.islands} · pop {args.pop} · "
          f"multi {args.multi} ({args.eval_mode}) · {args.gen_length:.0f}s/gen · "
          f"out -> {args.out}", flush=True)

    elite_q, res_q = Queue(), Queue()
    if args.islands > 1:
        procs = []
        for wid in range(args.islands):
            p = Process(target=_worker,
                        args=(wid, args.islands, args, track_seed,
                              random.randrange(1, 10 ** 9), elite_q, res_q),
                        daemon=True)
            p.start()
            procs.append(p)
        results = [res_q.get() for _ in range(args.islands)]
        for p in procs:
            p.join()
    else:
        _worker(0, 1, args, track_seed, random.randrange(1, 10 ** 9),
                elite_q, res_q)
        results = [res_q.get()]

    if args.multi > 1:
        wid, genome, progress, fit, history, best_lap = max(results, key=lambda r: r[3])
    else:
        wid, genome, progress, fit, history, best_lap = max(results, key=lambda r: r[2])
    print(f"winning island: {wid}")

    save_champion(args.out, genome, progress, best_lap)
    if args.track_json:
        with open(args.track_json) as fh:
            n = nr.Track.from_spec(json.load(fh)["track"]).n
    else:
        n = nr.Track(track_seed).n
    with open(args.history, "w") as fh:
        fh.write("generation,best_points,median_points,best_laps,median_laps\n")
        for g, b, m in history:
            fh.write(f"{g},{b:.1f},{m:.1f},{b / n:.4f},{m / n:.4f}\n")
    print(f"done · record {progress / n:.2f} laps"
          + (f" · combined fit {fit:.2f}" if args.multi > 1 else "")
          + (f" · best lap {best_lap:.2f}s" if best_lap else "")
          + f" -> {args.out}, {args.history}")

    for i in range(args.eval):
        car, probe = evaluate(genome)
        laps = car.progress / probe.track.n
        verdict = "survived 120s" if car.alive else f"crashed at {car.age:5.1f}s"
        extra = f" · best lap {probe.best_lap:.2f}s" if probe.best_lap else ""
        print(f"  eval {i + 1}: {verdict} · {laps:5.2f} laps{extra}")


if __name__ == "__main__":
    main()
