# NeuroRacer

Neural networks learn to race via neuro-evolution on procedurally generated —
or hand-drawn — tracks.

## Install & run

    pip install -r requirements.txt     # numpy optional but recommended
    ./run.sh                            # GUI (auto-uses OpenGL when available)
    NEURORACER_GL=0 ./run.sh            # force the raster (QPainter) renderer

## Files

| file            | purpose                                                          |
|-----------------|------------------------------------------------------------------|
| `neuroracer.py` | GUI app: sim, evolution, editor, replays, renderer, manual drive |
| `vecsim.py`     | numpy-vectorised core (5-20x faster; needs numpy)                |
| `glview.py`     | OpenGL renderer (auto-selected; needs numpy)                     |
| `train.py`      | headless trainer: islands, multi-track, custom tracks            |

## GUI keys

Space pause · N new track · **T track editor** (draw loop, Enter to build,
Backspace undo, `[` `]` road width) · **G record generation** · **O open
replay** (Space play, ←/→ step, Esc live) · R restart (applies multi/spec/
racing-line settings) · E end generation · F follow cam · M drive (WASD/
arrows) · +/− speed · wheel zoom · drag pan · double-click reset

## Training

    python train.py --gens 300 --pop 96 --racing-line --eval 5
    python train.py --multi 3 --eval-mode min --gens 400   # breed generalists
    python train.py --islands 4 --gens 400                 # one island per core
    python train.py --track-json my_track.json --gens 300  # your drawn circuit
    python train.py --gens 200 --resume                    # continue champion

Outputs `champion.json` (auto-loaded by the GUI on start) and `history.csv`.

## Architecture

- **Brain**: tanh MLP 13→10→2 + 4 recurrent context units (Elman-style);
  old feedforward brains auto-migrate (`migrate_genome`).
- **Learning**: GA with elitism, tournament selection, uniform crossover,
  gaussian mutation, meteor re-seeding on stagnation, speciation via fitness
  sharing, optional curriculum (narrower/twistier tiers at 2 clean laps).
- **Multi-track fitness**: every genome is scored on N fixed "exam" circuits
  per generation; selection uses mean (or min) laps — breeds generalists.
- **Track**: radii → Catmull-Rom → arc-length resample → curvature-validated;
  physics/sensors on a rasterised occupancy grid + chamfer distance field
  (sphere-traced lidar). Editor tracks pass through the identical pipeline.
- **Replays** (`.nrr`): genomes + track spec + per-frame telemetry + network
  I/O — scrubbing is exact; the brain view and rays rebuild from the file.
