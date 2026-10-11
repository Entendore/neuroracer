#!/usr/bin/env bash
# Launch the NeuroRacer GUI.  NEURORACER_GL=0 ./run.sh forces the 2D renderer.
cd "$(dirname "$0")"
exec python neuroracer.py "$@"
