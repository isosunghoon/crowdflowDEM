# strategy-based-crowd-flow-dynamics
![Static Badge](https://img.shields.io/badge/status-done-brightgreen?style=for-the-badge)
![Static Badge](https://img.shields.io/badge/type-research-purple?style=for-the-badge)

This repository contains the code behind our paper on strategy‑based crowd flow dynamics. The paper received the Minister’s Award at the 70th National Science Exhibition (Research; largest student contest in Korea) — 제70회 전국과학전람회 산업 및 에너지 부문 특상 (4위).

## Abstract

We present an agent-based crowd simulator that blends decision-level strategy with physical interaction modeling. Each pedestrian observes a limited field of view, reasons about nearby people and obstacles, and rotates a target heading using a perception-dependent “will” term. This intent is coupled with spring–damper style contact forces for human–human and human–obstacle interactions, enabling smooth avoidance as well as realistic collision responses. The model is calibrated against real-world trajectories: recorded tracks are projected into the ground plane, simulated paths are generated with matched spawn timing, and hyperparameters are optimized via dynamic time warping to minimize trajectory distance. The codebase reproduces the evaluation videos (crossroads, narrow passages, maze, obstacle courses, and real-world scenes) and provides scripts for running new simulations or parameter fitting on additional datasets.

## Pipeline Overview

The project optimizes a crowd simulation against real-world trajectories through:

1. **Crowd localization** — Detect and localize people in video (density maps, centroids).
2. **Inverse perspective mapping** — Calibrate camera and project image coordinates to the ground plane (bird's-eye view).
3. **Per-person separation** — Associate detections across frames into individual trajectories (tracking).
4. **Simulation connect** — Align real tracks with simulated ones (spawn timing, DTW comparison).
5. **Simulation optimization** — Tune simulator hyperparameters via Bayesian optimization (e.g. `skopt`) to minimize trajectory distance (e.g. DTW).

## Project Structure

Only the main code and notebooks are listed; support modules (e.g. `datasets/`, `model/HR_Net/`) are omitted for brevity.

```
strategy-based-crowd-flow-dynamics/
├── README.md
├── crowd_localization/                    # 1. Crowd localization + 2–3. IPM & separation
│   ├── crowd_tracking.ipynb               # ★ Main: localization → separation → inverse mapping (full pipeline on video)
│   ├── image_processing.ipynb             # ★ Image/data prep, calibration for inverse perspective mapping
│   ├── seperate_person.ipynb              # ★ Per-person separation (tracking) algorithm
│   ├── pipeline.py                        # Localization + visualization pipeline (script)
│   ├── test.py                            # Crowd localization inference on image list
│   ├── train.py / trainer.py              # Training the localization model
│   ├── config.py                          # Config and paths
│   ├── DLT_data.txt                       # Calibration / DLT data for IPM
│   ├── model/                             # Localization model (locator, PBM, HR_Net backbone)
│   │   ├── locator.py
│   │   ├── PBM.py
│   │   └── HR_Net/
│   ├── misc/                              # Transforms, inflation, metrics, utils
│   └── requirements.txt
│
├── simulation/                            # 4. Simulation connect + 5. Bayesian optimization
│   ├── simulate_manager.py                # Run scenario simulations (orchestrator)
│   ├── simulate/                          # Scenario scripts (4way, maze, narrow, obstacle, road, twopeople, etc.)
│   │   ├── standard.json                  # Default simulation parameters
│   │   └── *.py
│   │
│   ├── optimization/                      # Connect real↔sim + Bayesian optimization
│   │   ├── connecting_simulation.ipynb    # ★ Match real tracks with simulation; DTW-based comparison
│   │   ├── optimize.py                    # ★ Bayesian optimization (skopt gp_minimize) over sim params
│   │   ├── optirun.py                     # ★ Simulation engine used inside optimization loop
│   │   ├── dtw.py                         # Dynamic time warping for trajectory distance
│   │   └── draw.py                        # Plotting / visualization
│   │
│   └── realworlddata_match/               # Run simulation with real-world data + IPM
│       ├── reverse_projection.ipynb       # ★ Inverse perspective: project tracks to ground plane (homography)
│       ├── run.py                         # Run simulation matched to real-world scenario
│       ├── standard.json
│       └── processed_tracks.csv           # Processed real-world tracks
│
└── results/                               # Output videos, tracks, and figures
```

**Legend:** ★ = primary notebooks/scripts for the pipeline steps above.