# afscripts

`afscripts` is the core simulation and calculation engine powering **AtomField**. It is a library of modular computational drivers designed to automate quantum chemistry, molecular dynamics, and machine learning potential calculations.

---

## Overview

Traditional materials modeling and reaction discovery often rely on fragmented, manual scripts that are difficult to reproduce, scale, or chain together. 

`afscripts` solves this by providing a unified driver architecture. Each calculation task—from initial geometry optimization to reactive dynamics and transition state searches—is encapsulated into an independent, headless driver with a standardized lifecycle. These drivers can be executed standalone or composed into automated pipelines.

---

## Core Capabilities

- **Geometry Optimization**: Fast structure relaxation using semi-empirical quantum methods and force fields.
- **Reactive Molecular Dynamics**: Automated trajectory generation to sample chemical rearrangements and reactive collisions.
- **Reaction Network Mapping**: Automated perception of chemical bond changes across simulation trajectories to construct reaction graphs.
- **Machine Learning Force Fields (MLFF)**: High-accuracy ground-state energy evaluation using equivariant neural network potentials.
- **Transition State & Barrier Estimation**: First-order saddle-point searches to estimate kinetic barriers and reaction pathways.

---

## Engineering Highlights

- **Standardized Driver Lifecycle**: Every driver follows a strict initialization, calculation, and export pattern for predictable execution and error reporting.
- **Zero-Copy Pipeline Linking**: Designed for DAG orchestration; downstream tasks automatically read outputs from parent steps without data duplication.
- **Headless & Scalable**: Pure CLI and programmatic interfaces with structured logging, designed to run locally or inside background worker queues (e.g., Redis/ARQ).
- **Clean Cheminformatics Integration**: Converts 3D atomistic trajectories into topological representations for automated chemical state detection.

---

## Tech Stack

- **Core**: Python 3.11+
- **Chemistry & Simulation**: ASE (Atomic Simulation Environment), xTB, RDKit, NetworkX, Sella, ML potentials (SevenNet, MACE)
- **Infrastructure**: Headless execution, isolated filesystem workspaces, structured logging

---

## Proprietary Notice

This software is proprietary and confidential to **AtomField Labs**. Unauthorized copying, distribution, or commercial modification of this codebase is strictly prohibited.
