# A deconstructed comparison of DWM and LES wake aerodynamics

This project is part of the Wake Aerodynamics special course at the Technical University of Denmark (DTU).

**Authors:** Ignacio Hector Caviglia and Joey Ray Bink

## Research question

Can the three components of the Dynamic Wake Meandering (DWM) model, **wind speed deficit**, **wake meandering**, and **added turbulence**, be modeled separately and then superposed to reproduce the wake behavior captured by Large Eddy Simulation (LES)?

## Study setup

The study considers a single IEA 22 MW wake-generating wind turbine and a downstream "ghost" IEA 22 MW turbine aligned with the wake-generating WTG at different downstream distances.

## Methodology

1. Extract the wind speed deficit, wake meandering, and added-turbulence components from the LES simulation results.
2. Construct a flow box by superposing the three components.
3. Run a HAWC2 simulation using the reconstructed flow box.
4. Compare it with a HAWC2 simulation using directly the LES waked flow.

## Input data

The input data comes from the same LES simulation used in [the study published in Wind Energy Science, volume 11, page 1679 (2026)](https://wes.copernicus.org/articles/11/1679/2026/wes-11-1679-2026.pdf).

## Repository structure

| Directory | Contents |
| --- | --- |
| `data/` | Study data |
| `documents/` | Project documents |
| `illustrations/` | Illustrations |
| `notebooks/` | Analysis notebooks |
| `plots/` | Plots |
| `presentations/` | Presentation materials |
| `scripts/` | Analysis scripts |