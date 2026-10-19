# A deconstructed comparison of DWM and LES wake aerodynamics

This project is part of the Wake Aerodynamics special course at the Technical University of Denmark (DTU).

**Authors:** Ignacio Hector Caviglia and Joey Ray Bink

## Research question

Is the "passive tracer" assumption valid for wake modeling? In other words, can the three components of the Dynamic Wake Meandering (DWM) model, **wind speed deficit**, **wake meandering**, and **added turbulence**, be modeled separately and then superposed to reproduce the wake behavior captured by Large Eddy Simulation (LES)?

## Study setup

The study considers a single IEA 22 MW wake-generating wind turbine and a downstream "ghost" IEA 22 MW turbine aligned with the wake-generating WTG at different downstream distances (5D and 10D).

## Methodology

1. Extract the wind speed deficit, wake meandering, and added-turbulence components from the LES simulation results.
2. Construct a flow box by superposing the three components.
3. Run a HAWC2 simulation using the reconstructed flow box.
4. Compare it with a HAWC2 simulation using directly the LES waked flow.

## Input data

The input data comes from the same LES simulation used in [Steiner et al., 2026](https://wes.copernicus.org/articles/11/1679/2026/wes-11-1679-2026.pdf).

- **Full dataset:** stored on the DTU HPC (GBAR) under `/work3/emlh/IgnacioJoey/`. It is not included in this repository.
- **Test subset:** [`data/test_les_data/`](data/test_les_data/) contains sample planes of the LES flow field in NetCDF format: y-z planes at 2D upstream (`Flowpos_n2D.nc`), 5D (`Flowpos_05D.nc`), and 10D downstream (`Flowpos_10D.nc`), and the hub-height x-y plane (`Flowpos_hubheightplane.nc`).

## Recommended reading

The [reading guide](documents/readings/READING_GUIDE.md) lists the recommended literature for the project, with notes on what to take from each reference. The PDFs are in [`documents/readings/`](documents/readings/).

## Repository structure

| Directory | Contents |
| --- | --- |
| `data/` | LES test data |
| `documents/` | Project documents, including the recommended readings and the reading guide |
| `illustrations/` | Illustrations of the methodology |
| `notebooks/` | Analysis notebooks, including the LES flow analysis (`les_flow_analysis.ipynb`) |
| `plots/` | Plots |
| `presentations/` | Presentation materials |
| `scripts/` | Analysis scripts |
