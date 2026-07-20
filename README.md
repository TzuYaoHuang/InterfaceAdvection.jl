# InterfaceAdvection.jl

[![Build Status](https://github.com/TzuYaoHuang/InterfaceAdvection.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TzuYaoHuang/InterfaceAdvection.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Overview

**InterfaceAdvection.jl** is a high-fidelity multiphase flow solver built on top of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl). It simulates unsteady, incompressible, immiscible two-phase flows on uniform Cartesian grids, with a focus on mass and momentum conservation. Implementation details are available in [this paper](https://arxiv.org/abs/2606.02467).

## Key Features

- **Whole-field formulation:** the two fluid phases are treated as a unified meta-fluid with spatially varying density and viscosity, determined by a color function that identifies the fluid regions.
- **Conservative Volume-of-Fluid (VOF) method:** mass is preserved through robust advection of the indicator function, based on the [conservative VOF method](https://www.researchgate.net/publication/220206130_Conservative_Volume-of-Fluid_method_for_free-surface_simulations_on_Cartesian-grids).
- **Consistent Mass-Momentum (CMOM) transport:** the Navier-Stokes equations are solved in momentum form with identical mass flux in the mass and momentum advection steps, improving accuracy and stability, conserving momentum, and, in special cases, bounding mechanical energy.
- **Synchronized Donor Region of Momentum flux (SynDRoM):** a custom flux limiter that suppresses velocity oscillations arising from the CMOM formulation.
- **Consistent time integration:** a custom density-weighted second-order Runge-Kutta scheme improves consistency and robustness.
- **Bounded viscosity interpolation:** unlike traditional arithmetic or harmonic averaging, viscosity is interpolated in a way that guarantees bounded values.
- **Flexible backend:** runs efficiently on serial CPU, multi-threaded CPU, or GPU.

## Installation

InterfaceAdvection.jl is not yet registered. Install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/TzuYaoHuang/InterfaceAdvection.jl")
```

## Development Goals

- [ ] Reintroduce the boundary data immersion method, potentially with a Heun's predictor-corrector time integration scheme, but still consistent.
- [ ] Fix problem on symmetry boundary condition on the ceiling under gravitational effect.

## License

InterfaceAdvection.jl is licensed under the [MIT License](LICENSE).
