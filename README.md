# InterfaceAdvection.jl

[![Build Status](https://github.com/TzuYaoHuang/InterfaceAdvection.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TzuYaoHuang/InterfaceAdvection.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Overview

**InterfaceAdvection.jl** is a high-fidelity multiphase flow solver built on top of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl). It is designed to simulate unsteady, incompressible, and immiscible two-phase flows on uniform Cartesian grids with a focus on mass and momentum conservation. Implementation details can be found in [this SNH paper](https://arxiv.org/abs/2606.02467).

## Key Features

- **Whole-field Formulation:** The two fluid phases are treated as a unified meta-fluid with spatially varying density and viscosity, determined by a color function that identifies the fluid regions.
- **Conservative Volume-of-Fluid (VOF) Method:** Mass is preserved through a robust advection of the indicator function, based on the [conservative VOF method](https://www.researchgate.net/publication/220206130_Conservative_Volume-of-Fluid_method_for_free-surface_simulations_on_Cartesian-grids).
- **Consistent Mass-Momentum (CMOM) Transport:** The Navier-Stokes equations are solved in momentum formulation with identical mass flux in mass and momentum advection steps, improving accuracy and stability, conserving momentum and, in special cases, bounding mechanical energy.
- **Synchronized Donor Region of Momentum flux (SynDRoM):** This custom flux limiter addresses velocity oscillations in the CMOM formulation.
- **Consistent Time Integration:** A custom density-weighted second-order Runge-Kutta scheme is used for temporal integration, improving consistency and robustness.
- **Bounded viscosity interpolation:** Viscosity interpolation is now bounded. Traditional approach such as arithmetic and harmonic average cannot achieve this.
- **Flexible Backend:** Runs efficiently on serial CPU, multi-threaded CPU, or GPU platforms.

## Development Goals

- [ ] Implement an interface-aware geometric multigrid solver to accelerate pressure convergence and alleviate biased volume loss.
- [ ] Reintroduce the boundary data immersion method, potentially with a Heun's predictor-corrector time integration scheme, but still consistent.
- [ ] Fix problem on symmetry boundary condition on the ceiling under gravitational effect.
