# InterfaceAdvection.jl

[![Build Status](https://github.com/TzuYaoHuang/InterfaceAdvection.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TzuYaoHuang/InterfaceAdvection.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Overview

**InterfaceAdvection.jl** is a high-fidelity multiphase flow solver built on top of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl). It is designed to simulate unsteady, incompressible, and immiscible two-phase flows on uniform Cartesian grids with a focus on mass and momentum conservation.

## Key Features

- **Whole-field Formulation:** The two fluid phases are treated as a unified meta-fluid with spatially varying density and viscosity, determined by a color function that identifies the fluid regions.
- **Conservative Volume-of-Fluid (VOF) Method:** Mass is preserved through a robust advection of the indicator function, based on the [conservative VOF method](https://www.researchgate.net/publication/220206130_Conservative_Volume-of-Fluid_method_for_free-surface_simulations_on_Cartesian-grids).
- **Consistent Mass-Momentum (CMOM) Transport:** The Navier-Stokes equations are solved in conservative form using mass transport information, improving accuracy and stability, enhancing momentum conservation and, in special cases, preserving mechanical energy.
- **Interface-aware Flux Limiter:** This custom flux limiter addresses oscillations in the CMOM formulation.
- **Consistent Time Integration:** A custom density-weighted second-order Runge-Kutta scheme is used for temporal integration, improving consistency and robustness.
- **Flexible Backend:** Runs efficiently on serial CPU, multi-threaded CPU, or GPU platforms.

## Development Goals

- [ ] Implement an interface-aware geometric multigrid solver to accelerate pressure convergence and alleviate biased volume loss.
- [ ] Reintroduce the boundary data immersion method, potentially with a Heun's predictor-corrector time integration scheme, but still consistent.
- [ ] Fix problem on symmetry boundary condition on the ceiling under gravitational effect.
