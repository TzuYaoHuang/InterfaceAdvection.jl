# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

InterfaceAdvection.jl is a two-phase (immiscible, incompressible) flow solver on uniform Cartesian grids,
built as an extension layer on top of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl).
Method paper: https://arxiv.org/abs/2606.02467 (Huang & Weymouth). Main ingredients:

- **Whole-field "meta-fluid"**: one velocity/pressure field; density and viscosity vary with the volume fraction `f`.
- **Conservative geometric VOF** (operator-split, Weymouth & Yue 2010) with PLIC reconstruction.
- **CMOM**: momentum `ρu` is advected with exactly the mass flux produced by the VOF sweep.
- **SynDRoM** momentum-flux limiter (`ϕq` in `src/flow.jl`).
- **Density-weighted explicit-midpoint RK2** time step, bounded viscosity interpolation (`getμ`),
  height-function surface tension, CPU / multithreaded / GPU via WaterLily's `@loop`.

Unregistered package, ~1.4k lines in `src/`.

## WaterLily.jl dependency — read this first

This package is not standalone: it reuses WaterLily's `Flow`/`Simulation` structs, `@loop` kernel macro,
boundary conditions and Poisson solvers, and replaces only the momentum step.

- **Local dev** uses the sibling checkout `../WaterLily.jl` through an *uncommitted* `[sources]` entry in
  `Project.toml`. When you need to know how a WaterLily function behaves, read that checkout, not your memory
  of WaterLily: its internals change between versions.
- **CI** has no `[sources]` entry and resolves WaterLily from the General registry (`compat WaterLily = "1.9.1"`,
  which also forces `julia = "1.11"`). Don't rely on unreleased WaterLily features: define them locally, or
  raise the compat bound once WaterLily registers a release containing them. As of 1.9.1, `face` boundary
  loops, the GPU `perdot` fix and 2D `ω(I,u)` are still master-only.
- Many imports are **non-exported WaterLily internals**, so a WaterLily refactor can break this package. When
  WaterLily is bumped, check these first:
  - `src/InterfaceAdvection.jl`: `@loop, div, inside, ∂, inside_u, CI, CIj, slice, size_u, NoBody, check_fn, sim_step!, sim_info, load!`
  - `src/flow.jl`: `accelerate!, median, update!, scale_u!, exitBC!, perBC!, residual!, mult, flux_out, vanLeer, L∞, ϕ, udf!`
  - `src/VOFutil.jl`: `kern₀`; `src/metrics.jl`: `fSV, fsum, shiftDir`
  - also used via exports: `δ, loc, BC!, @inside, Flow, Poisson, MultiLevelPoisson, solver!, L₂, AbstractSimulation, Simulation`

Where things live in `../WaterLily.jl`:
- `src/core.jl`: `@loop`, `@inside`, `inside`, `inside_u`, `slice`/`face`, `δ`, `CI`, `CIj`, `loc`, `BC!`, `perBC!`, `exitBC!`, backend preference
- `src/Flow.jl`: `Flow` struct (`u, u⁰, f, p, σ, V, μ₀, μ₁, uBC, Δt, ν, g, exitBC, perdir, λ`), `mom_step!` (the single-phase step this package replaces), `CFL`, `flux_out`, `udf!`, `accelerate!`
- `src/Poisson.jl`, `src/MultiLevelPoisson.jl`: `Poisson`/`MultiLevelPoisson`, `solver!`, `update!`, `residual!`
- `src/WaterLily.jl`: `Simulation`, generic `sim_step!(sim, t_end)` loop, `sim_time`, `measure!`, `load!`/`vtkWriter` hooks

### How the wrapping works
- `TwoPhaseSimulation <: AbstractSimulation` holds `sim::WaterLily.Simulation` and `intf::cVOF`.
  `getproperty`/`setproperty!` fall through to `sim` for anything other than `sim`/`intf`, so `sim.flow`,
  `sim.pois`, `sim.L`, `sim.U` and `sim.body` all work.
- The constructor forwards all WaterLily `Simulation` kwargs (`Δt, ν, g, U, u0, perdir, exitBC, …`), passes
  `λ=Koren` (this package's limiter, stored on `flow.λ`), builds `cVOF`, then caps the initial `Δt` by `MPCFL`.
- `sim_step!(::TwoPhaseSimulation; …)` is overloaded to call `MPFMomStep!(flow, pois, intf, body)` instead of
  `mom_step!`. WaterLily's generic `sim_step!(sim, t_end; max_steps, verbose, udf, kwargs...)` drives it.

### `@loop`, backends, kernel rules
- All field loops use `@loop expr over I ∈ R`. Depending on the backend it compiles to a `@simd` loop
  (`"SIMD"`) or a KernelAbstractions kernel (`"KernelAbstractions"`, multithreaded CPU or GPU). The backend is a
  compile-time Preference in the gitignored `LocalPreferences.toml` (currently `KernelAbstractions`). Changing
  it requires a fresh Julia session:
  ```toml
  [WaterLily]
  backend = "KernelAbstractions"   # or "SIMD"
  ```
- Anything called inside `@loop` must be GPU-safe: no allocation, type-stable, no dynamic dispatch, `ifelse`
  rather than type-changing branches, no `argmax` (hence `myArgAbsMax`). Scalar indexing of device arrays
  outside kernels must go through `_scalar_op` (the CUDA/AMDGPU extensions wrap it in `@allowscalar`). Don't
  put `@fastmath` on CFL code because `maximum` misbehaves on GPU.
- Kernel pattern: a host method runs `@loop foo!(args..., I) over I ∈ R`, and an `@inline` per-cell method
  `foo!(args..., I)` does the work (see `applyVOF!`, `reconstructInterface!`, `ρu2u!`, `getVOFFlux!`).
- Direction-split flux loops handle the first and last face with `lowerBoundary…!`/`upperBoundary…!` building
  blocks dispatched on `Val{j ∈ perdir}` (Neumann vs periodic), mirroring WaterLily's `conv_diff!`. This package
  uses `slice(N,i,j,low)` (CartesianIndices) where current WaterLily uses `face(...)`.

## Commands

Run from the repo root. Locally, Julia 1.13 via juliaup; CI tests 1.11 and 1.12.

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'

# Main suite: needs backend "KernelAbstractions" and >1 thread (~20 s on CPU)
julia --project -t auto -e 'include("test/runtests.jl")'

# Allocation suite: needs backend "SIMD" (edit LocalPreferences.toml) and 1 thread; Pkg.test supplies BenchmarkTools
JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()'
```

- `test/runtests.jl` picks the suite by thread count: more than one thread runs `test/maintests.jl`, otherwise
  `test/alloctest.jl` (`MPFMomStep!` on a 32³ droplet must allocate under 50 kB). Each file throws if the
  backend is wrong.
- GPU arrays (`CuArray`/`ROCArray`) are added to the test matrix automatically when `nvcc`/`hipcc` is on PATH and functional.
- `test/helper.jl` provides reusable setups: `TGVDropletSim` (periodic Taylor–Green droplet, checks volume
  conservation) and `quiescentDropletSim` (checks the step is a no-op).
- New behaviour gets a testset in `test/maintests.jl`, which groups testsets by source file.
- There is no linter or formatter.

Minimal usage (verified to run):
```julia
using InterfaceAdvection, WaterLily
N = 64
g(i,x,t) = i==2 ? -1f0 : 0f0                # WaterLily checks it is type-stable and returns T
sim = TwoPhaseSimulation((N,N), (0,0), N;   # dims, uBC, length scale L
        U=1, ν=1e-3, g,                     # WaterLily kwargs; U is required because uBC==0 gives U=0
        λρ=1e-3, λμ=1e-2, η=1e-3,           # light/dark density & viscosity ratios, surface tension
        InterfaceSDF = x -> x[2]-N/2,       # dark fluid where SDF<0, i.e. below y=N/2
        T=Float32, mem=Array)               # mem=CuArray / ROCArray for GPU
sim_step!(sim, 1.0; verbose=true)           # advance to tU/L = 1
sim.intf.f                                  # volume fraction, including ghost cells
```

## Source map (`src/`)

| File | Contents |
|---|---|
| `InterfaceAdvection.jl` | module, imports, exports, `TwoPhaseSimulation`, `sim_step!`/`sim_info` overloads, extension loading |
| `cVOF.jl` | `cVOF` struct: interface fields, CMOM buffers, physical parameters |
| `flow.jl` | **core step `MPFMomStep!`**; momentum limiters (`upwind, minmod, Koren, vanAlbada1, Sweby, superbee, TVDcen, TVDdown`); SynDRoM flux `ϕq`/`ϕu*`; coupled advection `advectfq!` → `advectVOFρuu!` → `advectρuu1D!`; viscous stress `visc!`; `updateU!`; `updateL!` (Poisson coefficients 1/ρ); `MPCFL`; projection `myproject!`/`inproject!`/`psolver!` |
| `advection.jl` | VOF-only advection (`advect!`, `advectVOF!`, `advectVOF1d!`); geometric donor-acceptor flux `getVOFFlux!` (also accumulates the mass flux `ρuf`); `reportFillError` |
| `PLIC.jl` | Scardovelli & Zaleski (2000) volume↔intercept: `getIntercept`, `getVolumeFraction`, `f2α`, `α2f`, cubic root `proot` |
| `normalEstimation.jl` | `reconstructInterface!`; normal schemes `getInterfaceNormal_{WH (default), WY, Column, MYC, Y, CD, PCD, SLIC, XYLIC}!` (only WH, WY, MYC, Y, PCD are exported) |
| `VOFutil.jl` | SDF initialisation `applyVOF!`; BCs `BCVOF!`, `BCf!`, `BCv!`, `BCv1D!`; `cleanWisp!`; properties `getρ`, `getμ`, `linInterpProp`; `ρu2u!`/`u2ρu!`; `f2face!`; `fᶠ2ρuf` |
| `surfaceTension.jl` | `surfTen!` (per momentum direction, on face-interpolated f); height-function curvature `getCurvature`/`getPopinetHeight` (Popinet 2009) |
| `redistaning.jl` | `LevelSet` + `redistaning!`: level-set reinitialisation (SSP-RK3, ENO2) starting from ϕ = 2f−1 |
| `metrics.jl` | multiphase diagnostics `ρkeI`, `ρgh`, `EnsI`, `ρuI` |
| `util.jl` | `myArgAbsMax`, `δd` (signed δ), `getXdir`/`getXYdir`, `majorDir`, `inside_uWB`, `validCI` |

`ext/`: `IntfAdvCUDAExt` and `IntfAdvAMDGPUExt` only add `_scalar_op` for device arrays. `IntfAdvReadVTKExt` adds
`load!(::TwoPhaseSimulation, Val(:pvd); fname)` for restarts. It reads point data named `"u"`, `"p"` and `"f"`,
so files must be written with those attribute names rather than WaterLily's defaults (`"Velocity"`/`"Pressure"`).

## Conventions and data layout

- **Grid**: every array has one ghost layer. Scalars are `N .+ 2`, vectors `(N.+2..., D)`; use `inside(a)` for
  the interior. Velocity is staggered (MAC): `u[I,i]` sits on the lower `i`-face of cell `I`, and the centre of
  cell `I` is `loc(0,I) = I .- 1.5`.
- **Phases**: `f ∈ [0,1]` is the volume fraction of the **dark** (reference, usually heavy) fluid: `f=1` is
  dark, `f=0` is light. `InterfaceSDF(x) < 0` inside the dark fluid, and PLIC normals `n̂` point into the light
  fluid. Without an `InterfaceSDF` the whole domain is dark fluid (the default stated in the
  `TwoPhaseSimulation` docstring is stale).
- **Properties** (nondimensional): the dark fluid has ρ=1 and dynamic viscosity μ=`ν` (WaterLily's `ν` kwarg).
  The light fluid has ρ=`λρ` and μ=`λμ·ν`. Defaults are `λρ=1e-3` and `λμ=1e-2` (air/water). The pressure
  Poisson equation uses coefficients 1/ρ.
- `μ==0` and `η==0` are stored as `nothing`, which switches off the viscous or surface-tension term through
  dispatch on `::Nothing`.
- Momentum limiters `λ(u,c,d)` (upstream, centre, downstream) are redefined in `flow.jl`. The comment there says
  WaterLily's versions break TVD-conservation. The default is `Koren`.
- Unicode/Greek names are standard and meaningful (`λρ, λμ, η, ν, ρu, ρuf, n̂, α, fᶠ, c̄, δt, Δt`); keep them.
  Non-trivial functions carry docstrings; keep that too.
- Leave these names as they are even though they look wrong: `redistaning` (sic; exported and used
  downstream), and `minmod`, which has both a 3-arg limiter (`flow.jl`) and a 2-arg helper (`redistaning.jl`).

## One time step: `MPFMomStep!` (`src/flow.jl`)

It is a density-weighted explicit midpoint RK2 starting from the state fⁿ, uⁿ:

1. **Predictor** (weight ½): copy uⁿ→`u⁰` and fⁿ→`f⁰`, set `ρu = ρ(fⁿ)uⁿ`, and `advectfq!` advects `f⁰` and
   `ρu` over the full δt with uⁿ. Then set `f⁰ ← (fⁿ+f*)/2` and add the viscous and surface-tension momentum
   forcing at `f⁰`. `updateU!` forms `ρu^{n+½} = (ρuⁿ + ρu* + Fδt)/2`, divides by ρ(f⁰) and adds gravity/`udf`
   velocity forcing ×δt/2. `updateL!` sets `flow.μ₀ = 1/ρ(f⁰)`, and the step projects with Δt/2.
2. **Corrector** (weight 1): reset `f⁰ ← fⁿ` and `ρu = ρ(fⁿ)uⁿ`, then advect `intf.f` and `ρu` with u^{n+½}. Add
   forcing at f^{n+1}, compute `u = ρu/ρ(f^{n+1})`, apply gravity/`udf` at t^{n+1}, and project with Δt using
   1/ρ(f^{n+1}).
3. `push!(Δt, min(MPCFL, 1.2δt))`. `MPCFL` takes the minimum of the advection, cVOF, gravity, viscous and
   surface-tension limits (Kang, Fedkiw & Liu 2000) with a 0.8 safety factor. Δt may grow by at most 20 % per step.

Inside `advectVOFρuu!` the splitting is first-order Lie–Trotter, and the direction order rotates every step via
`length(Δt)` (Strang/Auzinger–Ketcheson variants are commented out). For each direction d:
- `reconstructInterface!` (PLIC) → `getVOFFlux!` geometric flux plus the dilation term `c̄·∂u/∂x`, where
  `c̄ = f ≥ 0.5` (Weymouth & Yue) → `cleanWisp!` → BCs.
- That sweep's mass flux (`ρuf = λρ·δl + (1-λρ)·fᶠ`, divided by δt) advects every momentum component in direction d
  (`advectρuu1D!`), with SynDRoM limiting (`ϕq`) and the matching dilation correction. This shared flux *is* the
  CMOM property, so never compute the momentum mass flux any other way.

Pressure: the default `MultiLevelPoisson` uses WaterLily's `solver!(tol=1e-4, itmx=200)`. A plain `Poisson`
uses the local PCG `psolver!`.

`udf` is called as `udf(flow, u, t; fc, kwargs...)`, or as `udf(flow, t; …)` if that 3-arg method doesn't
exist. It must write an **acceleration** (velocity forcing) into `flow.f`. `fc` is the volume fraction at that
stage, so a udf must accept the `fc` keyword (or `kwargs...`).

## Buffer aliasing — read before touching `flow.jl`

To keep the step allocation-free, fields are reused for unrelated data within a step. After `MPFMomStep!`
these do **not** hold their nominal meaning:

| Field | Nominal | Also used as |
|---|---|---|
| `intf.n̂` | PLIC normal | face-interpolated f (`visc!`), old momentum ρuⁿ (`updateU!`), velocity copy `uStar` in momentum advection, normals in `surfTen!` |
| `intf.α` | PLIC intercept | `ρ̄∂ⱼuⱼ` dilation buffer in `advectρuu1D!` |
| `intf.fᶠ` | VOF face flux | face-interpolated f in `surfTen!` |
| `intf.f⁰` | old f | midpoint f during the predictor |
| `intf.dρ` | (buffer) | face-interpolated old f for SynDRoM |
| `flow.f` | force | momentum RHS `r`, velocity scratch, velocity forcing |
| `flow.σ` | divergence | flux scratch `Φ`, copy of old f, CFL scratch |
| `flow.μ₀` | BDIM moment | Poisson coefficient 1/ρ, overwritten every step |

`LevelSet(sim)` aliases `intf.f⁰`, `intf.α`, `intf.fᶠ` and `flow.σ` as `ϕ, ϕ⁰, ϕini, L`, so build it only
between steps. Before using `n̂`/`α` for post-processing, rerun `reconstructInterface!(f, α, n̂, normalScheme)`.

## Known limitations and gotchas

- **Immersed bodies (BDIM) are not supported.** `MPFMomStep!` overwrites `flow.μ₀` and ignores `flow.V`/`μ₁`, so
  a `body` kwarg is silently ignored. Reintroducing it is a README development goal.
- Known open issue (README): the symmetry BC on the ceiling misbehaves under gravity.
- `reportFillError` only prints when f leaves [0,1] by more than tol. It throws on NaN, or when |∇·u| > 10.
- The `cVOF` constructor `println`s `μ` and `λρ` every time it runs. That output is expected.
- If `uBC` is all zeros you must pass `U`. Otherwise WaterLily sets `U=0`, so `sim_time` never advances and
  `sim_step!(sim, t_end)` never returns.

## Downstream consumer: `../PhDTest`

The sibling research sandbox `../PhDTest` has its own CLAUDE.md, and its case builders are in `allCases.jl`. It
uses this package through a path source and imports many internals: `getρ`, `u2ρu!`, `ρu2u!`, `updateU!`,
`updateL!`, `myproject!`, `inproject!`, `psolver!`, `MPCFL`, `BCf!`, `BCv!`, `reconstructInterface!`,
`getInterfaceNormal_*`, `advectVOF1d!`, `advectρuu1D!`, `getCurvature`, and others. Its driver scripts also
redefine `InterfaceAdvection.inproject!` to change Poisson tolerances. Renaming internals or changing their
signatures can break it, so call that out when you do. Example cases (dam break, rising bubble, TGV, capillary
wave, …) live there. `examples/` in this repo holds only gitignored local output.

## CI

`.github/workflows/CI.yml` runs Julia 1.11 and 1.12 on ubuntu, macOS and windows. The 1-thread jobs write
backend `SIMD` and run the alloc tests; the `auto`-thread jobs write `KernelAbstractions` and run the main
suite. CI triggers on PRs and on pushes to `master`, but the default branch is `main`, so direct pushes to
`main` don't run CI. Open a PR instead.

## Working preferences

- Prefer concise code and minimal diffs. Don't add refactors, abstractions or scaffolding nobody asked for.
- Anything inside the time-step loop must stay allocation-free (alloc test budget: 50 kB per step).
