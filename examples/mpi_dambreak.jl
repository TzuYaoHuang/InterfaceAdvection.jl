"""
mpi_dambreak.jl — MPI-parallel 2D dam break (column collapse) with InterfaceAdvection.jl
==========================================================================================
A rectangular water column collapses under gravity inside a closed tank (free-slip
walls, no periodic directions). Demonstrates the MPI-parallel plumbing InterfaceAdvection.jl
reuses from WaterLily.jl's `mpi-igg` branch:

  - `@distributed TwoPhaseSimulation(...)` decomposes the domain (via
    `WaterLily.init_waterlily_mpi`) and builds the flow *and* the VOF interface field with
    matching rank-local sizes — `@distributed` works on any `dims`-first constructor, not
    just `WaterLily.Simulation`.
  - `InterfaceSDF` is evaluated at *global* coordinates (`global_offset`), so every rank
    initializes the same water column regardless of how the domain is split.
  - `MPCFL` synchronizes `Δt` across ranks (`global_min`), and the VOF/normal/intercept
    halo exchange (`scalar_halo!`/`velocity_halo!`) keeps interface reconstruction correct
    at rank boundaries.

Global interior: 192×64. Dark fluid (negative `InterfaceSDF`) = water, light = air.

`examples/` has its own `Project.toml` (MPI/ImplicitGlobalGrid/WriteVTK, plus this repo
itself and the `WaterLily.jl@mpi-igg` branch it needs, via `[sources]`) so it doesn't
pull those into the package's own dependencies. Instantiate it once:
    julia --project=examples -e 'using Pkg; Pkg.instantiate()'

Run:
    julia --project=examples examples/mpi_dambreak.jl   # 1 rank (still goes through the
                                                          # same MPI-init path, just with
                                                          # no decomposition)

For the actual MPI-parallel run, launch through `MPI.mpiexec()` rather than a bare
`mpiexec` from your shell — the two can be *different MPI implementations*
(e.g. a system OpenMPI/PRRTE `mpiexec` vs. the `MPICH_jll` MPI.jl links by default),
and mixing launcher and library then fails at `MPI_Init` with something like
"Runtime environment uses unsupported PMI version PMIx". `MPI.mpiexec()` always
returns the launcher matching whatever MPI implementation Julia's `MPI.jl` is
actually linked against (system MPI too, if configured via `MPIPreferences.jl`):
    julia --project=examples -e 'using MPI; run(`\$(MPI.mpiexec()) -n 4 julia --project=examples examples/mpi_dambreak.jl`)'
    julia --project=examples -e 'using MPI; run(`\$(MPI.mpiexec()) -n 4 julia --project=examples examples/mpi_dambreak.jl -o0`)' # skip VTK output

To use a bare `mpiexec` from your shell instead (e.g. for a cluster's launcher), point
Julia's `MPI.jl` at that same system MPI once, via `MPIPreferences.jl` (needs the system
MPI's dev tools on PATH, e.g. `mpicc`/`ompi_info` — on Fedora, `dnf install openmpi-devel`
or `mpich-devel` first, then `module load mpi/openmpi-x86_64` or similar):
    julia --project=examples -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
This writes `examples/LocalPreferences.toml`; after that, `mpiexec -n 4 julia
--project=examples examples/mpi_dambreak.jl` works directly.
"""

using WaterLily
using MPI, ImplicitGlobalGrid
using InterfaceAdvection
using WriteVTK   # loads WaterLily's VTK extension; `output=false` just skips calling it
using Printf

const output = "-o0" ∉ ARGS

const T = Float32

# ── Tank / column geometry (global, interior cells) ──────────────────────────
const nx, ny = 192, 64             # tank interior size
const Wcol   = nx ÷ 4               # water column width
const Hcol   = round(Int, 0.75ny)   # water column height

# ── Physical parameters (nondimensionalized by U, L) ──────────────────────────
const L  = T(Hcol)          # length scale: initial column height
const U  = T(1)             # velocity scale, set explicitly: uBC=(0,0) here, and
                             # sim_time = t*U/L needs U≠0 to ever advance
const Fr = T(1)              # Froude number U/√(g·L) — picks the gravity magnitude
const g_mag = U^2/(Fr^2*L)
gravity(i,x,t) = i==2 ? -g_mag : zero(T)   # y (i=2) is vertical, gravity points down

const Re = T(300)            # moderate Re: coarse grid, numerically robust
const ν  = U*L/Re

# Water column sits in the tank's bottom-left corner; `max` of two half-plane
# distances approximates a box SDF (negative = dark fluid = water).
dam_sdf(x) = max(x[1]-Wcol, x[2]-Hcol)

# ── Build the distributed simulation ──────────────────────────────────────────
# `@distributed` extracts `dims=(nx,ny)` and `perdir` from this call, runs
# `init_waterlily_mpi((nx,ny); perdir=())`, and substitutes the rank-local dims
# back in — identical mechanism to `@distributed Simulation(...)`.
sim = @distributed TwoPhaseSimulation(
    (nx, ny), (zero(T), zero(T)), L;
    T, U, ν, g=gravity,
    λμ=T(1e-2), λρ=T(1e-3),      # air/water-like viscosity & density ratios
    InterfaceSDF=dam_sdf,
    perdir=(),                    # closed tank: no periodic directions
)

const me   = mpi_rank()
const comm = mpi_comm()

me==0 && @info "dam break: $(nx)×$(ny) global, column $(Wcol)×$(Hcol), " *
               "Re=$(Re), Fr=$(Fr), ranks=$(mpi_nprocs())"

# Initial water volume (in cells) — should stay ~constant; cVOF is conservative.
V0 = global_sum(@view sim.intf.f[inside(sim.intf.f)])

# `save!`/`vtkWriter` are generic over `AbstractSimulation` and just call
# `a.flow.*`/`a.intf.*` — no InterfaceAdvection-specific VTK extension needed.
wr = if output
    attrib = merge(default_attrib(), Dict("VOF" => a -> a.intf.f))
    vtkWriter("mpi_dambreak"; attrib, dir="vtk_data")
else
    nothing
end

# ── Time-stepping ─────────────────────────────────────────────────────────────
MPI.Barrier(comm)
t_start = MPI.Wtime()

t_end, max_steps = T(3), 2000
for step in 1:max_steps
    sim_step!(sim)   # TwoPhaseSimulation's own single-step method (MPFMomStep!)

    if step==1 || step%20==0
        umax = global_max(maximum(abs, sim.flow.u))
        pmax = global_max(maximum(abs, sim.flow.p))
        V    = global_sum(@view sim.intf.f[inside(sim.intf.f)])
        if me==0
            @printf("step %4d  t=%7.4f  Δt=%.3e  max|u|=%7.4f  max|p|=%7.3f  ΔVol/V0=%+.2e\n",
                    step, sim_time(sim), last(sim.flow.Δt), umax, pmax, (V-V0)/V0)
        end
        !isfinite(umax) && (me==0 && @error "blew up at step $step"; break)
        output && save!(wr, sim)
    end

    sim_time(sim) >= t_end && break
end
MPI.Barrier(comm)
t_elapsed = MPI.Wtime() - t_start

nsteps = length(sim.flow.Δt)-1
me==0 && @info "Done: $(nsteps) steps, t=$(round(sim_time(sim),digits=4)), " *
               "wall time $(round(t_elapsed,digits=2))s on $(mpi_nprocs()) rank(s)"
me==0 && output && (close(wr); @info "VTK output → vtk_data/mpi_dambreak_*.vti(pvti)")

# ── Finalize ───────────────────────────────────────────────────────────────────
finalize_global_grid()
