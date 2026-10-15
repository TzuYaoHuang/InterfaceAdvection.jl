"""
dambreakwithbody.jl — 2D dam break impacting a fixed square obstacle with InterfaceAdvection.jl
================================================================================================
A rectangular water column collapses under gravity inside a closed tank (free-slip walls, no
periodic directions) and slides across the floor until it hits a square block that sits well
downstream of the dam, away from the initial water column. This exercises the fluid-structure
coupling of InterfaceAdvection.jl's two-phase solver with a `WaterLily.AutoBody`: the block is
immersed via the same BDIM machinery WaterLily uses for single-phase flow, `measure!`d once at
t=0 (it is static, so no `remeasure` is needed during time-stepping), and the two-phase momentum
solve (`MPFMomStep!`) forces the velocity field through `sim.body` exactly like `mom_step!` does.

Global interior: 192×64. Dark fluid (negative `InterfaceSDF`) = water, light = air. The block is
a fixed square of side `Sside` resting on the tank floor at `x≈Scx`, far to the right of the
`Wcol`-wide water column so the column has room to collapse and accelerate before impact.

`examples/` has its own `Project.toml` (Plots/StaticArrays, plus this repo itself via `[sources]`)
so it doesn't pull those into the package's own dependencies. Instantiate it once:
    julia --project=examples -e 'using Pkg; Pkg.instantiate()'

Run:
    julia --project=examples examples/dambreakwithbody.jl
This writes an animated `examples/dambreakwithbody.gif` showing the block (grey), the vorticity
field (red/blue), and the water interface (black line) as the column collapses, hits the block,
and splashes over it.
"""

using WaterLily
using InterfaceAdvection
using StaticArrays
using Plots
using Printf

ENV["GKSwstype"] = "nul"   # headless GR: don't try to open a window while rendering frames

const T = Float32

# ── Tank / column geometry (interior cells) ──────────────────────────────────
const nx, ny = 192, 96             # tank interior size
const Wcol   = nx ÷ 4               # water column width
const Hcol   = round(Int, 0.75ny)   # water column height

# ── Square obstacle: fixed, resting on the floor, away from the dam ──────────
const Sside = ny ÷ 6                       # block side length
const Scx   = nx * 3 ÷ 4                   # block center x (far from the water column)
const Scy   = Sside / 2                    # block center y (sitting on the floor)

# ── Physical parameters (nondimensionalized by U, L) ──────────────────────────
const L  = T(Hcol)          # length scale: initial column height
const U  = T(1)             # velocity scale, set explicitly since uBC=(0,0)
const Fr = T(1)              # Froude number U/√(g·L) — picks the gravity magnitude
const g_mag = U^2/(Fr^2*L)
gravity(i,x,t) = i==2 ? -g_mag : zero(T)   # y (i=2) is vertical, gravity points down

const Re = T(Inf)            # moderate Re: coarse grid, numerically robust
const ν  = U*L/Re

# Water column sits in the tank's bottom-left corner; `max` of two half-plane
# distances approximates a box SDF (negative = dark fluid = water).
dam_sdf(x) = max(x[1]-Wcol, x[2]-Hcol)

# Exact signed distance to the axis-aligned square obstacle (negative inside).
function square_sdf(x,t)
    q = abs.(x .- SA[T(Scx),T(Scy)]) .- T(Sside)/2
    return √sum(abs2, max.(q, zero(T))) + min(maximum(q), zero(T))
end

# ── Build the simulation ──────────────────────────────────────────────────────
sim = TwoPhaseSimulation(
    (nx, ny), (zero(T), zero(T)), L;
    T, U, ν, g=gravity,
    λμ=T(1e-2), λρ=T(1e-3),      # air/water-like viscosity & density ratios
    InterfaceSDF=dam_sdf,
    body=AutoBody(square_sdf),    # fixed square block, immersed via BDIM
    perdir=(),                    # closed tank: no periodic directions
)

@info "dam break with body: $(nx)×$(ny), column $(Wcol)×$(Hcol), " *
      "block $(Sside)×$(Sside) at x=$(Scx), Re=$(Re)"

# Initial water volume (in cells) — should stay ~constant; cVOF is conservative.
V0 = sum(@view sim.intf.f[inside(sim.intf.f)])

# ── Animate: body (grey), vorticity (red/blue), interface (black line) ───────
R = inside(sim.flow.p)
bx = [Scx-Sside/2, Scx+Sside/2, Scx+Sside/2, Scx-Sside/2]
by = [Scy-Sside/2, Scy-Sside/2, Scy+Sside/2, Scy+Sside/2]

t_end, step = T(10), T(0.05)
anim = @animate for tᵢ in range(zero(T), t_end; step)
    sim_step!(sim, tᵢ; remeasure=false)   # body is static: measured once at t=0 already

    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ[R]; clims=(-25,25), border=:none,
          title=@sprintf("tU/L=%.2f",tᵢ), legend=false)
    contour!(sim.intf.f[R]'; levels=[0.5], lines=(:black,2))
    addbody(bx, by; c=:grey30)

    V = sum(@view sim.intf.f[inside(sim.intf.f)])
    @printf("tU/L=%6.3f  Δt=%.3e  max|u|=%7.4f  ΔVol/V0=%+.2e\n",
            tᵢ, last(sim.flow.Δt), maximum(abs, sim.flow.u), (V-V0)/V0)
end
gif(anim, joinpath(@__DIR__,"dambreakwithbody.gif"); fps=15)
@info "Done: wrote examples/dambreakwithbody.gif"
