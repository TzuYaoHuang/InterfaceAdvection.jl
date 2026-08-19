module InterfaceAdvection

# some necessary function from WaterLily
using WaterLily,Printf
import WaterLily: @loop,div,inside,∂,inside_u,CI,CIj,slice,size_u, NoBody, check_fn

include("util.jl")

include("VOFutil.jl")
export applyVOF!, BCVOF!,containInterface,fullorempty

include("normalEstimation.jl")
export reconstructInterface!, getInterfaceNormal_WY!, getInterfaceNormal_PCD!, getInterfaceNormal_WH!, getInterfaceNormal_MYC!, getInterfaceNormal_Y!

include("PLIC.jl")
export getIntercept, getVolumeFraction

include("cVOF.jl")
export cVOF

include("advection.jl")
export advect!,advectVOF!,getVOFFlux!

include("surfaceTension.jl")
export surfTen!,getCurvature,getPopinetHeight

include("flow.jl")
export MPFMomStep!, upwind, minmod, Koren, vanAlbada1, Sweby, superbee

include("metrics.jl")

include("redistaning.jl")
export LevelSet, redistaning


"""
    TwoPhaseSimulation(dims::NTuple{N}, u_BC, L::Number;
                        λμ=1e-2, λρ=1e-3, η=0,
                        InterfaceSDF::Function=(x) -> -5-x[1],
                        λ=Koren, normalScheme=getInterfaceNormal_WH!,
                        T=Float32, mem=Array, kwargs...)

Constructor for a two-phase flow simulation based on WaterLily.jl.
Wraps a `WaterLily.Simulation` together with a `cVOF` interface field, and accepts
all of `WaterLily.Simulation`'s keyword arguments (e.g. `Δt`, `ν`, `g`, `U`, `ϵ`,
`perdir`, `exitBC`, `body`) plus the following for multiphase flow:

- `λμ`: ratio of dynamic viscosity, light/dark.
- `λρ`: ratio of density, light/dark.
- `η`: surface tension coefficient.
- `InterfaceSDF`: signed distance function for the interface, where the dark
  fluid occupies the region of negative distance, e.g. `sdf(x) = 5-x[1]`
- `λ`: momentum-flux interpolation scheme `λ(u,c,d)` (upstream, central, downstream)
  used by `MPFMomStep!`'s mass-momentum consistent advection, e.g. `Koren` (default),
  `minmod`, `vanAlbada1`, `superbee`, `upwind`, or `quick`.
  Forwarded to `WaterLily.Simulation` and stored on `sim.flow.λ`, same as `WaterLily`'s own
  convective scheme keyword. See `flow.jl`.
- `normalScheme`: interface-normal reconstruction scheme `normalScheme(f,n̂,I)` used by
  `reconstructInterface!` during VOF advection, e.g. `getInterfaceNormal_WH!` (default),
  `getInterfaceNormal_WY!`, `getInterfaceNormal_MYC!`, or `getInterfaceNormal_Column!`.
  Forwarded to `cVOF` and stored on `sim.intf.normalScheme`. See `normalEstimation.jl`.
- `T`: array element type.
- `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or
  AMD devices, respectively.

See: `WaterLily.Simulation`.
"""
mutable struct TwoPhaseSimulation <: AbstractSimulation
    sim :: Simulation
    intf :: cVOF
    function TwoPhaseSimulation(dims::NTuple{N}, u_BC, L::Number;
                        T=Float32, mem=Array,
                        λμ=1e-2, λρ=1e-3, η=nothing, InterfaceSDF=nothing,
                        λ=Koren, normalScheme=getInterfaceNormal_WH!,
                        kwargs...) where N

        # generate base simulation
        sim = Simulation(dims, u_BC, L; T, mem, λ, kwargs...)
        flow = sim.flow

        # multipahse part
        intf = cVOF(dims;mem,T,InterfaceSDF,μ=flow.ν,λμ,λρ,η,normalScheme,perdir=flow.perdir)

        # correct wrong CFL
        flow.Δt[end] = min(last(flow.Δt),MPCFL(flow,intf))

        new(sim,intf)
    end
end
Base.getproperty(f::TwoPhaseSimulation, s::Symbol) = s in propertynames(f) ? getfield(f, s) : getfield(f.sim, s)
Base.setproperty!(f::TwoPhaseSimulation, s::Symbol, x) = s in propertynames(f) ? setfield!(f,s,x) : setproperty!(f.sim,s,x)

export TwoPhaseSimulation

# overload for simStep
# solutoin from https://discourse.julialang.org/t/functions-from-different-modules-with-the-same-name/61505/2
import WaterLily: sim_step!, sim_info
# TODO: support BDIM body
# `sim_step!(sim,t_end;...)` falls through to WaterLily's generic AbstractSimulation loop,
# which drives this per-step method and `sim_info` below.
function sim_step!(sim::TwoPhaseSimulation;remeasure=false,udf=nothing,kwargs...)
    remeasure && measure!(sim)
    MPFMomStep!(sim.flow,sim.pois,sim.intf,sim.body;udf,kwargs...)
end
function sim_info(sim::TwoPhaseSimulation)
    @printf "    tU/L=%10.6f, ΔtU/L=%.10f\n" sim_time(sim) last(sim.flow.Δt)*sim.U/sim.L
    flush(stdout)
end

import WaterLily: load!
function load! end

# Backward compatibility for extensions
if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/IntfAdvAMDGPUExt.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/IntfAdvCUDAExt.jl")
        @require ReadVTK = "dc215faf-f008-4882-a9f7-a79a826fadc3" include("../ext/IntfAdvReadVTKExt.jl")
    end
end

end # module
