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
export MPFMomStep!

include("metrics.jl")


"""
    TwoPhaseSimulation(dims::NTuple{N}, u_BC, L::Number;
                        λμ=1e-2, λρ=1e-3, η=0,
                        InterfaceSDF::Function=(x) -> -5-x[1],
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
- `T`: array element type.
- `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or
  AMD devices, respectively.

See: `WaterLily.Simulation`.
"""
mutable struct TwoPhaseSimulation <: AbstractSimulation
    sim :: Simulation
    intf :: cVOF
    function TwoPhaseSimulation(dims::NTuple{N}, args...;
                        T=Float32, mem=Array,
                        λμ=1e-2, λρ=1e-3, η=nothing, InterfaceSDF=nothing,
                        kwargs...) where N 

        # generate base simulation
        sim = Simulation(dims,args...; T, mem, kwargs...)

        # multipahse part
        intf = cVOF(dims;mem,T,InterfaceSDF,μ=sim.flow.ν,λμ,λρ,η,perdir=sim.flow.perdir)

        # correct wrong CFL
        sim.flow.Δt[end] .= min(last(flow.Δt),MPCFL(flow,intf))

        new(sim,intf)
    end
end
Base.getproperty(f::TwoPhaseSimulation, s::Symbol) = s in propertynames(f) ? getfield(f, s) : getfield(f.sim, s)
Base.setproperty!(f::TwoPhaseSimulation, s::Symbol, x) = s in propertynames(f) ? setproperty!(f,s,x) : setproperty!(f.sim,s,x)

export TwoPhaseSimulation

# overload for simStep
# solutoin from https://discourse.julialang.org/t/functions-from-different-modules-with-the-same-name/61505/2
import WaterLily: sim_step!
# TODO: support BDIM body
function sim_step!(sim::TwoPhaseSimulation,t_end;remeasure=false,max_steps=typemax(Int),verbose=false)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure)
        verbose && @printf("    tU/L=%10.6f, ΔtU/L=%.10f\n",sim_time(sim),last(sim.flow.Δt)*sim.U/sim.L);
        flush(stdout)
    end
end
function sim_step!(sim::TwoPhaseSimulation;remeasure=false)
    remeasure && measure!(sim)
    MPFMomStep!(sim.flow,sim.pois,sim.intf,sim.body)
end

export sim_step!

import WaterLily: load!
function load! end
export load!

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
