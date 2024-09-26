module InterfaceAdvection
using WaterLily

include("util.jl")

include("VOFutil.jl")
export applyVOF!, BCVOF!,containInterface,fullorempty

include("normalEstimation.jl")
export reconstructInterface!, getInterfaceNormal_WY!, getInterfaceNormal_PCD!

include("PLIC.jl")
export getIntercept, getVolumeFraction

include("cVOF.jl")
export cVOF

include("advection.jl")
export advect!,advectVOF!,getVOFFlux!

include("surfaceTension.jl")

include("flow.jl")

mutable struct TwoPhaseSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    intf :: cVOF
    function TwoPhaseSimulation(dims::NTuple{N}, u_BC, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        λμ=1e-2,λρ=1e-3,η=0,
                        InterfaceSDF::Function=(x) -> -5-x[1],
                        uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array) where N 
        @assert !(isa(u_BC,Function) && isa(uλ,Function)) "`u_BC` and `uλ` cannot be both specified as Function"
        @assert !(isnothing(U) && isa(u_BC,Function)) "`U` must be specified if `u_BC` is a Function"
        isa(u_BC,Function) && @assert all(typeof.(ntuple(i->u_BC(i,zero(T)),N)).==T) "`u_BC` is not type stable"
        uλ = isnothing(uλ) ? ifelse(isa(u_BC,Function),(i,x)->u_BC(i,0.),(i,x)->u_BC[i]) : uλ
        U = isnothing(U) ? √sum(abs2,u_BC) : U # default if not specified
        flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        intf = cVOF(dims;arr=mem,T,InterfaceSDF,μ=1*ν,λμ,λρ,η,perdir)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir),intf)
    end
end

export TwoPhaseSimulation

end
