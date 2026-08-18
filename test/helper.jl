## Shared helpers for the main test suite. Use for reusable simulation setups across test files.

# Circular droplet advected by a 2D Taylor-Green vortex, fully periodic. Used to exercise
# `MPFMomStep!`/`cVOF` under a genuinely divergence-free, non-trivial velocity field (`flow.jl`,
# `cVOF.jl` test files), where the conservative VOF scheme should keep the dark-fluid volume constant.
function TGVDropletSim(N=16; T=Float64, mem=Array, λμ=0.1, λρ=0.1, kwargs...)
    κ = T(2π/N)
    R = T(N/4)
    TGV(i,x) = i==1 ? -sin(x[1]*κ)*cos(x[2]*κ) : cos(x[1]*κ)*sin(x[2]*κ)
    Inter(x) = √sum(abs2, x .- N/2) - R
    return TwoPhaseSimulation((N,N), (T(0),T(0)), N;
        T, mem, λμ, λρ, InterfaceSDF=Inter, u0=(i,x)->TGV(i,x), perdir=(1,2), kwargs...)
end

# Quiescent circular droplet: zero velocity, no forcing (g, viscosity, and surface tension all
# off by default). With nothing driving the flow, `MPFMomStep!` should be an exact no-op.
quiescentDropletSim(N=8; T=Float64, mem=Array, kwargs...) = TwoPhaseSimulation((N,N), (T(0),T(0)), T(N);
    T, mem, InterfaceSDF=x->√sum(abs2, x .- N/2) - N/4, perdir=(1,2), kwargs...)
