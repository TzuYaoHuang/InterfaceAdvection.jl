"""
    LevelSet(sim::TwoPhaseSimulation)

Signed level-set field `ϕ`, built from the VOF field `sim.intf.f` via `ϕ = 2f-1`.
Reuses `sim.intf.f⁰`'s storage as the buffer for `ϕ`, so no extra memory is allocated.
Not an actual signed-distance function until reinitialized with [`redistaning`](@ref).
"""
struct LevelSet{D,T,Sf<:AbstractArray{T}}
    ϕ :: Sf
    function LevelSet(sim::TwoPhaseSimulation)
        intf = sim.intf
        D,T,Sf = typeof(intf).parameters[1:3]
        ϕ = intf.f⁰
        @. ϕ = 2intf.f-1
        new{D,T,Sf}(ϕ)
    end
end

"""
    redistaning(ls::LevelSet; d=5, dτ=0.25, perdir=())

Reinitialize `ls.ϕ` into a signed-distance function by marching the pseudo-time PDE

    ∂ₜϕ = sign(ϕ)*(1-|𝛁ϕ|)

to steady state, using forward-Euler pseudo-timestep `dτ` for a total pseudo-time `d`, effective the affected layer thickness.
"""
function redistaning!(ls::LevelSet{D,T}; d=5, dτ=0.25, perdir=()) where {D,T}
    ϕ = ls.ϕ
    itmx = round(T,d/dτ)
    for _∈1:itmx
        @inside ϕ[I] = ϕ[I]+T(dτ)*redistaning(I,ϕ)
        BCf!(ϕ;perdir)
    end
end

"""
    redistaning(I,ϕ)

Pointwise pseudo-time right-hand side `sign(ϕ)*(1-|𝛁ϕ|)`, with `𝛁ϕ` estimated by
central difference at cell `I`.
"""
function redistaning(I,ϕ::AbstractArray{T,D}) where {T,D}
    𝛁ϕ² = zero(T)
    for i∈1:D
        𝛁ϕ² += abs2(ϕ[I+δ(i,I)]-ϕ[I-δ(i,I)])
    end
    return sign(ϕ[I])*(1-sqrt(𝛁ϕ²)/2)
end
