"""
    LevelSet(sim::TwoPhaseSimulation)

Signed level-set field `ϕ`, built from the VOF field `sim.intf.f` via `ϕ = 2f-1`.
Reuses `sim.intf.f⁰`'s storage as the buffer for `ϕ`, so no extra memory is allocated.
Not an actual signed-distance function until reinitialized with [`redistaning`](@ref).
"""
struct LevelSet{D,T,Sf<:AbstractArray{T}}
    ϕ :: Sf
    ϕ⁰:: Sf
    ϕini::Sf
    function LevelSet(sim::TwoPhaseSimulation)
        intf = sim.intf
        D,T,Sf = typeof(intf).parameters[1:3]
        ϕ = intf.f⁰
        ϕ⁰= intf.α
        ϕini = intf.fᶠ
        @. ϕ = 2intf.f-1
        ϕini .= ϕ

        new{D,T,Sf}(ϕ,ϕ⁰,ϕini)
    end
end

_redistaningStage!(ϕ,ϕ⁰,ϕini,dτ::T,α::T) where T = @loop ϕ[I] = α*ϕ⁰[I]+(1-α)*(ϕ[I]+T(dτ)*redistaning(I,ϕ,ϕini)) over I ∈ inside(ϕ)

"""
    redistaning(ls::LevelSet; d=5, dτ=0.25, perdir=())

Reinitialize `ls.ϕ` into a signed-distance function by marching the pseudo-time PDE

    ∂ₜϕ = sign(ϕ_ini)*(1-|𝛁ϕ|)

to steady state, using the third-order SSP Runge-Kutta (Shu-Osher) pseudo-time
integration with step `dτ` for a total pseudo-time `d`, effectively the
affected layer thickness.
"""
function redistaning!(ls::LevelSet{D,T}; d=5, dτ=0.5, perdir=()) where {D,T}
    ϕ,ϕ⁰,ϕini = ls.ϕ,ls.ϕ⁰,ls.ϕini
    itmx = round(T,d/dτ)
    for _∈1:itmx
        ϕ⁰ .= ϕ
        _redistaningStage!(ϕ,ϕ⁰,ϕini,T(dτ),T(0))    # ϕ¹  = ϕⁿ+dτL(ϕⁿ)                (stage 1)
        BCf!(ϕ;perdir)
        _redistaningStage!(ϕ,ϕ⁰,ϕini,T(dτ),T(3/4))  # ϕ²  = ¾ϕⁿ+¼ϕ¹+¼dτL(ϕ¹)          (stage 2)
        BCf!(ϕ;perdir)
        _redistaningStage!(ϕ,ϕ⁰,ϕini,T(dτ),T(1/3))  # ϕⁿ⁺¹= ⅓ϕⁿ+⅔ϕ²+⅔dτL(ϕ²)          (stage 3)
        BCf!(ϕ;perdir)
    end
end

"""
    redistaning(I,ϕ)

Pointwise pseudo-time right-hand side `sign(ϕ_ini)*(1-|𝛁ϕ|)`, with `𝛁ϕ` estimated by
central difference at cell `I`.
Derivative estimated with first-order ENO (minmod) method by [Sussman & Fatemi (1999)](https://doi.org/10.1137/S1064827596298245)
"""
function redistaning(I,ϕ::AbstractArray{T,D},ϕini) where {T,D}
    𝛁ϕ² = zero(T)
    s = sign(ϕini[I])
    for i∈1:D
        dϕ⁺ = ϕ[I+δ(i,I)]-ϕ[I]
        dϕ⁻ = ϕ[I]-ϕ[I-δ(i,I)]
        dϕᶜ = (dϕ⁺+dϕ⁻)/2
        if dϕ⁺*s < 0 && dϕᶜ*s < 0
            𝛁ϕ² += abs2(dϕ⁺)
        elseif dϕ⁻*s > 0 && dϕᶜ*s > 0
            𝛁ϕ² += abs2(dϕ⁻)
        end
    end
    return s*(1-sqrt(𝛁ϕ²))
end
