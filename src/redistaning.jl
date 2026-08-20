"""
    LevelSet(sim::TwoPhaseSimulation)

Signed level-set field `ϕ`, built from the VOF field `sim.intf.f` via `ϕ = 2f-1`.
Reuses `sim.intf.f⁰`'s storage as the buffer for `ϕ`, so no extra memory is allocated.
Not an actual signed-distance function until reinitialized with [`redistaning!`](@ref).
"""
struct LevelSet{D,T,Sf<:AbstractArray{T}}
    ϕ :: Sf
    ϕ⁰:: Sf
    ϕini::Sf
    L :: Sf
    perdir :: NTuple
    function LevelSet(sim::TwoPhaseSimulation)
        intf = sim.intf
        D,T,Sf = typeof(intf).parameters[1:3]
        ϕ = intf.f⁰
        ϕ⁰= intf.α
        ϕini = intf.fᶠ
        @. ϕ = 2intf.f-1
        ϕini .= ϕ
        L = sim.flow.σ

        new{D,T,Sf}(ϕ,ϕ⁰,ϕini,L,intf.perdir)
    end
end

function _redistaningStage!(ϕ,ϕ⁰,ϕini,L,dτ::T,α::T;perdir=()) where T
    computeL!(L,ϕ,ϕini;perdir)
    @loop ϕ[I] = α*ϕ⁰[I]+(1-α)*(ϕ[I]+T(dτ)*L[I]) over I ∈ inside(ϕ)
end

"""
    redistaning!(ls::LevelSet; d=5, dτ=0.5, perdir=())

Reinitialize `ls.ϕ` into a signed-distance function by marching the pseudo-time PDE

    ∂ₜϕ = sign(ϕ_ini)*(1-|𝛁ϕ|)

to steady state, using the third-order SSP Runge-Kutta (Shu-Osher) pseudo-time
integration with step `dτ` for a total pseudo-time `d`, effectively the
affected layer thickness.
"""
function redistaning!(ls::LevelSet{D,T}; d=5, dτ=0.5, perdir=()) where {D,T}
    ϕ,ϕ⁰,ϕini,L = ls.ϕ,ls.ϕ⁰,ls.ϕini,ls.L
    itmx = round(T,d/dτ)
    for _∈1:itmx
        ϕ⁰ .= ϕ
        _redistaningStage!(ϕ,ϕ⁰,ϕini,L,T(dτ),T(0);perdir)    # ϕ¹  = ϕⁿ+dτL(ϕⁿ)                (stage 1)
        BCf!(ϕ;perdir)
        _redistaningStage!(ϕ,ϕ⁰,ϕini,L,T(dτ),T(3/4);perdir)  # ϕ²  = ¾ϕⁿ+¼ϕ¹+¼dτL(ϕ¹)          (stage 2)
        BCf!(ϕ;perdir)
        _redistaningStage!(ϕ,ϕ⁰,ϕini,L,T(dτ),T(1/3);perdir)  # ϕⁿ⁺¹= ⅓ϕⁿ+⅔ϕ²+⅔dτL(ϕ²)          (stage 3)
        BCf!(ϕ;perdir)
    end
end

"""
    computeL!(L,ϕ,ϕini;perdir=())

Fill `L[I] = ϕini[I]*(1-|𝛁ϕ|)`, the reinitialization PDE's pseudo-time
right-hand side, at every interior cell. `|𝛁ϕ|²` is accumulated direction by
direction via [`𝛁ϕᵢ²`](@ref), using one-sided stencils near the domain edges
in each direction listed in `perdir` (periodic) or not (Neumann); see
`lowerL!`/`upperL!` for those boundary building blocks.
"""
function computeL!(L::AbstractArray{T,D},ϕ,ϕini;perdir=()) where {T,D}
    fill!(L,0)
    N = size(L)
    # L += ∇ϕᵢ²
    for i∈1:D
        tagper = (i in perdir)
        # lower boundary cell
        lowerL!(L,ϕ,ϕini,i,N,Val{tagper}())
        # inner cell
        @loop L[I] += 𝛁ϕᵢ²(
            ϕ[I-2δ(i,I)], ϕ[I-δ(i,I)], ϕ[I], ϕ[I+δ(i,I)], ϕ[I+2δ(i,I)],
            sign(ϕini[I])
        ) over I∈CartesianIndices(ntuple(k -> k==i ? (3:N[i]-2) : (2:N[k]-1), D))
        # upper boundary cell
        upperL!(L,ϕ,ϕini,i,N,Val{tagper}())
    end
    # L = smoothed_sign(ϕ_ini) * (1-√∇ϕᵢ²)
    @loop L[I] = (ϕini[I])*(1-sqrt(L[I])) over I∈inside(L)
end

# Neumann building block: the point 2 cells out is invalid, so it's mirrored
# from the 1-cell-out neighbor (zero-curvature extrapolation past the ghost).
lowerL!(L,ϕ,ϕini,i,N,::Val{false}) = @loop L[I] += 𝛁ϕᵢ²(
    ϕ[I-δ(i,I)], ϕ[I-δ(i,I)], ϕ[I], ϕ[I+δ(i,I)], ϕ[I+2δ(i,I)],
    sign(ϕini[I])
) over I∈slice(N,2,i,2)
upperL!(L,ϕ,ϕini,i,N,::Val{false}) = @loop L[I] += 𝛁ϕᵢ²(
    ϕ[I-2δ(i,I)], ϕ[I-δ(i,I)], ϕ[I], ϕ[I+δ(i,I)], ϕ[I+δ(i,I)],
    sign(ϕini[I])
) over I∈slice(N,N[i]-1,i,2)

# Periodic building block: the 1-cell-out neighbor is already valid (BCf! wraps
# the ghost layer); the 2-cell-out neighbor sits past the ghost, so it's fetched
# directly from the wrapped-around interior cell via CIj.
lowerL!(L,ϕ,ϕini,i,N,::Val{true}) = @loop L[I] += 𝛁ϕᵢ²(
    ϕ[CIj(i,I,N[i]-2)], ϕ[I-δ(i,I)], ϕ[I], ϕ[I+δ(i,I)], ϕ[I+2δ(i,I)],
    sign(ϕini[I])
) over I∈slice(N,2,i,2)
upperL!(L,ϕ,ϕini,i,N,::Val{true}) = @loop L[I] += 𝛁ϕᵢ²(
    ϕ[I-2δ(i,I)], ϕ[I-δ(i,I)], ϕ[I], ϕ[I+δ(i,I)], ϕ[CIj(i,I,3)],
    sign(ϕini[I])
) over I∈slice(N,N[i]-1,i,2)

minmod(a,b) = ifelse(abs(a)<=abs(b), a, b)

"""
    𝛁ϕᵢ²(a,b,c,d,e,s)

One-sided contribution to `|𝛁ϕ|²` along a single direction, from the 5-point
stencil `a,b,c,d,e = ϕ[I-2],ϕ[I-1],ϕ[I],ϕ[I+1],ϕ[I+2]` and upwind sign `s`
(typically `sign(ϕ_ini[I])`). Returns `0` when neither one-sided derivative is
upwind (i.e. this direction doesn't propagate information into `I`).
Derivative estimated with second-order ENO (minmod-limited) method by
[Sussman et al. (1999)](https://doi.org/10.1006/jcph.1998.6106).
"""
function 𝛁ϕᵢ²(a,b,c,d,e,s) where {T,D}
    dϕ⁺ = d-c
    dϕ⁻ = c-b
    
    d²ϕ⁺ = e+c-2d
    d²ϕ⁰ = d+b-2c
    d²ϕ⁻ = c+a-2b

    dϕᴿ = dϕ⁺-minmod(d²ϕ⁺,d²ϕ⁰)/2
    dϕᴸ = dϕ⁻+minmod(d²ϕ⁰,d²ϕ⁻)/2

    wᴿ = dϕᴿ*s
    wᴸ = dϕᴸ*s

    if     wᴿ < 0 && (wᴿ+wᴸ) < 0
        return abs2(dϕᴿ)
    elseif wᴸ > 0 && (wᴿ+wᴸ) > 0
        return abs2(dϕᴸ)
    else
        return zero(dϕᴿ)
    end
end
