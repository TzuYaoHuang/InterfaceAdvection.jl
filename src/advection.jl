using Printf
import Random: shuffle


"""
    advect!(a,c,f,u¹,u²,dt)

This is the spirit of the operator-split cVOF calculation.
It calculates the volume fraction after one full time step.
Volume fraction field `f` is being fluxed with the averaged of two velocity -- `u¹` and `u²`.
"""
advect!(a::Flow{D}, c::cVOF, f=c.f, u¹=a.u⁰, u²=a.u, dt=a.Δt[end]) where {D} = advectVOF!(
    f,c.fᶠ,c.α,c.n̂,u¹,u²,dt,c.c̄,c.ρuf,c.λρ; 
    perdir=a.perdir, 
    # dirO=1:D
    # dirO=shuffle(1:D)
    dirO=ntuple(i->mod(length(a.Δt)+i,D)+1, D)
)

"""
    advectVOF!(f,fᶠ,α,n̂,u,u⁰,δt,c̄,ρuf,λρ; perdir)

This is the expanded function for `advect!`. 
`fᶠ` is where to store face flux in one direction.
`c̄` is used to take care (de-)activation of dilation term.
`ρuf` stores the mass flux for mass-momentum consistent method.
"""
function advectVOF!(f::AbstractArray{T,D},fᶠ,α,n̂,u,u⁰,Δt,c̄,ρuf,λρ; perdir=(), dirO=shuffle(1:D)) where {T,D}
    tol = 10eps(T)

    ρuf .= 0

    # get for dilation term
    @loop c̄[I] = ifelse(f[I]<0.5,0,1) over I ∈ CartesianIndices(f)

    dirOrder = isnothing(dirO) ? shuffle(1:D) : dirO

    # Operator splitting to avoid bias
    # Reference for splitting method: http://www.othmar-koch.org/splitting/index.php

    # Second-order Auzinger-Ketcheson
    # s2 = 1/√2
    # OpOrder = D==2 ? SVector{4,Int8}(1, 2, 1, 2) : SVector{6,Int8}(1, 2, 3, 2, 3, 1)
    # OpCoeff = D==2 ? SVector{4,T}(1-s2, s2, s2, 1-s2) : SVector{6,T}(1/2, 1-s2, s2, s2, 1-s2, 1/2)

    # Second-order Strang
    # OpOrder = D==2 ? SVector{3,Int8}(1, 2, 1) : SVector{5,Int8}(1, 2, 3, 2, 1)
    # OpCoeff = D==2 ? SVector{3,T}(1/2, 1, 1/2) : SVector{5,T}(1/2, 1/2, 1, 1/2, 1/2)

    # First-order Lie-Trotter
    OpOrder = D==2 ? SVector{2,Int8}(1, 2) : SVector{3,Int8}(1, 2, 3)
    OpCoeff = D==2 ? SVector{2,T}(1, 1) : SVector{3,T}(1, 1, 1)

    for iOp∈eachindex(OpOrder)
        d = dirOrder[OpOrder[iOp]]
        δt = OpCoeff[iOp]*Δt

        # advect VOF field in d direction
        reconstructInterface!(f,α,n̂;perdir)
        getVOFFlux!(fᶠ,f,α,n̂,u,u⁰,δt,d,ρuf,λρ)
        @loop f[I] += fᶠ[I]-fᶠ[I+δ(d,I)] + c̄[I]*(∂(d,I,u)+∂(d,I,u⁰))*δt/2 over I∈inside(f)

        reportFillError(f,n̂,u,u⁰,δt,d,tol)

        cleanWisp!(f,tol)
        BCf!(f;perdir)
    end
    # NOTE: It is not necessary to remove the mass addition. This correction is useless and make explosion faster.
    # @loop f[I] -= c̄[I]*(div(I,u)+div(I,u⁰))*Δt/2 over I∈inside(f)
    # cleanWisp!(f,tol)
    # BCf!(f;perdir)
end

NVTX.@annotate function advectVOF1d!(f::AbstractArray{T,D},fᶠ,α,n̂,u,u⁰,δt,c̄,ρuf,λρ,d; perdir=(), tol=10eps(T)) where {T,D}
    reconstructInterface!(f,α,n̂;perdir)
    getVOFFlux!(fᶠ,f,α,n̂,u,u⁰,δt,d,ρuf,λρ)
    NVTX.@range "update f" begin
        @loop f[I] += fᶠ[I]-fᶠ[I+δ(d,I)] + c̄[I]*(∂(d,I,u)+∂(d,I,u⁰))*δt/2 over I∈inside(f)
        backend_sync!(f)
    end
    # reportFillError(f,n̂,u,u⁰,δt,d,tol)

    cleanWisp!(f,tol)
    BCf!(f;perdir)
end

function advectρuu1d!(ρu, uOld, um, ρuf, c̄, u, u⁰, d; perdir=())
end

"""
    getVOFFlux!(fᶠ,f,α,n̂,u,u⁰,δt,d,ρuf,λρ)
    getVOFFlux!(fᶠ,f,α,n̂,δl,d,IFace,ρuf,λρ)

Get the face flux according to upwind donor-acceptor cell concept. 
The reconstructed dark fluid volume orverlapped with the advection sweep volume is advected to the next cell. 
- `fᶠ`: where the flux is stored
- `f`: volume fraction field
- `α`: intercept
- `n̂`: interface normal
- `u`, `u⁰`: the VOF is fluxed with the average of two velocity
- `δt`: time step size
- `δl`: advection sweep length, essentially `uδt`
- `d`: the direction of cell faces that flux is calculated at
- `ρuf`: mass flux of collocated faces
- `λρ`: density ratio
"""
NVTX.@annotate function getVOFFlux!(fᶠ,f,α,n̂,u,u⁰,δt,d,ρuf,λρ)
    fᶠ .= 0
    @loop getVOFFlux!(fᶠ,f,α,n̂,δt/2*(u[IFace,d]+u⁰[IFace,d]),d,IFace, ρuf,λρ) over IFace∈inside_uWB(size(f),d)
    # 👿🤬 do not FUCKING put `ρuf ./= δt` here or else the second direction will be devided twice and make simulation explode
    backend_sync!(f)
end
function getVOFFlux!(fᶠ,f::AbstractArray{T,D},α,n̂,δl,d,IFace,ρuf,λρ) where {T,D}
    # if face velocity is zero
    if δl == 0
        return nothing
    end

    # check upwind cell
    ICell = ifelse(δl>0, IFace-δ(d,IFace), IFace)
    
    # Full or empty cell
    sumAbsNhat=T(0)
    for ii∈1:D sumAbsNhat+= abs(n̂[ICell,ii]) end
    if sumAbsNhat==0 || fullorempty(f[ICell])
        fᶠ[IFace] = f[ICell]*δl
        ρuf[IFace,d] += fᶠ2ρuf(IFace,fᶠ,δl,λρ)
        return nothing
    end

    # general case
    a = ifelse(δl>0, α[ICell]-n̂[ICell,d]*(1-δl), α[ICell])
    n̂Cell = ntuple((ii)->n̂[ICell,ii]*ifelse(ii==d,abs(δl),1),D)
    fᶠ[IFace] = getVolumeFraction(n̂Cell, a)*δl
    ρuf[IFace,d] = fᶠ2ρuf(IFace,fᶠ,δl,λρ)
    return nothing
end

"""
reportFillError(f,n̂,u,u⁰,d,tol)

Report whenever `f` contains overfill or underfill elements with tolerence of `tol`.
Meanwhile the divergence of `u` and `u⁰` is displayed, so is the normal components `n̂`.
"""
function reportFillError(f::AbstractArray{T,D},n̂,u,u⁰,δt,d,tol) where {T,D}
    maxf, maxid = findmax(f)
    minf, minid = findmin(f)
    if maxf-1 > tol
        du⁰,du = abs(div(maxid,u⁰)),abs(div(maxid,u))
        @printf("|∇⋅u⁰| = %+13.8f, |∇⋅u| = %+13.8f\n",du⁰,du)
        for d∈1:D
            @printf("    %d -- uLeftδt: %+13.8f, uRightδt: %+13.8f\n", d, (u[maxid,d]+u⁰[maxid,d])*δt, (u[maxid+δ(d,maxid),d]+u⁰[maxid+δ(d,maxid),d])*δt)
        end
        for d∈1:D
            @printf("   n%d -- %+13.8f\n", d, n̂[maxid,d])
        end
        errorMsg = "max VOF @ $(maxid.I) ∉ [0,1] @ direction $d, Δf = $(maxf-1)"
        (du⁰+du > 10) && error("divergence, $(du⁰+du), is exploding!")
        try
            error(errorMsg)
        catch e
            Base.printstyled("ERROR: "; color=:red, bold=true)
            Base.showerror(stdout, e, Base.catch_backtrace()); println()
        end
    end
    if minf < -tol
        du⁰,du = abs(div(minid,u⁰)),abs(div(minid,u))
        @printf("|∇⋅u⁰| = %+13.8f, |∇⋅u| = %+13.8f\n",du⁰,du)
        for d∈1:D
            @printf("    %d -- uLeftδt: %+13.8f, uRightδt: %+13.8f\n", d, (u[minid,d]+u⁰[minid,d])*δt, (u[minid+δ(d,minid),d]+u⁰[minid+δ(d,minid),d])*δt)
        end
        for d∈1:D
            @printf("   n%d -- %+13.8f\n", d, n̂[minid,d])
        end
        errorMsg = "min VOF @ $(minid.I) ∉ [0,1] @ direction $d, Δf = $(-minf)"
        (du⁰+du > 10) && error("divergence, $(du⁰+du), is exploding!")
        try
            error(errorMsg)
        catch e
            Base.printstyled("ERROR: "; color=:red, bold=true)
            Base.showerror(stdout, e, Base.catch_backtrace()); println()
        end
    end
end