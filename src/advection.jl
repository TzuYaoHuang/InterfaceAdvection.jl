using Printf
import Random: shuffle


"""
    advect!(a,c,f,u¹,u²)

This is the spirit of the operator-split cVOF calculation.
It calculates the volume fraction after one fluxing.
Volume fraction field `f` is being fluxed with the averaged of two velocity -- `u¹` and `u²`.
"""
advect!(a::Flow{D}, c::cVOF, f=c.f, u¹=a.u⁰, u²=a.u) where {D} = advectVOF!(
    f,c.fᶠ,c.α,c.n̂,u¹,u²,a.Δt[end],c.c̄, c.ρuf,c.λρ; perdir=a.perdir
)

"""
    advectVOF!(f,fᶠ,α,n̂,u,u⁰,δt,c̄; perdir)

This is the expanded function for `advect!`. 
`fᶠ` is where to store face flux in one direction.
`c̄` is used to take care (de-)activation of dilation term.
"""
function advectVOF!(f::AbstractArray{T,D},fᶠ,α,n̂,u,u⁰,δt,c̄, ρuf,λρ; perdir=()) where {T,D}
    tol = 10eps(eltype(f))

    # get for dilation term
    @loop c̄[I] = ifelse(f[I]<0.5,0,1) over I ∈ CartesianIndices(f)
    
    # quasi-Strang splitting to avoid bias
    for d∈shuffle(1:D)
        # advect VOF field in d direction
        reconstructInterface!(f,α,n̂;perdir)
        getVOFFlux!(fᶠ,f,α,n̂,u,u⁰,δt,d, ρuf,λρ)
        @loop f[I] += fᶠ[I]-fᶠ[I+δ(d,I)] + c̄[I]*(∂(d,I,u)+∂(d,I,u⁰))*δt/2 over I∈inside(f)

        reportFillError(f,u,u⁰,d,tol)

        cleanWisp!(f,tol)
        BCf!(f;perdir)
    end
end

"""
    getVOFFlux!(fᶠ,f,α,n̂,u,u⁰,δt,d)
    getVOFFlux!(fᶠ,f,α,n̂,δl,d,IFace)

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
"""
function getVOFFlux!(fᶠ,f,α,n̂,u,u⁰,δt,d, ρuf,λρ)
    fᶠ .= 0
    @loop getVOFFlux!(fᶠ,f,α,n̂,δt/2*(u[IFace,d]+u⁰[IFace,d]),d,IFace, ρuf,λρ) over IFace∈inside_uWB(size(f),d)
    # 👿🤬 do not FUCKING put `ρuf ./= δt` here or else the second direction will be devided twice and make simulation explode
end
function getVOFFlux!(fᶠ,f::AbstractArray{T,D},α,n̂,δl,d,IFace, ρuf,λρ) where {T,D}
    # if face velocity is zero
    if δl == 0
        ρuf[IFace,d] = 0
        return nothing
    end

    # check upwind cell
    ICell = ifelse(δl>0, IFace-δ(d,IFace), IFace)
    
    # Full or empty cell
    sumAbsNhat=0
    for ii∈1:D sumAbsNhat+= abs(n̂[ICell,ii]) end
    if sumAbsNhat==0 || fullorempty(f[ICell])
        fᶠ[IFace] = f[ICell]*δl
        ρuf[IFace,d] = fᶠ2ρuf(IFace,fᶠ,δl,λρ)
        return nothing
    end

    # general case
    a = ifelse(δl>0, α[ICell]-n̂[ICell,d]*(1-δl), α[ICell])
    n̂dOrig = n̂[ICell,d]
    n̂[ICell,d] *= abs(δl)
    fᶠ[IFace] = getVolumeFraction(n̂, ICell, a)*δl
    ρuf[IFace,d] = fᶠ2ρuf(IFace,fᶠ,δl,λρ)
    n̂[ICell,d] = n̂dOrig
    return nothing
end

"""
reportFillError(f,u,u⁰,d,tol)

Report whenever `f` contains overfill or underfill elements with tolerence of `tol`.
Meanwhile the divergence of `u` and `u⁰` is displayed.
"""
function reportFillError(f,u,u⁰,d,tol)
    maxf, maxid = findmax(f)
    minf, minid = findmin(f)
    if maxf-1 > tol
        du⁰,du = abs(div(maxid,u⁰)),abs(div(maxid,u))
        @printf("|∇⋅u⁰| = %+13.8f, |∇⋅u| = %+13.8f\n",du⁰,du)
        errorMsg = "max VOF @ $(maxid.I) ∉ [0,1] @ direction $d, Δf = $(maxf-1)"
        (du⁰+du > 10) && error(errorMsg)
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
        errorMsg = "min VOF @ $(minid.I) ∉ [0,1] @ direction $d, Δf = $(-minf)"
        (du⁰+du > 10) && error(errorMsg)
        try
            error(errorMsg)
        catch e
            Base.printstyled("ERROR: "; color=:red, bold=true)
            Base.showerror(stdout, e, Base.catch_backtrace()); println()
        end
    end
end