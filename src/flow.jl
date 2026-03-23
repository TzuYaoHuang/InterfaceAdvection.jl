import WaterLily: accelerate!, median, update!, project!, scale_u!, exitBC!,perBC!,residual!,mult, flux_out, vanLeer, L∞, ϕ
import LinearAlgebra: ⋅, rmul!, axpy!

# I need to re-define the flux limiter or else the TVD property cannot conserve
@fastmath upwind(u,c,d) = c
@fastmath cen(u,c,d) = (c+d)/2
@fastmath minmod(u,c,d) = median((3c-u)/2,c,(c+d)/2)
@fastmath trueKoren(u,c,d) = median((7c+d-2u)/6,c,median(2c-u,c,d))
@fastmath koren(u,c,d) = median((5c+2d-u)/6,c,median(2c-1u,c,d))
@fastmath function vanAlbada1(u,c,d)
    α,β = c-u,d-c
    return c+max(α*β,0)*ifelse(α==β && α==0, 0, (α+β)/(α^2+β^2))/2
end
@fastmath Sweby(u,c,d,γ=1.5,s=sign(d-u)) = (c≤min(u,d) || c≥max(u,d)) ? c : c + s*max(0, min(s*γ*(c-u),s*(d-c)), min(s*(c-u),s*γ*(d-c)))/2
@inline superbee(u,c,d) = Sweby(u,c,d,2)
@fastmath TVDcen(u,c,d,s=sign(d-u)) = (c≤min(u,d) || c≥max(u,d)) ? c : c + s*min(s*(c-u),s*(d-c)/2)
@fastmath TVDdown(u,c,d,s=sign(d-u)) = (c≤min(u,d) || c≥max(u,d)) ? c : c + s*min(s*(c-u),s*(d-c))


@inline limiter(u,c,d) = trueKoren(u,c,d)

# u: advecting, f: advected
@inline ϕu(j,i,I,Ψ,u,f,ρuf,fOld,δt,λρ,λ=limiter) = (@inbounds Ψ>0 ? 
    ϕq(j,i,I,fOld,ρuf,u,f[I-2δ(j,I)],f[I-δ(j,I)],f[I],δt,λρ,λ) : 
    ϕq(j,i,I,fOld,ρuf,u,f[I+δ(j,I)],f[I],f[I-δ(j,I)],δt,λρ,λ)
)
@inline ϕuP(j,i,Ip,I,Ψ,u,f,ρuf,fOld,δt,λρ,λ=limiter) = (@inbounds Ψ>0 ? 
    ϕq(j,i,I,fOld,ρuf,u,f[Ip],f[I-δ(j,I)],f[I],δt,λρ,λ) : 
    ϕq(j,i,I,fOld,ρuf,u,f[I+δ(j,I)],f[I],f[I-δ(j,I)],δt,λρ,λ)
)
@inline ϕuL(j,i,I,Ψ,u,f,ρuf,fOld,δt,λρ,λ=limiter) = (@inbounds Ψ>0 ? 
    ϕq(j,i,I,fOld,ρuf,u,2f[I-δ(j,I)]-f[I],f[I-δ(j,I)],f[I],δt,λρ,λ) : 
    ϕq(j,i,I,fOld,ρuf,u,f[I+δ(j,I)],f[I],f[I-δ(j,I)],δt,λρ,λ)
)
@inline ϕuR(j,i,I,Ψ,u,f,ρuf,fOld,δt,λρ,λ=limiter) = (@inbounds Ψ<0 ? 
    ϕq(j,i,I,fOld,ρuf,u,2f[I]-f[I-δ(j,I)],f[I],f[I-δ(j,I)],δt,λρ,λ) : 
    ϕq(j,i,I,fOld,ρuf,u,f[I-2δ(j,I)],f[I-δ(j,I)],f[I],δt,λρ,λ)
)

function ϕq(j,i,Ii,fOld::AbstractArray{T,Dv},ρuf,u,uu,cc,dd,δt,λρ,λ) where {T,Dv}
    # I the lower face of staggered cell I
    I = CI(Ii.I[1:end-1])
    Ψ = ϕ(i,CI(I,j),ρuf)

    IiCell = ifelse(Ψ>0, Ii-δ(j,Ii), Ii)

    vI = cc
    vd = λ(uu,cc,dd)
    va = 2vI-vd
    # if fullorempty(fOld[IiCell]) return Ψ*vd end

    mOut = abs(Ψ)*δt
    mOld = getρ(IiCell,fOld,λρ)
    if mOut > mOld return Ψ*vI end
    l2 = abs(mOut)/mOld
    l1 = 1-l2

    vb = l2*va + l1*vd
    return Ψ*(vb+vd)/2
end


@fastmath function MPFMomStep!(a::Flow{D,T}, b::AbstractPoisson, c::cVOF, d::AbstractBody;δt = a.Δt[end]) where {D,T}
    copyto!(a.u⁰, a.u); copyto!(c.f⁰, c.f)
    t₁ = sum(a.Δt); t₀ = t₁-δt; tₘ = t₁-δt/2
    # TODO: check if BC doable for ρu

    # predictor u(n) → u(n+1/2∘) with u(n)
    @log "p"
    dtCoeff = T(1/2)

    u2ρu!(c.ρu,a.u⁰,c.f⁰,c.λρ); BC!(c.ρu,a.uBC,a.exitBC,a.perdir)
    advectfq!(a, c, c.f⁰, a.u⁰, a.u, a.u, δt)

    # TODO: include measure
    fill!(a.μ₀,1)
    @. c.f⁰ = (c.f⁰+c.f)/2
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f⁰,c.α,c.n̂,c.fᶠ,c.λμ,c.μ,c.λρ,c.η;perdir=a.perdir)
    u2ρu!(c.n̂,a.u⁰,c.f,c.λρ) # steal n̂ as original momentum
    updateU!(a.u,c.ρu,c.n̂,a.f,δt,c.f⁰,c.λρ,tₘ,a.g,a.uBC,dtCoeff); BC!(a.u,a.uBC,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f⁰,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b,dtCoeff); BC!(a.u,a.uBC,a.exitBC,a.perdir)

    # copyto!(c.f, c.f⁰)
    # copyto!(a.u, a.u⁰)

    # corrector u(n) → u(n+1) with u(n+1/2∘)
    @log "c"
    copyto!(c.f⁰, c.f)

    u2ρu!(c.ρu,a.u⁰,c.f,c.λρ); BC!(c.ρu,a.uBC,a.exitBC,a.perdir)
    advectfq!(a, c, c.f, a.u, a.u, a.u⁰, δt)
    
    # TODO: include measure
    fill!(a.μ₀,1)
    # TODO: viscous term and surface tension term should be evaluated 
    # at the end of time step to avoid divide by wrong ρ
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f,c.α,c.n̂,c.fᶠ,c.λμ,c.μ,c.λρ,c.η;perdir=a.perdir) 
    u2ρu!(c.n̂,a.u⁰,c.f,c.λρ) # steal n̂ as original momentum
    updateU!(a.u,c.ρu,c.n̂,a.f,δt,c.f,c.λρ,t₁,a.g,a.uBC); BC!(a.u,a.uBC,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b); BC!(a.u,a.uBC,a.exitBC,a.perdir)

    push!(a.Δt,min(MPCFL(a,c),1.2δt))
end

# Forcing with the unit of ρu instead of u
function MPFForcing!(r,u,ρuf,Φ,f,α,n̂,fbuffer,λμ,μ,λρ,η;perdir=())
    N,D = size_u(u)
    fill!(r,0)

    # i is velocity direction (uᵢ)
    # j is face direction (differential) (∂ⱼ)
    # calculate the lower boundary for each momentum cell then use it to help the previous cell
    # Lower boundary of the I cell is the upper boundary of I-1 cell.
    for i∈1:D, j∈1:D
        tagper = (j∈perdir)
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,Val{tagper}())
        # inner cells
        @loop (Φ[I] = - viscF(i,j,I,u,f,λμ,μ,λρ);
                r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,Val{tagper}())
    end

    surfTen!(r,f,α,n̂,fbuffer,η;perdir)
end

# Viscous forcing overload
@inline viscF(i,j,I,u,f,λμ,μ::Number,λρ) = (i==j ? getμCell(i,j,I,f,λμ,μ,λρ) : getμEdge(i,j,I,f,λμ,μ,λρ)) *(∂(j,CI(I,i),u)+∂(i,CI(I,j),u))
@inline viscF(i,j,I,u,f,λμ,μ::Nothing,λρ) = zero(eltype(f))

# Neumann BC Building block
lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{false}) = @loop r[I,i] += - viscF(i,j,I,u,f,λμ,μ,λρ) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{false}) = @loop r[I-δ(j,I),i] += viscF(i,j,I,u,f,λμ,μ,λρ) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{true}) = @loop (
    Φ[I] = -viscF(i,j,I,u,f,λμ,μ,λρ); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)

advectfq!(a::Flow{D}, c::cVOF, f=c.f, u¹=a.u⁰, u²=a.u, u⁰=a.u, dt=a.Δt[end]) where {D} = advectVOFρuu!(
    f, c.fᶠ, c.α, c.n̂, u¹, u², dt, c.c̄,
    c.ρu, a.f, a.σ, c.ρuf, c.n̂, u⁰, c.α, c.dρ, c.λρ, a.uBC;
    perdir=a.perdir, exitBC=a.exitBC,
    # dirO=1:D
    # dirO=shuffle(1:D)
    dirO=ntuple(i->mod(length(a.Δt)+i,D)+1, D)
)

function advectVOFρuu!(
    f::AbstractArray{T,D},fᶠ,α,n̂,u,u⁰,Δt,c̄,
    ρu, r, Φ, ρuf, uStar, uOld, dilaU, dρ, λρ, uBC; 
    perdir=(),exitBC=false, dirO=shuffle(1:D)) where {T,D}
    tol = 10eps(T)

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

        # uStar is c.n̂ which will be overwritten in advecVOF so better to be another vector field first.
        ρu2u!(r,ρu,f,λρ); BC!(r,uBC,exitBC,perdir)

        copyto!(Φ, f) # store old volume fraction
        # advect VOF field in d direction
        fill!(ρuf, 0)
        advectVOF1d!(f,fᶠ,α,n̂,u,u⁰,δt,c̄,ρuf,λρ,d; perdir, tol)

        # advect uᵢ in d direction
        f2face!(dρ, Φ; perdir) # fold
        copyto!(uStar,r)
        rmul!(ρuf, inv(δt)); BC!(ρuf,uBC,exitBC,perdir)
        advectρuu1D!(ρu, r, Φ, ρuf, uStar, uOld, dρ, dilaU, u, u⁰, c̄, λρ, d, δt; perdir)
    end
end

function advectρuu1D!(ρu, r, Φ, ρuf, uStar, uOld, fOld, ρ̄∂ⱼuⱼ, u, u⁰, c̄, λρ, d, δt; perdir=())
    N,D = size_u(u)
    fill!(r,0)
    j = d
    @loop ρ̄∂ⱼuⱼ[I] = getρ(I,c̄,λρ)*(∂(d,I,u)+∂(d,I,u⁰))/2 over I∈inside(Φ)
    BCf!(ρ̄∂ⱼuⱼ;perdir)
    for i∈1:D
        tagper = (j∈perdir)
        # treatment for bottom boundary with BCs
        lowerBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,Val{tagper}())
        # inner cells
        @loop (Φ[I] = ϕu(j,i,CI(I,i),ϕ(i,CI(I,j),ρuf),u,uStar,ρuf,fOld,δt,λρ);
                r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,Val{tagper}())

        @loop r[I,i] += uOld[I,i] * ϕ(i,I,ρ̄∂ⱼuⱼ) over I ∈ inside(Φ)
    end
    axpy!(δt, r, ρu)
end

# Neumann BC Building block
lowerBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{false}) = @loop r[I,i] += ϕuL(j,i,CI(I,i),ϕ(i,CI(I,j),ρuf),u,uStar,ρuf,fOld,δt,λρ) over I ∈ slice(N,2,j,2)
upperBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,i,CI(I,i),ϕ(i,CI(I,j),ρuf),u,uStar,ρuf,fOld,δt,λρ) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,i,CIj(j,CI(I,i),N[j]-2),CI(I,i),ϕ(i,CI(I,j),ρuf),u,uStar,ρuf,fOld,δt,λρ); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)


function updateU!(u,ρu,ρu⁰,forcing,dt,f,λρ,tNow,g,uBC,w=1)
    a = inv(w)-1
    inv1a = inv(1+a)
    @loop ρu[Ii] = (a*ρu⁰[Ii] + ρu[Ii] + forcing[Ii]*dt)*inv1a over Ii∈CartesianIndices(ρu)
    ρu2u!(u,ρu,f,λρ)
    fill!(forcing,0)
    accelerate!(forcing,tNow,g,uBC)
    axpy!(dt*w, forcing, u)
end

function updateL!(μ₀,f::AbstractArray{T,D},λρ;perdir=()) where {T,D}
    for d∈1:D
        @loop μ₀[I,d] /= getρ(d,I,f,λρ) over I∈inside(f)
    end
    BC!(μ₀,zeros(SVector{D,T}),false,perdir)
end

# NOTE: Do not use @fastmath for CFL. It has problem dealing with maximum function in GPU.
@inline function MPCFL(a::Flow{D,T},c::cVOF; Δt_max=one(T),safetyMargin=T(0.8)) where {D,T}
    timeNow = sum(a.Δt)
    fill!(a.σ,0)

    # From WaterLily
    @inside a.σ[I] = flux_out(I,a.u)
    Δt_Adv = inv(maximum(a.σ)+5a.ν)

    @inside a.σ[I] = maxTotalFlux(I,a.u)
    Δt_cVOF = 1/2maximum(a.σ)
    Δt_Grav = isnothing(a.g) ? Δt_max : 1/(2*√sum(i->a.g(i,zeros(SVector{D,T}),timeNow)^2, 1:D))
    Δt_Visc = isnothing(c.μ) ? Δt_max : 3/(14*c.μ*max(1,c.λμ/c.λρ))
    Δt_SurfT = isnothing(c.η) ? Δt_max : sqrt((1+c.λρ)/(8π*c.η))  # 8 from kelli's code

    return safetyMargin*min(Δt_cVOF,Δt_Adv,Δt_Grav,Δt_Visc,Δt_SurfT)
end

@fastmath @inline function maxTotalFlux(I::CartesianIndex{D},u) where D
    s = zero(eltype(u))
    for i∈1:D
        s += max(abs(u[I,i]),abs(u[I+δ(i,I),i]))
    end
    return s
end


function psolver!(p::Poisson{T};log=false,tol=50eps(T),itmx=6e3) where T
    perBC!(p.x,p.perdir)
    residual!(p); r₂ = L₂(p)
    nᵖ=0
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    insideI = inside(x)
    rho = r ⋅ z
    @log @sprintf(", %4d, %10.4e, %10.4e\n", nᵖ, L∞(p), r₂)
    while (r₂>tol || (r₂>tol/4 && nᵖ==0)) && nᵖ<itmx
        # abs(rho)<10eps(eltype(z)) && break
        perBC!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)
        alpha = rho/(@view(z[insideI]) ⋅ @view(ϵ[insideI])) 
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        @inside z[I] = r[I]*p.iD[I]
        rho2 = r⋅z
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
        r₂ = L₂(p)
        nᵖ+=1
        @log @sprintf(", %4d, %10.4e, %10.4e\n", nᵖ, L∞(p), r₂)
    end
    perBC!(p.x,p.perdir)
end

function myproject!(a::Flow{n,T},b::AbstractPoisson,w=1) where {n,T}
    dt = w*a.Δt[end]
    inproject!(a,b,dt)
    for i ∈ 1:n  # apply solution and unscale to recover pressure
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    rmul!(b.x, inv(dt))
end

@inline function inproject!(a::Flow{n,T},b::Poisson,dt) where {n,T}
    fill!(b.z,0); fill!(b.ϵ,0); fill!(b.r, 0)
    @inside b.z[I] = div(I,a.u); rmul!(b.x, dt) # set source term & solution IC
    psolver!(b;tol=50eps(T),itmx=2000)
end

@inline function inproject!(a::Flow{n,T},b::MultiLevelPoisson,dt) where {n,T}
    # fill!(b.z,0); fill!(b.ϵ,0); fill!(b.r, 0)
    @inside b.z[I] = div(I,a.u); rmul!(b.x, dt) # set source term & solution IC
    solver!(b;tol=50eps(T),itmx=200)
end

import WaterLily: increment!
smooth!(p) = GaussSeidelRB!(p;it=32)
# smooth!(p) = WaterLily.pcg!(p)
function solver!(ml::MultiLevelPoisson;tol=1e-4,itmx=32)
    p = ml.levels[1]
    residual!(p); r₂ = L₂(p)
    nᵖ=0; @log ", $nᵖ, $(L∞(p)), $r₂\n"
    while nᵖ<itmx
        Vcycle!(ml)
        smooth!(p);
        r₂ = L₂(p); 
        nᵖ+=1
        @log ", $nᵖ, $(L∞(p)), $r₂\n"
        r₂<tol && break
    end
    perBC!(p.x,p.perdir)
    push!(ml.n,nᵖ);
    # println("$(nᵖ), $(r₂)")
end

import WaterLily: Jacobi!, restrict!,prolongate!
function Vcycle!(ml::MultiLevelPoisson;l=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    # set up coarse level
    Jacobi!(fine)
    restrict!(coarse.r,fine.r)
    fill!(coarse.x,0.)
    # solve coarse (with recursion if possible)
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1)
    smooth!(coarse)
    # correct fine
    prolongate!(fine.ϵ,coarse.x)
    increment!(fine)
end

"""
    GaussSeidelRB!(p::Poisson; it=6)

Gauss-Seidel Red-Black smoother run `it` times. 
Note: This runs for general backends, but `@loop`s over `inside(p.x)` twice.
A `@vecloop` over `odds` & `evens` would reduce work at the cost of a look-up.
"""
function GaussSeidelRB!(p; it=6)
    gauss(I,x,r,L,D,iD,flag) = sum(I.I)%2==flag && (x[I] += (r[I]-mult(I,L,D,x))*iD[I])
    @inside p.ϵ[I] = p.r[I]*p.iD[I]  # initialize ϵ
    for _ in 1:it
        perBC!(p.ϵ,p.perdir)
        @loop gauss(I,p.ϵ,p.r,p.L,p.D,p.iD,0) over I ∈ inside(p.x) # "red"
        @loop gauss(I,p.ϵ,p.r,p.L,p.D,p.iD,1) over I ∈ inside(p.x) # "black"
    end
    increment!(p) # increment solution and residual
    # println("Hi")
end