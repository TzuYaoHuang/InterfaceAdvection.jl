import WaterLily: accelerate!, median, update!, project!, scale_u!, exitBC!,perBC!,residual!,mult, flux_out, vanLeer, L∞, ϕ
using LinearAlgebra
import LinearAlgebra: ⋅
import BiotSavartBCs: @vecloop

backend_sync!(::Any) = nothing

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
    a.u⁰ .= a.u; c.f⁰ .= c.f
    t₁ = sum(a.Δt); t₀ = t₁-δt; tₘ = t₁-δt/2
    # TODO: check if BC doable for ρu

    # predictor u(n) → u(n+1/2∘) with u(n)
    @log "p"
    dtCoeff = T(1/2)

    NVTX.@range "u2ρu!" begin
        u2ρu!(c.ρu,a.u⁰,c.f⁰,c.λρ); 
    end
    BC!(c.ρu,a.uBC,a.exitBC,a.perdir)
    advectfq!(a, c, c.f⁰, a.u⁰, a.u, a.u, δt)

    # TODO: include measure
    a.μ₀ .= 1
    @. c.f⁰ = (c.f⁰+c.f)/2
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f⁰,c.α,c.n̂,c.fᶠ,c.λμ,c.μ,c.λρ,c.η;perdir=a.perdir)
    u2ρu!(c.n̂,a.u⁰,c.f,c.λρ) # steal n̂ as original momentum
    updateU!(a.u,c.ρu,c.n̂,a.f,δt,c.f⁰,c.λρ,tₘ,a.g,a.uBC,dtCoeff); BC!(a.u,a.uBC,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f⁰,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b,dtCoeff); BC!(a.u,a.uBC,a.exitBC,a.perdir)

    # c.f .= c.f⁰
    # a.u .= a.u⁰

    # corrector u(n) → u(n+1) with u(n+1/2∘)
    @log "c"
    c.f⁰ .= c.f

    NVTX.@range "u2ρu!" begin
        u2ρu!(c.ρu,a.u⁰,c.f,c.λρ)
    end
    BC!(c.ρu,a.uBC,a.exitBC,a.perdir)
    advectfq!(a, c, c.f, a.u, a.u, a.u⁰, δt)
    
    # TODO: include measure
    a.μ₀ .= 1
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
NVTX.@annotate function MPFForcing!(r,u,ρuf,Φ,f,α,n̂,fbuffer,λμ,μ,λρ,η;perdir=())
    N,D = size_u(u)
    r .= 0

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
    backend_sync!(f)
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


NVTX.@annotate function advectVOFρuu!(
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
        ρu2u!(r,ρu,f,λρ); 

        NVTX.@range "BC u" begin
            BC!(r,uBC,exitBC,perdir)
            backend_sync!(r)
        end

        NVTX.@range "Φ .= f" begin
            copyto!(Φ, f)  # store old volume fraction
            backend_sync!(r)
        end
        
        # advect VOF field in d direction
        NVTX.@range "ρuf .= 0" begin
            fill!(ρuf, 0)  # store old volume fraction
            backend_sync!(r)
        end
        advectVOF1d!(f,fᶠ,α,n̂,u,u⁰,δt,c̄,ρuf,λρ,d; perdir, tol)

        # advect uᵢ in d direction
        f2face!(dρ, Φ; perdir) # fold
        NVTX.@range "uStar .= r" begin
            copyto!(uStar, r)  # store old volume fraction
            backend_sync!(r)
        end
        NVTX.@range "ρuf ./= δt;" begin
            rmul!(ρuf, inv(δt));  # store old volume fraction
            backend_sync!(r)
        end
        NVTX.@range "BC ruf" begin
            BC!(ρuf,uBC,exitBC,perdir)
            backend_sync!(r)
        end
        
        advectρuu1D!(ρu, r, Φ, ρuf, uStar, uOld, dρ, dilaU, u, u⁰, c̄, λρ, d, δt; perdir)
    end
end

NVTX.@annotate function advectρuu1D!(ρu, r, Φ, ρuf, uStar, uOld, fOld, ρ̄∂ⱼuⱼ, u, u⁰, c̄, λρ, d, δt; perdir=())
    N,D = size_u(u)
    NVTX.@range "r .= 0" begin
    fill!(r,0)
    backend_sync!(r)
    end
    
    j = d
    NVTX.@range "ρ̄∂ⱼuⱼ" begin
    @loop ρ̄∂ⱼuⱼ[I] = getρ(I,c̄,λρ)*(∂(d,I,u)+∂(d,I,u⁰))/2 over I∈inside(Φ)
    BCf!(ρ̄∂ⱼuⱼ;perdir)
    backend_sync!(r)
    end
    
    
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
    backend_sync!(r)
    NVTX.@range "ru += r*dt" begin
        axpy!(δt,r,ρu)
    # @loop ρu[Ii] += r[Ii]*δt over Ii∈CartesianIndices(ρu)
    backend_sync!(r)
    end
end

# Neumann BC Building block
lowerBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{false}) = @loop r[I,i] += ϕuL(j,i,CI(I,i),ϕ(i,CI(I,j),ρuf),u,uStar,ρuf,fOld,δt,λρ) over I ∈ slice(N,2,j,2)
upperBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,i,CI(I,i),ϕ(i,CI(I,j),ρuf),u,uStar,ρuf,fOld,δt,λρ) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,i,CIj(j,CI(I,i),N[j]-2),CI(I,i),ϕ(i,CI(I,j),ρuf),u,uStar,ρuf,fOld,δt,λρ); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundaryρuu!(r,u,uStar,ρuf,Φ,fOld,δt,λρ,i,j,N,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)


NVTX.@annotate function updateU!(u,ρu,ρu⁰,forcing,dt,f,λρ,tNow,g,uBC,w=1)
    a = 1/w-1
    @loop ρu[Ii] = (a*ρu⁰[Ii] + ρu[Ii] + forcing[Ii]*dt)/(1+a) over Ii∈CartesianIndices(ρu)
    ρu2u!(u,ρu,f,λρ)
    forcing .= 0
    accelerate!(forcing,tNow,g,uBC)
    @loop u[Ii] += forcing[Ii]*dt*w over Ii∈CartesianIndices(u)
end

NVTX.@annotate function updateL!(μ₀,f::AbstractArray{T,D},λρ;perdir=()) where {T,D}
    for d∈1:D
        @loop μ₀[I,d] /= getρ(d,I,f,λρ) over I∈inside(f)
    end
    BC!(μ₀,zeros(SVector{D,T}),false,perdir)
end

# NOTE: Do not use @fastmath for CFL. It has problem dealing with maximum function in GPU.
@inline NVTX.@annotate function MPCFL(a::Flow{D,T},c::cVOF; Δt_max=one(T),safetyMargin=T(0.8)) where {D,T}
    timeNow = sum(a.Δt)
    a.σ .= zero(T)

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

NVTX.@annotate function myproject!(a::Flow{n,T},b::AbstractPoisson,w=1) where {n,T}
    dt = w*a.Δt[end]
    inproject!(a,b,dt)
    for i ∈ 1:n  # apply solution and unscale to recover pressure
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    b.x ./= dt
end

@inline NVTX.@annotate function inproject!(a::Flow{n,T},b::Poisson,dt) where {n,T}
    b.z .= 0; b.ϵ .= 0; b.r .= 0
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    psolver!(b;tol=50eps(T),itmx=2000)
end

@inline NVTX.@annotate function inproject!(a::Flow{n,T},b::MultiLevelPoisson,dt) where {n,T}
    # b.z .= 0; b.r .= 0
    @inside b.z[I] = div(I,a.u); 
    backend_sync!(a.u)
    NVTX.@range "scalep" begin 
        # b.x .*= dt
        rmul!(b.x,dt) 
        backend_sync!(a.u)
    end  # set source term & solution IC
    solver!(b;tol=50eps(T),itmx=4)
end

NVTX.@annotate function increment!(p::Poisson)
    perBC!(p.ϵ,p.perdir)
    NVTX.@range "sync_afterperBC!" begin backend_sync!(p.x) end
    @loop (p.r[I] = p.r[I]-mult(I,p.L,p.D,p.ϵ);
           p.x[I] = p.x[I]+p.ϵ[I]) over I ∈ inside(p.x)
    NVTX.@range "sync_afterrx!" begin backend_sync!(p.x) end
end
smooth!(p) = GaussSeidelRB!(p;it=6)
# smooth!(p) = pcg!(p) 
# smooth!(p) = WaterLily.Jacobi!(p;it=6)
NVTX.@annotate function solver!(ml::MultiLevelPoisson;tol=1e-4,itmx=32)
    p = ml.levels[1]
    NVTX.@range "residual!" begin residual!(p);  end
    NVTX.@range "sync_before_L₂(init)" begin backend_sync!(p.x) end
    NVTX.@range "L₂" begin r₂ = L₂(p) end
    nᵖ=0; @log ", $nᵖ, $(L∞(p)), $r₂\n"
    while nᵖ<itmx
        NVTX.@range "Vcycle!" begin Vcycle!(ml) end
        NVTX.@range "smooth!" begin smooth!(p) end
        # NVTX.@range "sync_before_L₂" begin backend_sync!(p.x) end
        NVTX.@range "r₂ =" begin r₂ = NVTX.@range "L₂" begin L₂(p) end end
        nᵖ+=1
        @log ", $nᵖ, $(L∞(p)), $r₂\n"
        # r₂<tol && break
    end
    NVTX.@range "perBC!" begin perBC!(p.x,p.perdir) end
    push!(ml.n,nᵖ);
    # println("$(nᵖ), $(r₂)")
end

import WaterLily: Jacobi!, restrict!,prolongate!
NVTX.@annotate function Vcycle!(ml::MultiLevelPoisson;l=1)
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
gauss(x,r,L,D,iD,I,flag) = sum(I.I)%2==flag && (x[I] += (r[I]-mult(I,L,D,x))*iD[I])
gauss(x,r,L,D,iD,I) = (x[I] += (r[I]-mult(I,L,D,x))*iD[I])

@fastmath @inline function mult_LU(I::CartesianIndex{d},L,x) where {d}
    s = zero(eltype(x))
    for i in 1:d
        s += @inbounds(x[I-δ(i,I)]*L[I,i] + x[I+δ(i,I)]*L[I+δ(i,I),i])
    end
    return s
end

@fastmath @inline function gauss(I::CartesianIndex{d},r,L,iD,x) where {d}
    s = @inbounds(r[I])
    for i in 1:d
        s -= @inbounds(x[I-δ(i,I)]*L[I,i] + x[I+δ(i,I)]*L[I+δ(i,I),i])
    end
    return s*@inbounds(iD[I])
end

function gauss_rb(x,r,L,D,iD,color,Iv)
    k = 2*Iv.I[end] - (sum(Iv.I[1:end-1]) + color) % 2
    I = CartesianIndex(Iv.I[1:end-1]..., k)+oneunit(Iv)
    x[I] = gauss(I,r,L,iD,x)
end

function half_rangez(x::AbstractArray{T,N}) where{T,N}
    Nin = size(x) .- 2
    return CartesianIndices(ntuple((i) -> ifelse(i==N,1:Nin[i]÷2,1:Nin[i]), N))
end
NVTX.@annotate function GaussSeidelRB!(p; it=6)
    @inside p.ϵ[I] = p.r[I]*p.iD[I]  # initialize ϵ

    half_range = half_rangez(p.ϵ)
    for _ in 1:it
        NVTX.@range "perBC!" begin perBC!(p.ϵ,p.perdir); backend_sync!(p.ϵ) end
        # NOTE: Put sync insdie perBC and check if there is raise condition
        # Check it that is also the case in PCG.
        NVTX.@range "RB" begin 
            @loop gauss_rb(p.ϵ,p.r,p.L,p.D,p.iD,0,I) over I ∈ half_range  # red
            @loop gauss_rb(p.ϵ,p.r,p.L,p.D,p.iD,1,I) over I ∈ half_range  # black
            backend_sync!(p.ϵ) 
        end
    end
    increment!(p) # increment solution and residual
    # println("Hi")
end

NVTX.@annotate function pcg!(p::Poisson{T};it=6) where T
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = r⋅z
    # abs(rho)<10eps(T) && return
    for i in 1:it
        NVTX.@range "perBC!" begin perBC!(ϵ,p.perdir); backend_sync!(ϵ) end
        NVTX.@range "z=mult" begin 
            @inside z[I] = mult(I,p.L,p.D,ϵ) # get value will be slow
            backend_sync!(ϵ)
        end
        NVTX.@range "alpha" begin 
            alpha = rho/(z⋅ϵ)
        end
        # (abs(alpha)<1e-2 || abs(alpha)>1e2) && return # alpha should be O(1)
        NVTX.@range "xrIncrement" begin 
            @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        end
        i==it && return
        NVTX.@range "z=r*iD" begin 
            @inside z[I] = r[I]*p.iD[I]
            backend_sync!(ϵ)
        end
        NVTX.@range "rho2" begin 
            rho2 = r⋅z
        end
        # abs(rho2)<10eps(T) && return
        beta = rho2/rho
        NVTX.@range "eps=beta*eps+I" begin 
            @inside ϵ[I] = beta*ϵ[I]+z[I]
        end
        rho = rho2
    end
end