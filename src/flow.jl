import WaterLily: accelerate!, median, update!, project!, BCTuple, scale_u!, exitBC!,perBC!,residual!,mult, flux_out, vanLeer, L∞
import LinearAlgebra: ⋅

@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])/2
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

@inline limiter(u,c,d) = trueKoren(u,c,d)

@inline ϕu(a,I,f,u,λ=limiter) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuP(a,Ip,I,f,u,λ=limiter) = @inbounds u>0 ? u*λ(f[Ip],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuL(a,I,f,u,λ=limiter) = @inbounds u>0 ? u*ϕ(a,I,f) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuR(a,I,f,u,λ=limiter) = @inbounds u<0 ? u*ϕ(a,I,f) : u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])


@fastmath function MPFMomStep!(a::Flow{D,T}, b::AbstractPoisson, c::cVOF, d::AbstractBody;δt = a.Δt[end]) where {D,T}
    a.u⁰ .= a.u; c.f⁰ .= c.f
    # TODO: check if BC doable for ρu

    # predictor u(n) → u(n+1/2∘) with u(n)
    @log "p"
    dtCoeff = T(1/2)
    dtList = @view(a.Δt[1:end-1])

    U = BCTuple(a.U,dtList,D)
    u2ρu!(c.ρu,a.u⁰,c.f⁰,c.λρ); BC!(c.ρu,U,a.exitBC,a.perdir)
    advectfq!(a, c, U, c.f⁰, a.u⁰, a.u, a.u, a.Δt[end])

    # TODO: include measure
    a.μ₀ .= 1
    @. c.f⁰ = (c.f⁰+c.f)/2
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f⁰,c.α,c.n̂,c.fᶠ,c.λμ,c.μ,c.λρ,c.η;perdir=a.perdir)
    u2ρu!(c.n̂,a.u⁰,c.f,c.λρ) # steal n̂ as original momentum
    updateU!(a.u,c.ρu,c.n̂,a.f,δt,c.f⁰,c.λρ,dtList,a.g,a.U,dtCoeff); BC!(a.u,U,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f⁰,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b,dtCoeff); BC!(a.u,U,a.exitBC,a.perdir)

    # c.f .= c.f⁰
    # a.u .= a.u⁰

    # corrector u(n) → u(n+1) with u(n+1/2∘)
    @log "c"
    c.f⁰ .= c.f

    U = BCTuple(a.U,a.Δt,D)
    u2ρu!(c.ρu,a.u⁰,c.f,c.λρ); BC!(c.ρu,U,a.exitBC,a.perdir)
    advectfq!(a, c, U, c.f, a.u, a.u, a.u⁰, a.Δt[end])
    
    # TODO: include measure
    a.μ₀ .= 1
    # TODO: viscous term and surface tension term should be evaluated 
    # at the end of time step to avoid divide by wrong ρ
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f,c.α,c.n̂,c.fᶠ,c.λμ,c.μ,c.λρ,c.η;perdir=a.perdir) 
    u2ρu!(c.n̂,a.u⁰,c.f,c.λρ) # steal n̂ as original momentum
    updateU!(a.u,c.ρu,c.n̂,a.f,δt,c.f,c.λρ,a.Δt,a.g,a.U); BC!(a.u,U,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b); BC!(a.u,U,a.exitBC,a.perdir)

    push!(a.Δt,min(MPCFL(a,c),1.2a.Δt[end]))
end

# Forcing with the unit of ρu instead of u
function MPFForcing!(r,u,ρuf,Φ,f,α,n̂,fbuffer,λμ,μ,λρ,η;perdir=())
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

advectfq!(a::Flow{D}, c::cVOF, U, f=c.f, u¹=a.u⁰, u²=a.u, u⁰=a.u, dt=a.Δt[end]) where {D} = advectVOFρuu!(
    f, c.fᶠ, c.α, c.n̂, u¹, u², dt, c.c̄,
    c.ρu, a.f, a.σ, c.ρuf, c.n̂, u⁰, c.α, c.λρ, U;
    perdir=a.perdir, exitBC=a.exitBC
)

function advectVOFρuu!(
    f::AbstractArray{T,D},fᶠ,α,n̂,u,u⁰,Δt,c̄,
    ρu, r, Φ, ρuf, uStar, uOld, dilaU, λρ, U; 
    perdir=(),exitBC=false) where {T,D}
    tol = 10eps(T)

    # get for dilation term
    @loop c̄[I] = ifelse(f[I]<0.5,0,1) over I ∈ CartesianIndices(f)

    dirOrder = shuffle(1:D)

    # Operator splitting to avoid bias
    # Reference for splitting method: http://www.othmar-koch.org/splitting/index.php

    # Second-order Auzinger-Ketcheson
    s2 = 1/√2
    OpOrder = D==2 ? SVector{4,Int8}(1, 2, 1, 2) : SVector{6,Int8}(1, 2, 3, 2, 3, 1)
    OpCoeff = D==2 ? SVector{4,T}(1-s2, s2, s2, 1-s2) : SVector{6,T}(1/2, 1-s2, s2, s2, 1-s2, 1/2)

    # Second-order Strang
    # OpOrder = D==2 ? SVector{3,Int8}(1, 2, 1) : SVector{5,Int8}(1, 2, 3, 2, 1)
    # OpCoeff = D==2 ? SVector{3,T}(1/2, 1, 1/2) : SVector{5,T}(1/2, 1/2, 1, 1/2, 1/2)

    # First-order Lie-Trotter
    # OpOrder = D==2 ? SVector{2,Int8}(1, 2) : SVector{3,Int8}(1, 2, 3)
    # OpCoeff = D==2 ? SVector{2,T}(1, 1) : SVector{3,T}(1, 1, 1)

    for iOp∈eachindex(OpOrder)
        d = dirOrder[OpOrder[iOp]]
        δt = OpCoeff[iOp]*Δt

        # uStar is c.n̂ which will be overwritten in advecVOF so better to be another vector field first.
        ρu2u!(r,ρu,f,λρ); BC!(r,U,exitBC,perdir)

        # advect VOF field in d direction
        ρuf .= 0
        advectVOF1d!(f,fᶠ,α,n̂,u,u⁰,δt,c̄,ρuf,λρ,d; perdir, tol)

        # advect uᵢ in d direction
        uStar .= r
        ρuf ./= δt; BC!(ρuf,U,exitBC,perdir)
        advectρuu1D!(ρu, r, Φ, ρuf, uStar, uOld, dilaU, u, u⁰, c̄, λρ, d, δt; perdir)
    end
end

function advectρuu1D!(ρu, r, Φ, ρuf, uStar, uOld, dilaU, u, u⁰, c̄, λρ, d, δt; perdir=())
    N,D = size_u(u)
    r .= 0
    j = d
    @loop dilaU[I] = (∂(d,I,u)+∂(d,I,u⁰))/2 over I∈inside(Φ)
    BCf!(dilaU;perdir)
    for i∈1:D
        tagper = (j∈perdir)
        # treatment for bottom boundary with BCs
        lowerBoundaryρuu!(r,uStar,ρuf,Φ,i,j,N,Val{tagper}())
        # inner cells
        @loop (Φ[I] = ϕu(j,CI(I,i),uStar,ϕ(i,CI(I,j),ρuf));
                r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundaryρuu!(r,uStar,ρuf,Φ,i,j,N,Val{tagper}())

        @loop r[I,i] += uOld[I,i] * (getρ(I,c̄,λρ)*dilaU[I] + getρ(I-δ(i,I),c̄,λρ)*dilaU[I-δ(i,I)])/2 over I ∈ inside(Φ)
    end
    @loop ρu[Ii] += r[Ii]*δt over Ii∈CartesianIndices(ρu)
end

# Neumann BC Building block
lowerBoundaryρuu!(r,u,ρuf,Φ,i,j,N,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) over I ∈ slice(N,2,j,2)
upperBoundaryρuu!(r,u,ρuf,Φ,i,j,N,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundaryρuu!(r,u,ρuf,Φ,i,j,N,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),ρuf)); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundaryρuu!(r,u,ρuf,Φ,i,j,N,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)


function updateU!(u,ρu,ρu⁰,forcing,dt,f,λρ,ΔtList,g,U,w=1)
    a = 1/w-1
    @loop ρu[Ii] = (a*ρu⁰[Ii] + ρu[Ii] + forcing[Ii]*dt)/(1+a) over Ii∈CartesianIndices(ρu)
    ρu2u!(u,ρu,f,λρ)
    forcing .= 0
    accelerate!(forcing,ΔtList,g,U)
    @loop u[Ii] += forcing[Ii]*dt*w over Ii∈CartesianIndices(u)
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
    a.σ .= zero(T)

    # From WaterLily
    @inside a.σ[I] = flux_out(I,a.u)
    Δt_Adv = inv(maximum(a.σ)+5a.ν)

    @inside a.σ[I] = maxTotalFlux(I,a.u)
    Δt_cVOF = 1/2maximum(a.σ)
    Δt_Grav = isnothing(a.g) ? Δt_max : 1/(2*√sum(i->a.g(i,timeNow)^2, 1:D))
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
    b.x ./= dt
end

@inline function inproject!(a::Flow{n,T},b::Poisson,dt) where {n,T}
    b.z .= 0; b.ϵ .= 0; b.r .= 0
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    psolver!(b;tol=sqrt(eps(T))/30,itmx=750)
end

@inline function inproject!(a::Flow{n,T},b::MultiLevelPoisson,dt) where {n,T}
    b.z .= 0; b.r .= 0
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    solver!(b;tol=10000eps(T),itmx=1e3)
end