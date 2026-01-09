import WaterLily: accelerate!, median, update!, project!, scale_u!, exitBC!,perBC!,residual!,mult, flux_out, vanLeer, L∞
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
    U = BCTuple(a.uBC,dtList,D)
    u2ρu!(c.ρu,a.u⁰,c.f⁰,c.λρ); BC!(c.ρu,U,a.exitBC,a.perdir)
    advect!(a,c,c.f⁰,a.u⁰,a.u); c.ρuf ./= δt; BC!(c.ρuf,U,a.exitBC,a.perdir)
    # TODO: include measure
    a.μ₀ .= 1
    @. c.f⁰ = (c.f⁰+c.f)/2
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f⁰,c.α,c.n̂,c.fᶠ,c.λμ,c.μ,c.λρ,c.η;perdir=a.perdir)
    updateU!(a.u,c.ρu,a.f,δt,c.f⁰,c.λρ,dtList,a.g,a.uBC,dtCoeff); BC!(a.u,U,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f⁰,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b,dtCoeff); BC!(a.u,U,a.exitBC,a.perdir)

    # c.f .= c.f⁰
    # a.u .= a.u⁰

    # corrector u(n) → u(n+1) with u(n+1/2∘)
    @log "c"
    U = BCTuple(a.uBC,a.Δt,D)
    # recover ρu @ t = n since it is modified for the predictor step
    u2ρu!(c.ρu,a.u⁰,c.f,c.λρ); BC!(c.ρu,U,a.exitBC,a.perdir)
    advect!(a,c,c.f,a.u,a.u); c.ρuf ./= δt; BC!(c.ρuf,U,a.exitBC,a.perdir)
    # TODO: include measure
    a.μ₀ .= 1
    # TODO: viscous term and surface tension term should be evaluated 
    # at the end of time step to avoid divide by wrong ρ
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f,c.α,c.n̂,c.fᶠ,c.λμ,c.μ,c.λρ,c.η;perdir=a.perdir) 
    updateU!(a.u,c.ρu,a.f,δt,c.f,c.λρ,a.Δt,a.g,a.uBC); BC!(a.u,U,a.exitBC,a.perdir)
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
        @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) - viscF(i,j,I,u,f,λμ,μ,λρ);
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
lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) - viscF(i,j,I,u,f,λμ,μ,λρ) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) + viscF(i,j,I,u,f,λμ,μ,λρ) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),ρuf)) - viscF(i,j,I,u,f,λμ,μ,λρ); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,λρ,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)


function updateU!(u,ρu,forcing,dt,f,λρ,ΔtList,g,U,w=1)
    @loop ρu[Ii] += forcing[Ii]*dt*w over Ii∈CartesianIndices(ρu)
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

# TODO: Still need to converge to the WaterLily method. This is only temporary approach.
BCTuple(f::Function,dt,N,t=sum(dt))=ntuple(i->f(i,t),N)
BCTuple(f::Tuple,dt,N)=f