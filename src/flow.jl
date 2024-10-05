import WaterLily: accelerate!, median, update!, project!, BCTuple, scale_u!, exitBC!,perBC!,residual!,mult, flux_out
import LinearAlgebra: ⋅


# I need to re-define the flux limiter or else the TVD property cannot conserve
@fastmath koren(u,c,d) = median((5c+2d-u)/6,c,median(2c-1u,c,d))
@fastmath cen(u,c,d) = 0.5*(c+d)
@inline ϕu(a,I,f,u,λ=koren) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuP(a,Ip,I,f,u,λ=koren) = @inbounds u>0 ? u*λ(f[Ip],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuL(a,I,f,u,λ=koren) = @inbounds u>0 ? u*ϕ(a,I,f) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuR(a,I,f,u,λ=koren) = @inbounds u<0 ? u*ϕ(a,I,f) : u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])


@fastmath function MPFMomStep!(a::Flow{D}, b::AbstractPoisson, c::cVOF, d::AbstractBody;δt = a.Δt[end]) where {D}
    a.u⁰ .= a.u;
    # TODO: check if BC doable for ρu

    # predictor u → u'
    U = BCTuple(a.U,@view(a.Δt[1:end-1]),D)
    u2ρu!(c.ρu,a.u,c.f⁰,c.λρ); BC!(c.ρu,U,a.exitBC,a.perdir)
    advect!(a,c,c.f,a.u⁰,a.u); c.ρuf ./= δt; BC!(c.ρuf,U,a.exitBC,a.perdir)
    # TODO: include measure
    a.μ₀ .= 1
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f⁰,c.λμ,c.μ;perdir=a.perdir)
    updateU!(δt,a.f,c.ρu,a.u,c.f,c.λρ,@view(a.Δt[1:end-1]),a.g,a.U); BC!(a.u,U,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b); BC!(a.u,U,a.exitBC,a.perdir)

    # corrector u → u¹
    U = BCTuple(a.U,a.Δt,D)
    # recover ρu @ t = n since it is modified for the predictor step
    u2ρu!(c.ρu,a.u⁰,c.f⁰,c.λρ); BC!(c.ρu,U,a.exitBC,a.perdir)
    @. c.f = (c.f+c.f⁰)/2
    advect!(a,c,c.f⁰,a.u⁰,a.u); c.ρuf ./= δt; BC!(c.ρuf,U,a.exitBC,a.perdir)
    # TODO: include measure
    a.μ₀ .= 1
    @. a.u = (a.u+a.u⁰)/2
    # TODO: think about which volume fraction should be applied for viscous and surface tneion terms
    # currently i use @. c.f = (c.f+c.f⁰)/2, should be fine for viscous flow but the sharpness of 
    # interface cannot be retain for surface tension calculation. If need to be consistent with pressure
    # solver than one should actually use c.f⁰.
    MPFForcing!(a.f,a.u,c.ρuf,a.σ,c.f,c.λμ,c.μ;perdir=a.perdir) 
    updateU!(δt,a.f,c.ρu,a.u,c.f⁰,c.λρ,a.Δt,a.g,a.U); BC!(a.u,U,a.exitBC,a.perdir)
    updateL!(a.μ₀,c.f⁰,c.λρ;perdir=a.perdir); 
    update!(b)
    myproject!(a,b); BC!(a.u,U,a.exitBC,a.perdir)

    c.f .= c.f⁰

    push!(a.Δt,min(MPCFL(a,c),1.5a.Δt[end]))
end

# Forcing with the unit of ρu instead of u
function MPFForcing!(r,u,ρuf,Φ,f,λμ,μ;perdir=())
    N,D = size_u(u)
    r .= 0.

    # i is velocity direction
    # j is face direction (differential)
    # calculate the lower boundary for each momentum cell then use it to help the previous cell
    # Lower boundary of the I cell is the upper boundary of I-1 cell.
    for i∈1:D, j∈1:D
        tagper = (j∈perdir)
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,Val{tagper}())
        # inner cells
        @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) - viscF(i,j,I,u,f,λμ,μ);
                r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,Val{tagper}())
    end

    # TODO: surface tension
end

# Viscous forcing overload
@inline viscF(i,j,I,u,f,λμ,μ::Number) = getμ(i,j,I,f,λμ,μ)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u))
@inline viscF(i,j,I,u,f,λμ,μ::Nothing) = zero(eltype(f))

# Neumann BC Building block
lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) - viscF(i,j,I,u,f,λμ,μ) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),ρuf)) + viscF(i,j,I,u,f,λμ,μ) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),ρuf)) - viscF(i,j,I,u,f,λμ,μ); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,ρuf,Φ,i,j,N,f,λμ,μ,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)


function updateU!(dt,forcing,ρu,u,f,λρ,ΔtList,g,U)
    @loop ρu[Ii] += forcing[Ii]*dt over Ii∈CartesianIndices(ρu)
    ρu2u!(u,ρu,f,λρ)
    forcing .= 0
    accelerate!(forcing,ΔtList,g,U)
    @loop u[Ii] += forcing[Ii]*dt over Ii∈CartesianIndices(u)
end

function updateL!(μ₀,f::AbstractArray{T,D},λρ;perdir=()) where {T,D}
    for d∈1:D
        @loop μ₀[I,d] /= getρ(d,I,f,λρ) over I∈inside(f)
    end
    BC!(μ₀,zeros(SVector{D,T}),false,perdir)
end

@fastmath @inline function MPCFL(a::Flow{D,T},c::cVOF; Δt_max=one(T),safetyMargin=T(0.8)) where {D,T}
    timeNow = sum(a.Δt)
    a.σ .= zero(T)

    # From WaterLily
    @inside a.σ[I] = flux_out(I,a.u)
    Δt_Adv = inv(maximum(a.σ)+5a.ν)

    @inside a.σ[I] = maxTotalFlux(I,a.u)
    Δt_cVOF = 0.5/maximum(a.σ)
    Δt_Grav = isnothing(a.g) ? Δt_max : 1/√sum(i->g(i,timeNow)^2,1:D)
    Δt_Visc = isnothing(c.μ) ? Δt_max : 3/14*1/(c.μ*max(1,c.λμ/c.λρ))
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


function psolver!(p::Poisson;log=false,tol=1e-14,itmx=6e3)
    perBC!(p.x,p.perdir)
    residual!(p); r₂ = L₂(p)
    nᵖ=0
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    insideI = inside(x) # [insideI]
    rho = r ⋅ z
    while (r₂>tol || nᵖ==0) && nᵖ<itmx
        # abs(rho)<10eps(eltype(z)) && break
        perBC!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)
        alpha = rho/(@views z[insideI]⋅ϵ[insideI])
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        @inside z[I] = r[I]*p.iD[I]
        rho2 = r⋅z
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
        r₂ = L₂(p)
        nᵖ+=1
    end
    perBC!(p.x,p.perdir)
end

function myproject!(a::Flow{n},b::AbstractPoisson,w=1) where n
    dt = w*a.Δt[end]
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    psolver!(b)
    for i ∈ 1:n  # apply solution and unscale to recover pressure
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    b.x ./= dt
end