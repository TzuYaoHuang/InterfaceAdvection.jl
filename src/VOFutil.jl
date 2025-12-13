using StaticArrays

"""
    applyVOF!(f,α,n̂,InterfaceSDF)

Calculate volume fraction, `f`, according to a given signed distance function, `InterfaceSDF`. The dark fluid is indicated with negative distance.
"""
function applyVOF!(f,α,n̂,InterfaceSDF)
    # set up the field with PLIC Calculation
    @loop applyVOF!(f,α,n̂,InterfaceSDF,I) over I∈inside(f)
    # clean wisp: value too close to 0 or 1
    cleanWisp!(f)
end
@inline function applyVOF!(f::AbstractArray{T,D},α::AbstractArray{T,D},n̂::AbstractArray{T,Dv},InterfaceSDF,I) where {T,D,Dv}
    # forwardDiff cause some problem so using finite difference
    Δd = T(0.01)
    for i∈1:D
        xyzpδ = SVector{D,T}(loc(0,I) .+Δd .*δ(i,I).I)
        xyzmδ = SVector{D,T}(loc(0,I) .-Δd .*δ(i,I).I)
        n̂[I,i] = InterfaceSDF(xyzpδ) - InterfaceSDF(xyzmδ)
    end
    sumN2 = 0; for i∈1:D sumN2+= n̂[I,i]^2 end

    # (n̂·𝐱 - α)/|n̂| = d
    α[I] = - √sumN2*InterfaceSDF(loc(0,I).-T(0.5))

    # the PLIC estimation
    f[I] = getVolumeFraction(n̂,I,α[I])
end

"""
    BCVOF!(f,α,n̂;perdir)

Apply boundary condition to volume fraction, intercept, and normal with Neumann or Periodic ways
"""
function BCVOF!(f,α,n̂;perdir=())
    N,D = size_u(n̂)
    for j∈1:D
        if j in perdir
            @loop fαn̂!(f,α,n̂, I,j,N[j]-1) over I ∈ slice(N,1,j)
            @loop fαn̂!(f,α,n̂, I,j,2) over I ∈ slice(N,N[j],j)
        else
            @loop f[I] = f[I+δ(j,I)] over I ∈ slice(N,1,j)
            @loop f[I] = f[I-δ(j,I)] over I ∈ slice(N,N[j],j)
        end
    end
end
function fαn̂!(f::AbstractArray{T,D},α,n̂, I,j,ii) where {T,D}
    f[I] = f[CIj(j,I,ii)]
    for i ∈ 1:D
        n̂[I,i] = n̂[CIj(j,I,ii),i]
    end
    α[I] = α[CIj(j,I,ii)]
end

function BCf!(f;perdir=())
    N = size(f); D = length(N)
    for j∈1:D
        if j in perdir
            @loop f[I] = f[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
            @loop f[I] = f[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
        else
            @loop f[I] = f[I+δ(j,I)] over I ∈ slice(N,1,j)
            @loop f[I] = f[I-δ(j,I)] over I ∈ slice(N,N[j],j)
        end
    end
end
function BCf!(d,f;perdir=())
    N = size(f); D = length(N)
    for j∈1:D
        if j in perdir
            @loop f[I] = f[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
            @loop f[I] = f[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
        elseif j==d
            @loop f[I] = f[I+2δ(j,I)] over I ∈ slice(N,1,j)
        else
            @loop f[I] = f[I+δ(j,I)] over I ∈ slice(N,1,j)
            @loop f[I] = f[I-δ(j,I)] over I ∈ slice(N,N[j],j)
        end
    end
end

function BCv!(f;perdir=())
    N = size(f)[1:end-1]; D = length(N)
    for d∈1:D, j∈1:D
        if j in perdir
            @loop f[I,d] = f[CIj(j,I,N[j]-1),d] over I ∈ slice(N,1,j)
            @loop f[I,d] = f[CIj(j,I,2),d] over I ∈ slice(N,N[j],j)
        elseif j==d
            @loop f[I,d] = f[I+2δ(j,I),d] over I ∈ slice(N,1,j)
        else
            @loop f[I,d] = f[I+δ(j,I),d] over I ∈ slice(N,1,j)
            @loop f[I,d] = f[I-δ(j,I),d] over I ∈ slice(N,N[j],j)
        end
    end
end

function BCv1D!(f,d;perdir=())
    N = size(f)
    D = length(N)
    for j∈1:D
        if j in perdir
            @loop f[I] = f[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
            @loop f[I] = f[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
        elseif j==d
            @loop f[I] = f[I+2δ(j,I)] over I ∈ slice(N,1,j)
        else
            @loop f[I] = f[I+δ(j,I)] over I ∈ slice(N,1,j)
            @loop f[I] = f[I-δ(j,I)] over I ∈ slice(N,N[j],j)
        end
    end
end


"""
    cleanWisp!(f; tol)

Clean out values in `f` too close to 0 or 1. The margin is 10 times the resolution of float type `T`.
"""
function cleanWisp!(f::AbstractArray{T,D}, tol=10eps(T)) where {T,D}
    @loop f[I] = ifelse(f[I]<  tol, T(0), f[I]) over I∈inside(f)
    @loop f[I] = ifelse(f[I]>1-tol, T(1), f[I]) over I∈inside(f)
end


"""
    containInterface(f)

Check whether `f` is interface cell.
"""
@inline containInterface(f) = 0<f<1

"""
    fullorempty(fc)

Check whether `fc` is full of dark or light fluid.
"""
@inline fullorempty(fc) = (fc==0 || fc==1)

"""
    get3CellHeight(f,I,summingDir)

Get three cell volume summation around index `I` along direction `summingDir`.
"""
@inline @fastmath get3CellHeight(f,I,summingDir) = f[I]+f[I-δ(summingDir,I)]+f[I+δ(summingDir,I)]

"""
    linInterpProp(f,λ,base=one(eltype(f)))

Linearly interpolate fluid properties (ρ, μ, ν, etc.) according to volume fraction `f` and the property's ratio of light to dark fluid.
The property of dark fluid is assumed to be 1, but can be specified with the third argument.
"""
@inline @fastmath linInterpProp(f,λ,base=one(eltype(f))) = base*(λ + (1-λ)*f)

using EllipsisNotation
"""
    getρ([d,]I,f,λρ)

Linearly interpolate density at either `I` or `I-0.5d`.
"""
@inline @fastmath getρ(I::CartesianIndex{D},f::AbstractArray{T,D},λρ) where {T,D} = linInterpProp(f[I],λρ)
@inline @fastmath getρ(Ii::CartesianIndex{Dv},f::AbstractArray{T,D},λρ) where {T,D,Dv} = getρ(Ii.I[end],CI(Ii.I[1:end-1]),f,λρ)
@inline @fastmath getρ(d,I,f,λρ) = linInterpProp(ϕ(d,I,f),λρ)

"""
    getμ(i,j,I,f,λμ,μ,λρ)

Calculate the viscosity corresponding to the term ∂ⱼuᵢ @ either `I-0.5i-0.5j` or `I-1i`.
The function return the linear interpolation at cell center (when `i==j`) or cell vertex (when `i≠j`).
Specify at `IJEQUAL` with `Val{i==j}()`.
The calculated viscosity is limited with the majority fluid's kinematic viscosity applied to interpolation.
The dynamic viscosity is then recovered using the minimal density of the cells who are going to use the stress flux.
"""
@inline @fastmath getμCell(i,j,I,f,λμ,μ,λρ) = μ*linInterpProp(f[I-δ(i,I)],λμ)
@inline @fastmath function getμEdge(i,j,I,f::AbstractArray{T,D},λμ,μ,λρ) where {T,D}
    f1,f2,f3,f4 = f[I],f[I-δ(i,I)],f[I-δ(i,I)-δ(j,I)],f[I-δ(j,I)]
    s = (f1+f2+f3+f4)/4
    fmin = λρ < 1 ? min(f1+f2,f2+f3,f3+f4,f4+f1)/2 : max(f1+f2,f2+f3,f3+f4,f4+f1)/2
    return μ*min(linInterpProp(s,λμ), ifelse(s>0.5,1,λμ/λρ)*linInterpProp(fmin,λρ))
end

"""
    ρu2u!(u,ρu,f,λρ[,I])

Convert mass flux `ρu` to velocity `u` at the corresponding momentum cell.
"""
ρu2u!(u,ρu,f,λρ) = @loop ρu2u!(u,ρu,f,λρ,I) over I∈inside(f)
@inline @fastmath ρu2u!(u,ρu,f::AbstractArray{T,D},λρ,I) where {T,D} = for d∈1:D
    u[I,d] = ρu[I,d]/getρ(d,I,f,λρ)
end

"""
    u2ρu!(ρu,u,f,λρ[,I])

Convert velocity `u` to mass flux `ρu` at the corresponding momentum cell.
"""
u2ρu!(ρu,u,f,λρ) = @loop u2ρu!(ρu,u,f,λρ,I) over I∈inside(f)
@inline @fastmath u2ρu!(ρu,u,f::AbstractArray{T,D},λρ,I) where {T,D} = for d∈1:D
    ρu[I,d] = u[I,d]*getρ(d,I,f,λρ)
end

"""
    fᶠ2ρuf(I,fᶠ,δl,λρ)

Convert volume flux `fᶠ` @ `I` to mash flux.
"""
@inline @fastmath fᶠ2ρuf(I,fᶠ,δl,λρ) = δl*λρ + (1-λρ)*fᶠ[I]

@fastmath getρratio!(vec, fnew::AbstractArray{T,D}, fold, λρ) where {T,D} = for d∈1:D
    @loop vec[I,d] = getρ(d,I,fnew,λρ)/getρ(d,I,fold,λρ) over I∈inside_uWB(size(fnew),d)
end

function f2face1D!(fFace::AbstractArray{T,D}, fCen, d; perdir=()) where {T,D}
    @loop fFace[I] = ϕ(d,I,fCen) over I∈inside(fCen)
    BCv1D!(fFace,d;perdir)
end

function f2face!(fFace, fCen::AbstractArray{T,D}; perdir=()) where {T,D}
    for d∈1:D
        @loop fFace[I,d] = ϕ(d,I,fCen) over I∈inside_uWB(size(fCen),d)
    end
    BCv!(fFace;perdir)
end

"""
    getInterfaceCenter(n̂,α,I)

To calculate the quasi-center of line or plane segments in cell `I` by projecting the cell center to the plane.
"""
function getInterfaceCenter(n̂::AbstractArray{T,nv},α::AbstractArray{T,n},I::CartesianIndex{n}) where{T,n,nv}
    nLocal = @views n̂[I,:]
    dis = (0.5sum(nLocal) - α[I])/√sum(abs2,nLocal)
    return -dis*nLocal/√sum(abs2,nLocal)
end