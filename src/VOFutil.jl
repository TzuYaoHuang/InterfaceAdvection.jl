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
    Δx = T(0.01)
    xcen = loc(0,I)

    sumN = zero(T)
    sumN2 = zero(T) 
    for i∈1:D
        xyzpδ = SVector{D,T}([xcen[j]+ifelse(j==i,Δx,T(0)) for j∈1:D]) 
        xyzmδ = SVector{D,T}([xcen[j]-ifelse(j==i,Δx,T(0)) for j∈1:D]) 
        Δd = InterfaceSDF(xyzpδ) - InterfaceSDF(xyzmδ)
        n̂[I,i] = Δd
        sumN += Δd
        sumN2 += Δd^2
    end

    # (n̂·(𝐱_cen-𝐱_blCorner) - α) = |n̂| d_cen
    # 𝐱_cen-𝐱_blCorner = (0.5,0.5,0.5)
    α[I] = sumN/2 - √sumN2*InterfaceSDF(xcen)

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

function BCf!(f::AbstractArray{T,D};perdir=()) where {T,D}
    N = size(f)
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
function BCf!(d,f::AbstractArray{T,D};perdir=()) where {T,D}
    N = size(f)
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
    N,D = size_u(f)
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

function BCv1D!(f::AbstractArray{T,D},d;perdir=()) where {T,D}
    N = size(f)
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
    @loop (
        f[I] = ifelse(f[I]<tol, 
            T(0), 
            ifelse(f[I]>1-tol, 
                T(1), 
                f[I])
            )
    ) over I∈inside(f)
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
@inline @fastmath get3CellHeight(f,I,summingDir) = f[I]+f[I-δd(summingDir,I)]+f[I+δd(summingDir,I)]

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
@inline @fastmath getρ(Ii::CartesianIndex{Dv},f::AbstractArray{T,D},λρ) where {T,D,Dv} = getρ(last(Ii.I),CI(Base.front(Ii.I)),f,λρ)
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

This is an approximation: the perpendicular foot of the cell center on the (infinite) PLIC plane need not
land on the actual interface segment/polygon, particularly when the cell is nearly full or empty. A 2D
method that instead returns the exact segment midpoint is provided below.
"""
function getInterfaceCenter(n̂::AbstractArray{T,nv},α::AbstractArray{T,n},I::CartesianIndex{n}) where{T,n,nv}
    nLocal = SVector{n,T}(ntuple(i->n̂[I,i],n))
    nrm = √sum(abs2,nLocal)
    dis = (0.5sum(nLocal) - α[I])/nrm
    return -dis*nLocal/nrm
end

"""
    getInterfaceCenter(n̂,α,I::CartesianIndex{2})

Exact midpoint of the PLIC line segment cut out of cell `I` (as opposed to the generic method above,
which only projects the cell center onto the infinite plane and can miss the cell entirely for short
segments). Cheap closed form: reflect to the octant where both normal components are ≥0 (same trick as
[`getIntercept`](@ref)/[`getVolumeFraction`](@ref)), locate the two edges the line crosses in that frame,
average their crossing points, then reflect back.
"""
function getInterfaceCenter(n̂::AbstractArray{T,nv},α::AbstractArray{T,2},I::CartesianIndex{2}) where{T,nv}
    n1 = n̂[I,1]; n2 = n̂[I,2]
    m1 = abs(n1); m2 = abs(n2)
    b = α[I] - min(n1,zero(T)) - min(n2,zero(T)) # intercept in the reflected (m1,m2≥0) frame

    mlo = min(m1,m2); mhi = max(m1,m2)

    mx,my = if b<=mlo          # cuts the two edges meeting at the (0,0) corner
        b/2m1, b/2m2
    elseif b>=mhi              # cuts the two edges meeting at the (1,1) corner
        T(0.5)+(b-m2)/2m1, T(0.5)+(b-m1)/2m2
    elseif m1<=m2               # cuts the two edges normal to axis 1 (x1=0 and x1=1)
        T(0.5), b/m2-m1/2m2
    else                        # cuts the two edges normal to axis 2 (x2=0 and x2=1)
        b/m1-m2/2m1, T(0.5)
    end

    cx = ifelse(n1<0, 1-mx, mx)
    cy = ifelse(n2<0, 1-my, my)
    return SVector{2,T}(cx-T(0.5), cy-T(0.5))
end