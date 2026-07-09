
"""
    surfTen!(forcing, f, α, n̂, fbuffer, η; perdir)

Compute surface tension force effect on momentum.
The curvature is calculated on each momentum direction separately, consequently, the surface tension force also.
"""
surfTen!(forcing,f::AbstractArray{T,D},α,n̂,fbuffer,η::Nothing;perdir=()) where {T,D} = nothing
surfTen!(forcing,f::AbstractArray{T,D},α,n̂,fbuffer,η::Number;perdir=()) where {T,D} = for d∈1:D
    @inside fbuffer[I] = ϕ(d,I,f)
    BCf!(d,fbuffer;perdir)
    @loop calNormal!(n̂,fbuffer,I) over I∈inside(fbuffer)
    @loop applySurfTen!(forcing,fbuffer,n̂,d,I,f,η)  over I∈inside(fbuffer)
end
@inline calNormal!(n̂,fbuffer,I) = if containInterface(fbuffer[I]) 
    getInterfaceNormal_WY!(fbuffer,n̂,I)
end
@inline applySurfTen!(forcing,fbuffer,n̂,d,I,f,η) = if containInterface(fbuffer[I]) 
    forcing[I,d] += η*getCurvature(I,fbuffer,majorDir(n̂,I))*-∂(d,I,f)
end

"""
    getCurvature(I,f,i)

Formula from [Patel et al. (2019)](https://doi.org/10.1016/j.compfluid.2019.104263) or on [Basilisk.fr](http://basilisk.fr/src/curvature.h).
Cross derivaties from [Wikipedia](https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences).
This function has been dispatched for 2D and 3D.
"""
function getCurvature(I::CartesianIndex{3},f::AbstractArray{T,3},i;filter=T(0.2)) where T
    ix,iy = getXYdir(i)
    H = @SMatrix [
        getPopinetHeight(I+xUnit*δd(ix,I)+yUnit*δd(iy,I),f,i)
        for xUnit∈-1:1,yUnit∈-1:1
    ]
    Hₓ = (H[3,2] - H[1,2])/2
    Hᵧ = (H[2,3] - H[2,1])/2
    Hₓₓ= (
            (H[3,2] + H[1,2] - 2*H[2,2]) + 
            (H[3,1] + H[1,1] - 2*H[2,1])*filter +
            (H[3,3] + H[1,3] - 2*H[2,3])*filter
        )/(1+2filter)
    Hᵧᵧ= (
            (H[2,3] + H[2,1] - 2*H[2,2]) + 
            (H[1,3] + H[1,1] - 2*H[1,2])*filter +
            (H[3,3] + H[3,1] - 2*H[3,2])*filter
        )/(1+2filter)
    Hₓᵧ= (H[3,3] + H[1,1] - H[3,1] - H[1,3])/4
    return (Hₓₓ*(1+Hᵧ^2) + Hᵧᵧ*(1+Hₓ^2) - 2Hₓᵧ*Hₓ*Hᵧ)/root1p5(1+Hₓ^2+Hᵧ^2)
end
function getCurvature(I::CartesianIndex{2},f::AbstractArray{T,2},i) where T
    ix = getXdir(i)
    H = @SArray [
        getPopinetHeight(I+xUnit*δd(ix,I),f,i)
        for xUnit∈-1:1
    ]
    Hₓ = (H[3]-H[1])/2
    Hₓₓ= (H[3]+H[1]-2H[2])
    return Hₓₓ/root1p5(1+Hₓ^2)
end

makeA(x::T,j,valid) where T = ifelse(valid, if j==1 x^2 elseif j==2 x else 1 end, T(0))
makey(y::T,  valid) where T = ifelse(valid, y, T(0))

"""
    getParabolicStencil(I,f,n̂)

Gather the 3 height-function points and 3 width-function points used by the 3x3 parabolic
curvature fit, together with their validity flags. Returns `(ix,iy,p,pvalid)`.

`iy` is the dominant grid direction (signed), `ix` the other one, chosen so that the normal
points "northeast". Each entry of `p` is `(x,y)` with `x` measured from the centered
column/row (in `ix` units, exact) and `y` measured from the bottom of the 3-cell stack (in
`iy` units, i.e. shifted by `+1.5` relative to cell `I`'s center).
"""
function getParabolicStencil(I::CartesianIndex{2},f::AbstractArray{T,2},n̂::AbstractArray{T,3}) where T
    # align all the case so that the normal is always toward northeast
    iy = majorDir(n̂,I)
    ix = getXdir(iy); nx = n̂[I,abs(ix)]; sgn = ifelse(ix*nx<0,-1,1); ix*=sgn
    absnx = abs(nx)

    # Height
    hy0 = get3CellHeight(f,I-δd(ix,I),iy)
    hy1 = get3CellHeight(f,I         ,iy)
    hy2 = get3CellHeight(f,I+δd(ix,I),iy)

    # Width
    wx0 = get3CellHeight(f,I-δd(iy,I),ix) - T(1.5)
    wx1 = get3CellHeight(f,I         ,ix) - T(1.5)
    wx2 = get3CellHeight(f,I+δd(iy,I),ix) - T(1.5)

    ∑f3x3 = hy0+hy1+hy2

    validhy2 = 9-1abs(nx)/2 > ∑f3x3 > 9abs(nx)/2
    validhy1 = 9-4abs(nx)/2 > ∑f3x3 > 4abs(nx)/2
    validhy0 = 9-9abs(nx)/2 > ∑f3x3 > 1abs(nx)/2

    validwx0 = 9-9/abs(nx)/2 > ∑f3x3 > 1/abs(nx)/2
    validwx1 = 9-4/abs(nx)/2 > ∑f3x3 > 4/abs(nx)/2
    validwx2 = 9-1/abs(nx)/2 > ∑f3x3 > 9/abs(nx)/2

    p = (
        (T(-1), hy0),
        (T( 0), hy1),
        (T( 1), hy2),
        (wx0, T(0.5)),
        (wx1, T(1.5)),
        (wx2, T(2.5))
    )

    pvalid = (validhy0,validhy1,validhy2,validwx0,validwx1,validwx2)
    return ix,iy,p,pvalid
end

"""
    getCurvature_Parabolic(I,f,n̂)

Fit a grid-aligned parabola `y=a₁x²+a₂x+a₃` through the 3x3 height/width points from
[`getParabolicStencil`](@ref) and return its curvature at `x=0`.
"""
function getCurvature_Parabolic(I::CartesianIndex{2},f::AbstractArray{T,2},n̂::AbstractArray{T,3}) where T
    _,_,p,pvalid = getParabolicStencil(I,f,n̂)

    S = @SMatrix [makeA(p[i][1],j,pvalid[i]) for i in 1:6, j in 1:3]
    y = @SArray  [makey(p[i][2],  pvalid[i]) for i in 1:6]
    a = (S'*S)\(S'*y)

    κ = 2a[1]/root1p5(1+a[2]^2)
    return ifelse(isnan(κ), T(0), κ)
end

makeAᵗ(s::T,j,valid) where T = ifelse(valid, ifelse(j==1, s^2, one(T)), zero(T))
maketᵗ(t::T,  valid) where T = ifelse(valid, t, zero(T))

"""
    toLocalST(Δx,Δy,ix,iy,cen,t1,t2,n1,n2)

Convert a stencil point `(Δx,Δy)` — given in the reflected `(ix,iy)` grid frame, relative to
cell `I`'s center — into `(s,t)` coordinates: `s` along the tangent `(t1,t2)`, `t` along the
normal `(n1,n2)`, both measured from the interface center `cen`.
"""
@inline function toLocalST(Δx::T,Δy::T,ix,iy,cen,t1,t2,n1,n2) where T
    gx = ifelse(abs(ix)==1, Δx*sign(ix), Δy*sign(iy)) - cen[1]
    gy = ifelse(abs(ix)==2, Δx*sign(ix), Δy*sign(iy)) - cen[2]
    return gx*t1+gy*t2, gx*n1+gy*n2
end

"""
    getCurvature_ParabolicInclined(I,f,n̂,α)

Same 3x3 height/width point selection and validity as [`getCurvature_Parabolic`](@ref), but
fit the parabola in an inclined frame whose centerline is the ray through the interface
center ([`getInterfaceCenter`](@ref)) that follows the (exact, unreflected) interface normal
— rather than a grid-aligned frame. Since the centerline is fixed by construction, the fit
only needs the symmetric model `t=a₁s²+a₃` (no linear term): the parabola's axis of symmetry
is forced to be the normal ray, but it need not pass through the interface center itself
(`a₃` is left free). The curvature at `s=0` then simplifies to `2a₁`.
"""
function getCurvature_ParabolicInclined(I::CartesianIndex{2},f::AbstractArray{T,2},n̂::AbstractArray{T,3},α::AbstractArray{T,2}) where T
    ix,iy,p,pvalid = getParabolicStencil(I,f,n̂)

    # exact (unreflected) unit normal/tangent: these define the parabola's centerline
    ninv = 1/sqrt(n̂[I,1]^2+n̂[I,2]^2)
    n1 = n̂[I,1]*ninv; n2 = n̂[I,2]*ninv
    t1 = -n2; t2 = n1

    cen = getInterfaceCenter(n̂,α,I)

    S  = @SMatrix [makeAᵗ(toLocalST(p[i][1],p[i][2]-T(1.5),ix,iy,cen,t1,t2,n1,n2)[1],j,pvalid[i]) for i in 1:6, j in 1:2]
    tv = @SArray  [maketᵗ(toLocalST(p[i][1],p[i][2]-T(1.5),ix,iy,cen,t1,t2,n1,n2)[2],  pvalid[i]) for i in 1:6]
    a = (S'*S)\(S'*tv)

    κ = 2a[1]
    return ifelse(isnan(κ), T(0), κ)
end

"""
    getHeight(I,f,i)

Calculate water height of a single column.
"""
function getPopinetHeight(I,f,i)
    H,consistent = getPopinetHeightAdaptive(I,f,i,true)
    return H
end

"""
    getPopinetHeightAdaptive(I,f,i)

Return the column height relative to cell `I` center along signed `i` direction, which points to where there is no water.
The function is based on the Algorithm 4 from [Popinet, JCP (2009)](https://doi.org/10.1016/j.jcp.2009.04.042).
If `monotonic` is activated, the summation will only cover the monotonic range. The monotonic condition is based on [Guo et al., Appl. Math. Model. (2015)](https://doi.org/10.1016/j.apm.2015.04.022).
"""
function getPopinetHeightAdaptive(I,f::AbstractArray{T,D},i,monotonic=true) where {T,D}
    consistent = true
    Inow = I; fnow = f[Inow]; H = (fnow-T(0.5))
    # Iterate till reach the cell full of air
    finishInd = fnow<1
    while !finishInd || containInterface(fnow)
        Inow += δd(i,I); !validCI(Inow,f) && break
        fnow = ifelse(monotonic && f[Inow]>fnow, zero(T), f[Inow]) # type stability is important in ifelse in GPU...
        H += fnow
        finishInd = ifelse(containInterface(fnow),true,finishInd)
    end
    consistent = (fnow==0) && consistent
    Inow = I; fnow = f[Inow]
    # Iterate till reach the cell full of water
    finishInd = fnow>0
    while !finishInd || containInterface(fnow)
        Inow -= δd(i,I); !validCI(Inow,f) && break
        fnow = ifelse(monotonic && f[Inow]<fnow, one(T), f[Inow])
        H += fnow-1  # a little trick that make `I` cell center the origin
        finishInd = ifelse(containInterface(fnow),true,finishInd)
    end
    consistent = (fnow==1) && consistent
    return H,consistent
end

@inline @fastmath root1p5(a) = sqrt(a^3)