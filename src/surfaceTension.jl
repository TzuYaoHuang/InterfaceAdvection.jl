
"""
    surfTen!(forcing, f, α, n̂, fbuffer, η; perdir)

Compute surface tension force effect on momentum.
The curvature is calculated on each momentum direction separately, consequently, the surface tension force also.
"""
surfTen!(forcing,f::AbstractArray{T,D},α,n̂,fbuffer,η::Nothing;perdir=()) where {T,D} = nothing
surfTen!(forcing,f::AbstractArray{T,D},α,n̂,fbuffer,η::Number;perdir=()) where {T,D} = for d∈1:D
    @inside fbuffer[I] = ϕ(d,I,f)
    # TODO: This did not take boundary shift into account, need revision.
    BCf!(f;perdir)
    @loop containInterface(fbuffer[I]) ? getInterfaceNormal_WY!(f,n̂,I) : nothing over I∈inside(fbuffer)
    @loop forcing[I,d] += containInterface(fbuffer[I]) ? η*getCurvature(I,fbuffer,majorDir(n̂,I))*-∂(d,I,f) : T(0)  over I∈inside(fbuffer)
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
    Hx = (H[3,2] - H[1,2])/2
    Hy = (H[2,3] - H[2,1])/2
    Hxx= (
            (H[3,2] + H[1,2] - 2*H[2,2]) + 
            (H[3,1] + H[1,1] - 2*H[2,1])*filter +
            (H[3,3] + H[1,3] - 2*H[2,3])*filter
        )/(1+2filter)
    Hyy= (
            (H[2,3] + H[2,1] - 2*H[2,2]) + 
            (H[1,3] + H[1,1] - 2*H[1,2])*filter +
            (H[3,3] + H[3,1] - 2*H[3,2])*filter
        )/(1+2filter)
    Hxy= (H[3,3] + H[1,1] - H[3,1] - H[1,3])/4
    return (Hxx*(1+Hy^2) + Hyy*(1+Hx^2) - 2Hxy*Hx*Hy)/(1+Hx^2+Hy^2)^T(1.5)
end
function getCurvature(I::CartesianIndex{2},f::AbstractArray{T,2},i) where T
    ix = getXdir(i)
    H = @SArray [
        getPopinetHeight(I+xUnit*δd(ix,I),f,i)
        for xUnit∈-1:1
    ]
    Hₓ = (H[3]-H[1])/2
    Hₓₓ= (H[3]+H[1]-2H[2])
    return Hₓₓ/(1+Hₓ^2)^1.5
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
