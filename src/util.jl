import WaterLily: @loop, inside

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
function applyVOF!(f::AbstractArray{T,D},α::AbstractArray{T,D},n̂::AbstractArray{T,Dv},InterfaceSDF,I) where {T,D,Dv}
    # forwarddiff cause some problem so using 
    for i∈1:D
        xyzpδ = SVector{D,T}(loc(0,I).+0.01 .*δ(i,I).I)
        xyzmδ = SVector{D,T}(loc(0,I).-0.01 .*δ(i,I).I)
        n̂[I,i] = FreeSurfsdf(xyzpδ) - FreeSurfsdf(xyzmδ)
    end
    # TODO: the PLIC estimation
end

"""
    cleanWisp!(f; tol)

Clean out values in `f` too close to 0 or 1. The margin is 10 times the resolution of float type `T`.
"""
function cleanWisp!(f::AbstractArray{T,D}; tol=10eps(T)) where {T,D}
    @loop f[I] = ifelse(f[I]<       tol, T(0), f[I]) over I∈inside(f)
    @loop f[I] = ifelse(f[I]>one(T)-tol, T(1), f[I]) over I∈inside(f)
end

