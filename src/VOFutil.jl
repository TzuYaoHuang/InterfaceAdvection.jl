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
function applyVOF!(f::AbstractArray{T,D},α::AbstractArray{T,D},n̂::AbstractArray{T,Dv},InterfaceSDF,I) where {T,D,Dv}
    # forwarddiff cause some problem so using finite diff
    δd = 0.01
    for i∈1:D
        xyzpδ = SVector{D,T}(loc(0,I) .+δd .*δ(i,I).I)
        xyzmδ = SVector{D,T}(loc(0,I) .-δd .*δ(i,I).I)
        n̂[I,i] = InterfaceSDF(xyzpδ) - InterfaceSDF(xyzmδ)
    end
    sumN2 = 0; for i∈1:D sumN2+= n̂[I,i]^2 end

    # (n̂·𝐱 - α)/|n̂| = d
    α[I] = - √sumN2*InterfaceSDF(loc(0,I).-0.5)

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
            # TODO: can we merge f,α,n̂ together?
            @loop f[I] = f[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
            @loop f[I] = f[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
            for i ∈ 1:D
                @loop n̂[I,i] = n̂[CIj(j,I,N[j]-1),i] over I ∈ slice(N,1,j)
                @loop n̂[I,i] = n̂[CIj(j,I,2),i] over I ∈ slice(N,N[j],j)
            end
            @loop α[I] = α[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
            @loop α[I] = α[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
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
    @loop f[I] = ifelse(f[I]<       tol, T(0), f[I]) over I∈inside(f)
    @loop f[I] = ifelse(f[I]>one(T)-tol, T(1), f[I]) over I∈inside(f)
end


"""
    containInterface(f)

Check whether `f` is interface cell.
"""
containInterface(f) = 0<f<1

"""
    fullorempty(fc)

Check whether `fc` is full of dark or light fluid.
"""
fullorempty(fc) = (fc==0.0 || fc==1.0)

"""
    get3CellHeight(f,I,summingDir)

Get three cell volume summation around index `I` along direction `summingDir`.
"""
get3CellHeight(f,I,summingDir) = f[I]+f[I-δ(summingDir,I)]+f[I+δ(summingDir,I)]

