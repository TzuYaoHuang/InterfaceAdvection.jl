
"""
    reconstructInterface!(f,α,n̂;perdir=())
    reconstructInterface!(f,α,n̂,I)

Reconstruct interface from volume fraction field (`f`), involving normal calculation (`n̂`) and then the intercept (`α`).
"""
function reconstructInterface!(f,α,n̂;perdir=())
    @loop reconstructInterface!(f,α,n̂,I) over I∈inside(f)
    BCVOF!(f,α,n̂;perdir)
end
function reconstructInterface!(f::AbstractArray{T,D},α,n̂,I) where {T,D}
    # guarding if to role out non-interface cell
    if fullorempty(f[I])
        for i∈1:D n̂[I,i] = 0 end
        α[I] = 0
        return nothing
    end

    # get normal
    getInterfaceNormal_WH!(f,n̂,I)
    # getInterfaceNormal_WY!(f,n̂,I)
    # getInterfaceNormal_MYC!(f,n̂,I)
    # getInterfaceNormal_Column!(f,n̂,I)

    # get intercept
    α[I] = getIntercept(n̂,I,f[I])
    return nothing
end

"""
    getInterfaceNormal_WY!(f,n̂,I)

Normal reconstructure scheme from [Weymouth & Yue (2010)](https://doi.org/10.1016/j.jcp.2009.12.018). It's 3x3 compact height function with some checks.
"""
function getInterfaceNormal_WY!(f::AbstractArray{T,D},n̂,I) where {T,D}
    # Initial guess on normal based on pure central difference
    getInterfaceNormal_PCD!(f,n̂,I)

    # Get guessed dominant direction
    dominantDir = myArgAbsMax(n̂,I)
    
    for d∈1:D
        if d==dominantDir
            n̂[I,d] = sign(n̂[I,d])
            continue
        end

        # height of column from i-1, i, i+1
        hl = get3CellHeight(f,I-δ(d,I),dominantDir)
        hc = get3CellHeight(f,I       ,dominantDir)
        hr = get3CellHeight(f,I+δ(d,I),dominantDir)

        # general case
        n̂[I,d] = (hl-hr)/2

        # Too steep
        n̂[I,d] = ifelse(
            abs(n̂[I,d])>0.5,   # abs cost is identical to abs2
            ifelse(
                n̂[I,d]*(hc-1.5) >= 0.0, 
                hc-hr, 
                hl-hc
            ), 
            n̂[I, d]
        )
    end
end

"""
    getInterfaceNormal_Column!(f,n̂,I)

Normal reconstructure scheme just using pure centered column difference.
"""
function getInterfaceNormal_Column!(f::AbstractArray{T,D},n̂,I) where {T,D}
    # Initial guess on normal based on pure central difference
    getInterfaceNormal_PCD!(f,n̂,I)

    # Get guessed dominant direction
    dominantDir = myArgAbsMax(n̂,I)
    
    for d∈1:D
        if d==dominantDir
            n̂[I,d] = sign(n̂[I,d])
            continue
        end

        # height of column from i-1, i, i+1
        hl = get3CellHeight(f,I-δ(d,I),dominantDir)
        hc = get3CellHeight(f,I       ,dominantDir)
        hr = get3CellHeight(f,I+δ(d,I),dominantDir)

        # general case
        n̂[I,d] = (hl-hr)/2
    end
end

function getInterfaceNormal_WH!(f::AbstractArray{T,D},n̂,I) where {T,D}
    # Initial guess on normal based on pure central difference
    getInterfaceNormal_Column!(f,n̂,I)

    # Get guessed dominant direction
    dominantDir = majorDir(n̂,I)
    absdoDir = abs(dominantDir)
    absn̂dom = abs(n̂[I,absdoDir])

    for d ∈ 1:D n̂[I,d] /= absn̂dom end
    
    for d ∈ 1:D
        if d==abs(dominantDir)
            continue
        end

        slope = abs(n̂[I,d])

        curDir = copysign(d, n̂[I,d])

        # height of column from i-1, i, i+1
        hl = get3CellHeight(f,I-δd(curDir,I),absdoDir)
        hc = get3CellHeight(f,I             ,absdoDir)
        hr = get3CellHeight(f,I+δd(curDir,I),absdoDir)
        ∑h = hl+hc+hr

        if 4.5slope ≤ ∑h ≤ 9-4.5slope
            continue
        end

        if ∑h < 4.5slope
            wb = get3CellHeight(f,I-δd(dominantDir,I), d)
            n̂[I,d] = copysign((hl-0.5)/(wb-0.5),curDir)
        end
        if ∑h > 9 - 4.5slope
            wt = get3CellHeight(f,I+δd(dominantDir,I), d)
            n̂[I,d] = copysign((2.5-hr)/(2.5-wt),curDir) # what if hr > 2.5
        end
    end
end

"""
    getInterfaceNormal_PCD!(f,n̂,I)

Normal reconstructure scheme from pure central difference approximation of 𝛁f.
"""
function getInterfaceNormal_PCD!(f::AbstractArray{T,D},n̂,I) where {T,D}
    for d∈1:D
        n̂[I,d] = f[I-δ(d,I)]-f[I+δ(d,I)]
    end
end

"""
    getInterfaceNormal_SLIC!f,n̂,I)

Normal reconstructure scheme for pure vertical or horizontal surfaces. on-off based on result from PCD.
"""
function getInterfaceNormal_SLIC!(f::AbstractArray{T,D},n̂,I) where {T,D}
    getInterfaceNormal_PCD!(f,n̂,I)
    d = myArgAbsMax(n̂,I)
    for i∈1:D n̂[I,i] = ifelse(i==d,sign(n̂[I,i]),T(0)) end
end

"""
    getInterfaceNormal_MYC!(f,nhat,I)

Mixed Youngs-Centered normal reconstructure scheme from [Aulisa et al. (2007)](https://doi.org/10.1016/j.jcp.2007.03.015), but I think best explained by 
[Duz (2005) page 81](https://doi.org/10.4233/uuid:e204277d-c334-49a2-8b2a-8a05cf603086) and [Baraldi et al. (2014)](http://doi.org/10.1016/j.compfluid.2013.12.018).
One can also be referred to the source code of [PARIS](http://www.ida.upmc.fr/~zaleski/paris/). It is in vofnonmodule.f90.
"""
function getInterfaceNormal_MYC!(f::AbstractArray{T,n},n̂,I) where {T,n}
    getInterfaceNormal_Y!(f,n̂,I)
    maxNhat = T(0)
    for i∈1:n maxNhat = ifelse(abs(n̂[I,i])>maxNhat, abs(n̂[I,i]), maxNhat) end

    curm0 = 0
    CCiz = 0
    for iz∈1:n
        curNhat = getInterfaceNormal_CCi(f,n̂,I,iz)
        if abs(curNhat[iz])>curm0 CCiz = iz end
        curm0 = abs(curNhat[iz])
    end
    CCNhat = getInterfaceNormal_CCi(f,n̂,I,CCiz)
    
    if abs(CCNhat[CCiz]) < maxNhat
        for i∈1:n n̂[I,i] = CCNhat[i] end
    end
end

"""
    getInterfaceNormal_CCi!(f,nCD,I,dc)

Normal reconstructure scheme from Center column method but only in `dc` direction.
Assume we have already calculated a guessed normal to set the direction (sign) of interface in `nCD`. 
"""
function getInterfaceNormal_CCi(f::AbstractArray{T,n},n̂,I,dc) where {T,n}
    nhat = ntuple(
        d -> if (d == dc)
            sign(n̂[I,d])
        else
            hu = get3CellHeight(f, I+δ(d,I), dc)
            hd = get3CellHeight(f, I-δ(d,I), dc)
            -(hu-hd)/2
        end,
        n
    )
    nhatN = nhat./sum(abs,nhat)
    return nhatN
end

"""
    getInterfaceNormal_Y!(f, nhat, I)

Calculate the interface normal from [Youngs (1982)](https://www.researchgate.net/publication/249970655_Time-Dependent_Multi-material_Flow_with_Large_Fluid_Distortion).
Note that `nhat` is view of `n̂[I,:]`.
"""
function getInterfaceNormal_Y!(f::AbstractArray{T,D},n̂,I) where {T,D}
    a = 0
    for d ∈ 1:D
        n̂[I,d] = (YoungSum(f,I-δ(d,I),d) - YoungSum(f,I+δ(d,I),d))*0.5
        a += abs(n̂[I,d])
    end
    for d ∈ 1:D
        n̂[I,d] /= a
    end
end
function YoungSum(f,I,d)
    δxy = oneunit(I)-δ(d,I)
    a = 0
    for II∈I-δxy:I
        for III∈II:II+δxy
            a+=f[III]
        end
    end
    return a
end


"""
    getInterfaceNormal_CD!(f, nhat, I)

Calculate the interface normal from the central difference scheme with the closest neighbor considered (4 neighbor in 3D).
Note that `nhat` is view of `n̂[I,:]`.
"""
function getInterfaceNormal_CD!(f::AbstractArray{T,n},n̂,I) where {T,n}
    for d ∈ 1:n
        n̂[I,d] = (crossSummation(f,I-δ(d,I),d)-crossSummation(f,I+δ(d,I),d))*0.5
    end
end
function crossSummation(f::AbstractArray{T,n},I,d,γ=1.0) where {T,n}
    a = f[I]
    # for iDir∈getAnotherDir(d,n)
    #     a += f[I-δ(iDir,I)]+f[I+δ(iDir,I)]
    # end
    for iDir∈1:n
        a += iDir≠d ? γ*(f[I-δ(iDir,I)]+f[I+δ(iDir,I)]) : 0
    end
    return a
end

function getInterfaceNormal_XYLIC!(f::AbstractArray{T,n},n̂,I,d=2) where {T,n}
    for i∈1:n
        n̂[I,i] = ifelse(i==d,1,0)
    end
end