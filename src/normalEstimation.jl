
"""
    reconstructInterface!(f,α,n̂)
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
    getInterfaceNormal_WY!(f,n̂,I)

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
        n̂[I,d] = (hl-hr)*0.5

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
    getInterfaceNormal_PCD!(f,n̂,I)

Normal reconstructure scheme from pure central difference approximation of 𝛁f.
"""
function getInterfaceNormal_PCD!(f::AbstractArray{T,D},n̂,I) where {T,D}
    for d∈1:D
        n̂[I,d] = f[I-δ(d,I)]-f[I+δ(d,I)]
    end
end