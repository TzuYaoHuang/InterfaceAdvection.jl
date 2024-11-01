
"""
    myArgAbsMax(vec[,I])

Return where is the absolute maximum since the original `argmax` function in julia is not working in GPU environment.
"""
function myArgAbsMax(vec)
    max = abs2(vec[1])
    iMax = 1
    for i∈2:length(vec)
        cur = abs2(vec[i])
        if cur > max
            max = cur
            iMax = i
        end
    end
    return iMax
end
function myArgAbsMax(vec,I)
    max = abs2(vec[I,1])
    iMax = 1
    for i∈2:size(vec)[end]
        cur = abs2(vec[I,i])
        if cur > max
            max = cur
            iMax = i
        end
    end
    return iMax
end

"""
    boxAroundI(I)

Return 3 surrunding cells in each direction of `I`, including diagonal ones.
The return grid number adds up to 3ᴰ 
"""
boxAroundI(I::CartesianIndex{D}) where D = (I-oneunit(I)):(I+oneunit(I))

"""
    inside_uWB(dims,j)

Return CartesianIndices range including the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`. (WB stands for with boundaries.)
"""
function inside_uWB(dims::NTuple{D},j) where {D}
    CartesianIndices(ntuple( i-> i==j ? (2:dims[i]) : (2:dims[i]-1), D))
end

"""
    δd(i,I)

It is still coordinate shifting stuff but with direction (+/-) support for `i`.
"""
δd(i,I) = sign(i)*δ(abs(i),I)

"""
    getXdir(i)
    getXYdir(i)

Get another 1 or 2 direction(s) of `i` with support to indicate direction (+/-) for `i`.
"""
getXdir(i) = ifelse(abs(i)==1,-2sign(i),sign(i))
getXYdir(i) = (sign(i)*((abs(i))%3+1),(abs(i)+1)%3+1)

"""
    majorDir(n̂, I)

Get the major direction of reconstructed cell with +/- direction copied
"""
function majorDir(n̂,I)
    i = myArgAbsMax(n̂,I)
    return copysign(i,n̂[I,i])
end

"""
    validCI(I,f)

Check whether the cartesian index is valid or not.
"""
validCI(I::CartesianIndex{D},f::AbstractArray{T,D}) where {T,D} = all(ntuple(i->1,D) .≤ I.I .≤ size(f))