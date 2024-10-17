
"""
    getIntercept(v, g)
    getIntercept(n̂, I, g)
    getIntercept(n1, n2, g)
    getIntercept(n1, n2, n3, g)

Calculate intercept from volume fraction.
These functions prepare `n̂` and `g` for `f2α`.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
@inline getIntercept(n̂::AbstractArray{T,3},I::CartesianIndex{2},g) where T = getIntercept(n̂[I,1],n̂[I,2],g)
@inline getIntercept(n̂::AbstractArray{T,4},I::CartesianIndex{3},g) where T = getIntercept(n̂[I,1],n̂[I,2],n̂[I,3],g)
@inline getIntercept(v::AbstractArray{T,1}, g) where T = (
    length(v)==2 ?
    getIntercept(v[1], v[2], g) :
    getIntercept(v[1], v[2], v[3], g)
)
@inline @fastmath function getIntercept(n1::T, n2::T, g::T) where T
    t = abs(n1)+abs(n2)
    if g!=0.5
        m1, m2 = sort2(abs(n1)/t, abs(n2)/t)
        a = f2α(m1, m2, ifelse(g<0.5, g, 1-g))
    else
        a = T(0.5)
    end
    return ifelse(g<0.5, a, 1-a)*t + min(n1, T(0)) + min(n2, T(0))
end
@inline @fastmath function getIntercept(n1::T, n2::T, n3::T, g::T) where T
    t = abs(n1)+abs(n2)+abs(n3)
    if g!=0.5
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        a = f2α(m1, m2, m3, ifelse(g<0.5, g, 1-g))
    else
        a = T(0.5)
    end
    return ifelse(g<0.5, a, 1-a)*t + min(n1, T(0)) + min(n2, T(0)) + min(n3, T(0))
end

"""
    getVolumeFraction(v, b)
    getVolumeFraction(n̂, I, b)
    getVolumeFraction(n1, n2, b)
    getVolumeFraction(n1, n2, n3, b)

Calculate volume fraction from intercept.
These functions prepare `n̂` and `b` for `α2f`.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
@inline getVolumeFraction(n̂::AbstractArray{T,3},I::CartesianIndex{2},b) where T = getVolumeFraction(n̂[I,1],n̂[I,2],b)
@inline getVolumeFraction(n̂::AbstractArray{T,4},I::CartesianIndex{3},b) where T = getVolumeFraction(n̂[I,1],n̂[I,2],n̂[I,3],b)
@inline getVolumeFraction(v::AbstractArray{T,1}, b) where T = (
    length(v)==2 ?
    getVolumeFraction(v[1], v[2], b) :
    getVolumeFraction(v[1], v[2], v[3], b)
)
@inline @fastmath function getVolumeFraction(n1::T, n2::T, b::T) where T
    t = abs(n1)+abs(n2)
    a = (b-min(n1, 0)-min(n2, 0))/t

    if a<=0 || a==0.5 || a>=1
        return min(max(a, T(0)), T(1))
    else
        m1, m2 = sort2(abs(n1)/t, abs(n2)/t)
        t = α2f(m1, m2, ifelse(a<0.5, a, 1-a))
        return ifelse(a<0.5, t, 1-t)
    end
end
@inline @fastmath function getVolumeFraction(n1::T, n2::T, n3::T, b::T) where T
    t = abs(n1) + abs(n2) + abs(n3)
    a = (b - min(n1, 0) - min(n2, 0) - min(n3, 0))/t

    if a<=0 || a==0.5 || a>= 1
        return min(max(a, T(0)), T(1))
    else
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        t = α2f(m1, m2, m3, ifelse(a<0.5, a, 1-a))
        return ifelse(a<0.5, t, 1-t)
    end
end

"""
    α2f(m1, m2, a)
    α2f(m1, m2, m3, a)

Two/Three-Dimensional Forward Problem.
Get volume fraction from intersection.
This is restricted to (1) 2/3D, (2) n̂ᵢ ≥ 0 ∀ i, (3) ∑ᵢ n̂ᵢ = 1, (4) a < 0.5.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
@inline @fastmath α2f(m1::T, m2::T, a::T) where T = a<m1 ? a^2/(2m1*m2) : (a-m1/2)/m2
@inline @fastmath function α2f(m1::T, m2::T, m3::T, a::T) where T
    m12 = m1+m2

    if a<m1
        return a^3/(6m1*m2*m3)
    elseif a<m2
        return a*(a-m1)/(2m2*m3) + ifelse(m2==0, T(1), m1/m2)*m1/6m3  # change proposed by Kelli Hendricson to avoid the divided by zero issue
    elseif a<min(m3, m12)
        return (a^2*(3m12-a) + m1^2*(m1-3a) + m2^2*(m2-3a))/(6m1*m2*m3)
    elseif m3<m12
        return (a^2*(3-2a) + m1^2*(m1-3a) + m2^2*(m2-3a) + m3^2*(m3-3a))/(6m1*m2*m3)
    else
        return (2a-m12)/2m3
    end
end

"""
    f2α(m1, m2, v)
    f2α(m1, m2, m3, v)

Two/Three-Dimensional Inverse Problem.
Get intercept with volume fraction.
This is restricted to (1) 2/3D, (2) n̂ᵢ ≥ 0 ∀ i, (3) ∑ᵢ n̂ᵢ = 1, (4) v < 0.5.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
@inline @fastmath f2α(m1::T, m2::T, v::T) where T = v<m1/2m2 ? sqrt(2*m1*m2*v) : m2*v+m1/2
@inline @fastmath function f2α(m1::T, m2::T, m3::T, v::T) where T
    m12 = m1 + m2
    
    p = 6m1*m2*m3
    v1 = ifelse(m2==0, one(T), m1/m2) * m1/6m3   # change proposed by Kelli Hendricson to avoid the divided by zero issue
    v2 = v1 + (m2-m1)/2m3
    v3 = ifelse(
        m3 < m12, 
        (m3^2*(3m12-m3) + m1^2*(m1-3m3) + m2^2*(m2-3m3))/p,
        m12/2m3
    )

    if v < v1
        return cbrt(p*v)
    elseif v < v2
        return (m1 + sqrt(m1^2 + 8*m2*m3*(v-v1)))/2
    elseif v < v3
        c0 = m1^3 + m2^3 - p*v
        c1 = -3*(m1^2 + m2^2)
        c2 = 3*m12
        c3 = -T(1)
        return proot(c0, c1, c2, c3)
    elseif m3 < m12
        c0 = m1^3 + m2^3 + m3^3 - p*v
        c1 = -3*(m1^2 + m2^2 + m3^2)
        c2 = T(3)
        c3 = -T(2)
        return proot(c0, c1, c2, c3)
    else
        return m3*v + m12/2
    end
end





############################################################
# +++++++ Auxiliary functions for PLIC calculation +++++++ #
############################################################

"""
    proot(c0, c1, c2, c3)

Calculate the roots of a third order polynomial, which has three real roots:
    c3 x³ + c2 x² + c1 x¹ + c0 = 0
"""
@inline @fastmath function proot(c0::T, c1::T, c2::T, c3::T) where T
    a0 = c0/c3
    a1 = c1/c3
    a2 = c2/c3

    p0 = a1/3 - a2^2/9
    q0 = (a1*a2 - 3a0)/6 - a2^3/27
    a = q0/sqrt(-p0^3)
    t = acos(ifelse(abs2(a)<=1,a,zero(T)))/3

    return sqrt(-p0)*(sqrt(T(3))*sin(t) - cos(t)) - a2/3
end

@inline @fastmath sort2(a,b) =  a<b ? (a, b) : (b, a)

"""
    sort3(a, b, c)

Sort three numbers with bubble sort algorithm to avoid too much memory assignment due to array creation.
see https://codereview.stackexchange.com/a/91920
"""
@inline @fastmath function sort3(a, b, c)
    if (a>c) a,c = c,a end
    if (a>b) a,b = b,a end
    if (b>c) b,c = c,b end
    return a,b,c
end