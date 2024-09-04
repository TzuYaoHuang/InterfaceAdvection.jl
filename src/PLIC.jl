

"""
    getIntercept(v, g)

Calculate intersection from volume fraction.
These functions prepare `n̂` and `g` for `f2α`.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
getIntercept(n̂::AbstractArray{T,3},I::CartesianIndex{2},g) where T = getIntercept(n̂[I,1],n̂[I,2],zero(T),g)
getIntercept(n̂::AbstractArray{T,4},I::CartesianIndex{3},g) where T = getIntercept(n̂[I,1],n̂[I,2],n̂[I,3],g)
getIntercept(v::AbstractArray{T,1}, g) where T = (
    length(v)==2 ?
    getIntercept(v[1], v[2], zero(T), g) :
    getIntercept(v[1], v[2], v[3], g)
)
function getIntercept(n1, n2, n3, g)
    t = abs(n1) + abs(n2) + abs(n3)
    if g != 0.5
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        a = f2α(m1, m2, m3, ifelse(g < 0.5, g, 1.0 - g))
    else
        a = 0.5
    end
    return ifelse(g < 0.5, a, 1.0 - a)*t + min(n1, 0.0) + min(n2, 0.0) + min(n3, 0.0)
end

"""
    getVolumeFraction(v, b)

Calculate intersection from volume fraction.
These functions prepare `n̂` and `b` for `α2f`.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
getVolumeFraction(n̂::AbstractArray{T,3},I::CartesianIndex{2},b) where T = getVolumeFraction(n̂[I,1],n̂[I,2],zero(T),b)
getVolumeFraction(n̂::AbstractArray{T,4},I::CartesianIndex{3},b) where T = getVolumeFraction(n̂[I,1],n̂[I,2],n̂[I,3],b)
getVolumeFraction(v::AbstractArray{T,1}, b) where T = (
    length(v)==2 ?
    getVolumeFraction(v[1], v[2], zero(T), b) :
    getVolumeFraction(v[1], v[2], v[3], b)
)
function getVolumeFraction(n1, n2, n3, b)
    t = abs(n1) + abs(n2) + abs(n3)
    a = (b - min(n1, 0.0) - min(n2, 0.0) - min(n3, 0.0))/t

    if a <= 0.0 || a == 0.5 || a >= 1.0
        return min(max(a, 0.0), 1.0)
    else
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        t = α2f(m1, m2, m3, ifelse(a < 0.5, a, 1.0 - a))
        return ifelse(a < 0.5, t, 1.0 - t)
    end
end

"""
    α2f(m1, m2, m3, a)

Three-Dimensional Forward Problem.
Get volume fraction from intersection.
This is restricted to (1) 3D, (2) n̂ᵢ ≥ 0 ∀ i, (3) ∑ᵢ n̂ᵢ = 1, (4) a < 0.5.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
function α2f(m1, m2, m3, a)
    m12 = m1 + m2

    if a < m1
        return a^3/(6.0*m1*m2*m3)
    elseif a < m2
        return a*(a - m1)/(2.0*m2*m3) + ifelse(m2 == 0.0, 1.0, m1 / m2) * (m1 / (6.0 * m3))  # change proposed by Kelli Hendricson to avoid the divided by zero issue
    elseif a < min(m3, m12)
        return (a^2*(3.0*m12 - a) + m1^2*(m1 - 3.0*a) + m2^2*(m2 - 3.0*a))/(6*m1*m2*m3)
    elseif m3 < m12
        return (a^2*(3.0 - 2.0*a) + m1^2*(m1 - 3.0*a) + m2^2*(m2 - 3.0*a) + m3^2*(m3 - 3.0*a))/(6*m1*m2*m3)
    else
        return (2.0*a - m12)/(2.0*m3)
    end
end

"""
    f2α(m1, m2, m3, v)

Three-Dimensional Inverse Problem.
Get intercept with volume fraction.
This is restricted to (1) 3D, (2) n̂ᵢ ≥ 0 ∀ i, (3) ∑ᵢ n̂ᵢ = 1, (4) v < 0.5.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
function f2α(m1, m2, m3, v)
    m12 = m1 + m2
    
    p = 6.0*m1*m2*m3
    v1 = ifelse(m2 == 0.0, 1.0, m1 / m2) * (m1 / (6.0 * m3))    # change proposed by Kelli Hendricson to avoid the divided by zero issue
    v2 = v1 + (m2 - m1)/(2.0*m3)
    v3 = ifelse(
        m3 < m12, 
        (m3^2*(3.0*m12 - m3) + m1^2*(m1 - 3.0*m3) + m2^2*(m2 - 3.0*m3))/p,
        m12*0.5/m3
    )

    if v < v1
        return (p*v)^(1.0/3.0)
    elseif v < v2
        return 0.5*(m1 + sqrt(m1^2 + 8.0*m2*m3*(v - v1)))
    elseif v < v3
        c0 = m1^3 + m2^3 - p*v
        c1 = -3.0*(m1^2 + m2^2)
        c2 = 3.0*m12
        c3 = -1
        return proot(c0, c1, c2, c3)
    elseif m3 < m12
        c0 = m1^3 + m2^3 + m3^3 - p*v
        c1 = -3.0*(m1^2 + m2^2 + m3^2)
        c2 = 3
        c3 = -2
        return proot(c0, c1, c2, c3)
    else
        return m3*v + m12*0.5
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
function proot(c0, c1, c2, c3)
    a0 = c0/c3
    a1 = c1/c3
    a2 = c2/c3

    p0 = a1/3.0 - a2^2/9.0
    q0 = (a1*a2 - 3.0*a0)/6.0 - a2^3/27.0
    a = q0/sqrt(-p0^3)
    t = acos(ifelse(abs2(a)<=1,a,0))/3.0

    return sqrt(-p0)*(sqrt(3.0)*sin(t) - cos(t)) - a2/3.0
end

"""
    sort3(a, b, c)

Sort three numbers with bubble sort algorithm to avoid too much memory assignment due to array creation.
see https://codereview.stackexchange.com/a/91920
"""
function sort3(a, b, c)
    if (a>c) a,c = c,a end
    if (a>b) a,b = b,a end
    if (b>c) b,c = c,b end
    return a,b,c
end