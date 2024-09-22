using InterfaceAdvection
using WaterLily
using StaticArrays

using Test

@testset "PLIC.jl" begin
    @test getIntercept(2/3,4/3,0,5/12) ≈ 8/9
    @test getIntercept(2/3,0,4/3,5/12) ≈ 8/9
    @test getIntercept(2/3,-4/3,0,5/12) ≈ -4/9
    @test getIntercept(3,-4,0,5/6) ≈ 1
    @test getIntercept(1/2,1/3,1,7/12) ≈ 1
    @test getIntercept(1,1,-1,1-1/48) ≈ 3/2
    n̂2D = zeros(1,1,2); n̂2D[1,1,:] .= (-7/6,4/9)
    @test getIntercept(n̂2D,CartesianIndex(1,1),7/23) == getIntercept(-7/6,4/9,0,7/23) == getIntercept(SA[-7/6,4/9],7/23)
    n̂3D = zeros(1,1,1,3); n̂3D[1,1,1,:] .= (7/6,4/9,-29/97)
    @test getIntercept(n̂3D,CartesianIndex(1,1,1),7/23) == getIntercept(7/6,4/9,-29/97,7/23) == getIntercept(SA[7/6,4/9,-29/97],7/23)

    @test getVolumeFraction(2/3,4/3,0,8/9) ≈ 5/12
    @test getVolumeFraction(2/3,0,4/3,8/9) ≈ 5/12
    @test getVolumeFraction(2/3,-4/3,0,-4/9) ≈ 5/12
    @test getVolumeFraction(3,-4,0,1) ≈ 5/6
    @test getVolumeFraction(1/2,1/3,1,1) ≈ 7/12
    @test getVolumeFraction(1,1,-1,3/2) ≈ 1-1/48
    n̂2D = zeros(1,1,2); n̂2D[1,1,:] .= (-7/6,4/9)
    @test getVolumeFraction(n̂2D,CartesianIndex(1,1),7/23) == getVolumeFraction(-7/6,4/9,0,7/23) == getVolumeFraction(SA[-7/6,4/9],7/23)
    n̂3D = zeros(1,1,1,3); n̂3D[1,1,1,:] .= (7/6,4/9,-29/97)
    @test getVolumeFraction(n̂3D,CartesianIndex(1,1,1),7/23) == getVolumeFraction(7/6,4/9,-29/97,7/23) == getVolumeFraction(SA[7/6,4/9,-29/97],7/23)

end

@testset "VOFutil.jl" begin
    N = (2,2)
    f = zeros(N.+2); α = similar(f); n̂ = zeros((N.+2 ...,2))
    interSDF=(x) -> (-x[1]-3x[2]+4.5)/√10
    fRef = [0 0 0 0; 0 0 2/3 0; 0 1/24 23/24 0; 0 0 0 0]
    applyVOF!(f,α,n̂,interSDF)
    @test f ≈ fRef
end

@testset "normalEstimation.jl" begin
    f = [5/12 1 2/3; 1/4 11/12 1/12; 1/12 1/3 0]
    n̂ = zeros(3,3,2)
    getInterfaceNormal_WY!(f,n̂,CartesianIndex(2,2))
    @test n̂[2,2,1] ≈ 1.; @test n̂[2,2,2]+0.5 ≈ 0.5
    f .= [0 1/3 1; 1/12 11/12 1; 1 1 1]
    getInterfaceNormal_WY!(f,n̂,CartesianIndex(2,2))
    @test n̂[2,2,1] ≈ -2/3; @test n̂[2,2,2] ≈ -1.
end

@testset "InterfaceAdvection.jl" begin
    # Write your tests here.
end
