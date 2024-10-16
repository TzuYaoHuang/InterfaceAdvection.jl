using InterfaceAdvection
using WaterLily
using StaticArrays

using Test

@testset "PLIC.jl" begin
    TList = [Float32,Float64]
    for T∈TList
        @test getIntercept(T(2/3),T(4/3),T(0),T(5/12)) ≈ getIntercept(T(2/3),T(4/3),T(5/12)) ≈ T(8/9)
        @test getIntercept(T(2/3),T(0),T(4/3),T(5/12)) ≈ getIntercept(T(2/3),T(4/3),T(5/12)) ≈ T(8/9)
        @test getIntercept(T(2/3),-T(4/3),T(0),T(5/12)) ≈ getIntercept(T(2/3),-T(4/3),T(5/12)) ≈ -T(4/9)
        @test getIntercept(T(3),-T(4),T(0),T(5/6)) ≈ getIntercept(T(3),-T(4),T(5/6)) ≈ T(1)
        @test getIntercept(T(1/2),T(1/3),T(1),T(7/12)) ≈ 1
        @test getIntercept(T(1),T(1),-T(1),T(1-1/48)) ≈ T(3/2)
        n̂2D = zeros(T,1,1,2); n̂2D[1,1,:] .= (-T(7/6),T(4/9))
        @test getIntercept(n̂2D,CartesianIndex(1,1),T(7/23)) == getIntercept(SA[-T(7/6),T(4/9)],T(7/23)) ≈ getIntercept(-T(7/6),T(4/9),T(0),T(7/23))
        n̂3D = zeros(T,1,1,1,3); n̂3D[1,1,1,:] .= (T(7/6),T(4/9),-T(29/97))
        @test getIntercept(n̂3D,CartesianIndex(1,1,1),T(7/23)) == getIntercept(T(7/6),T(4/9),-T(29/97),T(7/23)) ≈ getIntercept(SA[T(7/6),T(4/9),-T(29/97)],T(7/23))

        @test getVolumeFraction(T(2/3),T(4/3),T(0),T(8/9)) ≈ getVolumeFraction(T(2/3),T(4/3),T(8/9)) ≈ T(5/12)
        @test getVolumeFraction(T(2/3),T(0),T(4/3),T(8/9)) ≈ getVolumeFraction(T(2/3),T(4/3),T(8/9)) ≈ T(5/12)
        @test getVolumeFraction(T(2/3),-T(4/3),T(0),-T(4/9)) ≈ getVolumeFraction(T(2/3),-T(4/3),-T(4/9)) ≈ T(5/12)
        @test getVolumeFraction(T(3),-T(4),T(0),T(1)) ≈ getVolumeFraction(T(3),-T(4),T(1)) ≈ T(5/6)
        @test getVolumeFraction(T(1/2),T(1/3),T(1),T(1)) ≈ T(7/12)
        @test getVolumeFraction(T(1),T(1),-T(1),T(3/2)) ≈ T(1-1/48)
        n̂2D = zeros(T,1,1,2); n̂2D[1,1,:] .= (-T(7/6),T(4/9))
        @test getVolumeFraction(n̂2D,CartesianIndex(1,1),T(7/23)) == getVolumeFraction(SA[-T(7/6),T(4/9)],T(7/23)) ≈ getVolumeFraction(-T(7/6),T(4/9),T(0),T(7/23)) 
        n̂3D = zeros(T,1,1,1,3); n̂3D[1,1,1,:] .= (T(7/6),T(4/9),-T(29/97))
        @test getVolumeFraction(n̂3D,CartesianIndex(1,1,1),T(7/23)) == getVolumeFraction(T(7/6),T(4/9),-T(29/97),T(7/23)) == getVolumeFraction(SA[T(7/6),T(4/9),-T(29/97)],T(7/23))
    end

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

@testset "advection.jl" begin
    Ng = (3,3); Nv = (Ng...,2);
    Ic = CartesianIndex(2,2)
    f = zeros(Ng); f[Ic] = 0.32
    α = zeros(Ng); α[Ic] = -0.2
    n̂ = zeros(Nv); n̂[Ic,:] .= [1,-1]
    ρuf = zeros(Nv); λρ = 0.1
    fᶠ= zeros(Ng)
    d = 1
    getVOFFlux!(fᶠ,f,α,n̂,-0.4,d,Ic,ρuf,λρ)
    getVOFFlux!(fᶠ,f,α,n̂,0.4,d,Ic+δ(d,Ic),ρuf,λρ)
    @test fᶠ[Ic] ≈ -0.24
    @test fᶠ[Ic+δ(d,Ic)] ≈ 0.02
    @test ρuf[Ic,d] ≈ -0.256
    @test ρuf[Ic+δ(d,Ic),d] ≈ 0.058
    d = 2
    getVOFFlux!(fᶠ,f,α,n̂,-0.4,d,Ic,ρuf,λρ)
    getVOFFlux!(fᶠ,f,α,n̂,0.4,d,Ic+δ(d,Ic),ρuf,λρ)
    @test fᶠ[Ic] ≈ -0.02
    @test fᶠ[Ic+δ(d,Ic)] ≈ 0.24
    @test ρuf[Ic,d] ≈ -0.058
    @test ρuf[Ic+δ(d,Ic),d] ≈ 0.256
end

@testset "VOFutil.jl" begin
    import InterfaceAdvection: get3CellHeight,getρ,getμ
    Ng = (3,3)
    Ic = CartesianIndex(2,2)
    Iur= CartesianIndex(Ng)
    f = zeros(Ng); f[Ic] = 0.32; f[Ic+δ(2,Ic)] = 0.64

    @test containInterface(f[Ic])
    @test get3CellHeight(f,Ic,2) ≈ 0.96
    @test getρ(Ic,f,0.7) ≈ 0.796
    @test getρ(2,Ic,f,0.7) ≈ 0.748
    @test getμ(Val{1==1}(),1,1,Iur,f,0.1,0.2,1) ≈ 0.1352
    @test getμ(Val{1==2}(),1,2,Iur,f,0.1,0.2,0.2) == getμ(Val{1==2}(),2,1,Iur,f,0.1,0.2,0.2) ≈ 0.02
    # TODO: BCVOF!
end

@testset "InterfaceAdvection.jl" begin
    # Write your tests here.
end
