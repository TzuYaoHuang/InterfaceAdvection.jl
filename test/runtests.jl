using InterfaceAdvection
using WaterLily
using Test

@testset "PLIC.jl" begin
    # TODO: true run test
    @test getIntercept(1,2,3,0.1) == 2
end

@testset "InterfaceAdvection.jl" begin
    # Write your tests here.
end
