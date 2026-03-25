using BenchmarkTools, Printf

backend != "SIMD" && throw(ArgumentError("KernelAbstractions backend not allowed to run allocations tests, use SIMD backend"))
@testset "mom_step! allocations" begin
    function TGVDroplet3D(N; Re=Inf, We=Inf,perdir=(1,2,3))
        T=Float32
        NN = (N,N,N)
        U = 1
        R = T(N/4)
        ν = T(U*N/Re)
        η = T(U^2*2R/We)
        zeroT = T(0)
        κ = T(π/N)
        pertT = T(0.8)

        function uλ(i,xyz)
            x,y,z = @. (2xyz - N)*κ                # scaled coordinates
            i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
            i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
            return zeroT                              # u_z
        end

        # Interface function
        function Inter(xyz)
            x,y,z = @. (xyz-N÷2)
            return (√(x^2+y^2+z^2) - R)
        end

        return TwoPhaseSimulation(
            NN, (0, 0, 0), R;
            U, Δt=0.01, ν=ν, InterfaceSDF=Inter, T, uλ, perdir, η
        )
    end

    sim = TGVDroplet3D(32)
    sim_step!(sim) # runs with λ=quick
    b = @benchmarkable MPFMomStep!($sim.flow,$sim.pois,$sim.intf,$sim.body) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.2f", r.memory/1e3)*" KiB")
    @test r.memory < 50000 

    sim = TGVDroplet3D(32;Re=100)
    sim_step!(sim) # runs with finite viscosity
    b = @benchmarkable MPFMomStep!($sim.flow,$sim.pois,$sim.intf,$sim.body) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.2f", r.memory/1e3)*" KiB")
    @test r.memory < 50000

    sim = TGVDroplet3D(32;We=100)
    sim_step!(sim) # runs with finite surface tension
    b = @benchmarkable MPFMomStep!($sim.flow,$sim.pois,$sim.intf,$sim.body) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.2f", r.memory/1e3)*" KiB")
    @test r.memory < 50000
end