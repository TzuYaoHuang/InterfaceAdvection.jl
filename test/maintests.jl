
backend != "KernelAbstractions" && throw(ArgumentError("SIMD backend not allowed to run main tests, use KernelAbstractions backend"))
# TODO: @info "Test backends: $(join(arrays,", "))"

@testset "util.jl" begin
    import InterfaceAdvection: myArgAbsMax,δd,getXdir,getXYdir
    @test myArgAbsMax(SA[1/2, -2/3, 1/3]) == 2
    @test myArgAbsMax(SA[1/2  -2/3  1/3],CartesianIndex(1)) == 2

    I = CartesianIndex((1,1,1))
    @test δd(2,I) == CartesianIndex((0,1,0))
    @test δd(-3,I) == CartesianIndex((0,0,-1))

    @test getXdir(2) == 1
    @test getXdir(-1) == 2
    @test getXdir(-2) == -1
    @test getXdir(1) == -2

    @test getXYdir(3) == (1,2)
    @test getXYdir(-3) == (-1,2)
    @test getXYdir(-1) == (-2,3)
    @test getXYdir(1) == (2,3)
end

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
    import InterfaceAdvection: get3CellHeight,getρ,getμ,f2face!,BCf!,BCv!,BCv1D!
    Ng = (3,3)
    Ic = CartesianIndex(2,2)
    Iur= CartesianIndex(Ng)
    f = zeros(Ng); f[Ic] = 0.32; f[Ic+δ(2,Ic)] = 0.64

    @test containInterface(f[Ic])
    @test get3CellHeight(f,Ic,2) ≈ 0.96
    @test getρ(Ic,f,0.7) ≈ 0.796
    @test getρ(2,Ic,f,0.7) ≈ 0.748

    # getμ(i,j,I,fFace,λμ,μ,λρ) reads volume fraction interpolated to cell faces (fFace),
    # not the cell-centered field directly (i==j -> cell-normal viscosity, i≠j -> edge/vertex viscosity)
    fFace = zeros(3,3,2)
    fFace[3,2,1] = 0.1; fFace[3,3,1] = 0.2; fFace[2,3,2] = 0.3; fFace[3,3,2] = 0.4
    @test getμ(1,1,Iur,fFace,0.1,0.2,1) ≈ 0.02
    @test getμ(1,2,Iur,fFace,0.1,0.2,0.2) == getμ(2,1,Iur,fFace,0.1,0.2,0.2) ≈ 0.028

    # f2face! interpolates a cell-centered field to each of its D face directions
    fCen = zeros(4,4); fCen[2:3,2:3] .= [0.2 0.6; 0.4 0.8]
    fFace = zeros(4,4,2)
    f2face!(fFace,fCen)
    @test fFace[3,3,1] ≈ (fCen[2,3]+fCen[3,3])/2 ≈ 0.7
    @test fFace[3,3,2] ≈ (fCen[3,2]+fCen[3,3])/2 ≈ 0.6

    # --- Boundary conditions ---
    # Neumann (perdir=()): ghost cells mirror their nearest interior neighbor.
    # Periodic (j∈perdir): ghost cells wrap to the opposite side of the domain.
    Ngbc = (6,6)
    g = rand(Ngbc...)
    gN = copy(g); BCf!(gN)
    @test gN[1,:] == gN[2,:] && gN[end,:] == gN[end-1,:]
    @test gN[:,1] == gN[:,2] && gN[:,end] == gN[:,end-1]
    gP = copy(g); BCf!(gP;perdir=(1,2))
    @test gP[1,:] == gP[end-1,:] && gP[end,:] == gP[2,:]
    @test gP[:,1] == gP[:,end-1] && gP[:,end] == gP[:,2]

    # BCf!(d,f;perdir): a d-face scalar. On the non-periodic dimension d itself, only the
    # low boundary gets the special extrapolation f[I+2δ(d,I)]; the high boundary is left
    # untouched (it is not a "ghost" for this staggered storage). Other non-periodic
    # dimensions get plain Neumann; a periodic dimension still wraps, taking priority over
    # the d-special rule.
    g1 = copy(g); BCf!(1,g1)
    @test g1[1,2:end-1] == g[3,2:end-1]
    @test g1[end,2:end-1] == g[end,2:end-1]
    @test g1[:,1] == g1[:,2] && g1[:,end] == g1[:,end-1]
    g1p = copy(g); BCf!(1,g1p;perdir=(1,))
    @test g1p[1,:] == g1p[end-1,:] && g1p[end,:] == g1p[2,:]

    # BCv!/BCv1D!: a vector field gets the same d-special treatment component-by-component,
    # each component's own direction being its "staggered/normal" direction.
    u = rand(Ngbc...,2)
    uN = copy(u); BCv!(uN)
    @test uN[1,2:end-1,1] == u[3,2:end-1,1]
    @test uN[end,2:end-1,1] == u[end,2:end-1,1]
    @test uN[:,1,1] == uN[:,2,1] && uN[:,end,1] == uN[:,end-1,1]
    @test uN[2:end-1,1,2] == u[2:end-1,3,2]
    @test uN[2:end-1,end,2] == u[2:end-1,end,2]
    @test uN[1,:,2] == uN[2,:,2] && uN[end,:,2] == uN[end-1,:,2]

    u1 = copy(u); BCv1D!(view(u1,:,:,1),1) # single-component BC matches BCv!'s component 1
    @test u1[:,:,1] == uN[:,:,1]

    # BCVOF!: f follows BCf!'s convention in both branches, but α,n̂ are only extended on
    # periodic dimensions -- their Neumann-side ghosts are left untouched by design.
    f = rand(Ngbc...); α = rand(Ngbc...); n̂ = rand(Ngbc...,2)
    α0, n0 = copy(α), copy(n̂)
    BCVOF!(f,α,n̂)
    @test f[1,:] == f[2,:] && f[end,:] == f[end-1,:]
    @test α[1,:] == α0[1,:] && n̂[1,:,:] == n0[1,:,:]

    f = rand(Ngbc...); α = rand(Ngbc...); n̂ = rand(Ngbc...,2)
    BCVOF!(f,α,n̂;perdir=(1,2))
    @test f[1,:] == f[end-1,:]
    @test α[1,:] == α[end-1,:]
    @test n̂[1,:,:] == n̂[end-1,:,:]

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
    I = CartesianIndex(2,2)
    getInterfaceNormal_WY!(f,n̂,I)
    @test n̂[I,1] ≈ 1.; @test n̂[I,2]+0.5 ≈ 0.5
    f .= [0 1/3 1; 1/12 11/12 1; 1 1 1]
    getInterfaceNormal_WY!(f,n̂,I)
    @test n̂[I,1] ≈ -2/3; @test n̂[I,2] ≈ -1.

    f = [0 0 0; 0 0.1 0;0 0 0] .|> Float64
    n̂.=0
    getInterfaceNormal_WY!(f,n̂,I)
    @test n̂[I,1] == 1; @test n̂[I,2] == 0
    n̂.=0
    getInterfaceNormal_MYC!(f,n̂,I)
    @test n̂[I,1] == 0.5; @test n̂[I,2] == 0.5
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

@testset "surfaceTension.jl" begin
    f = [0.3 0.2 0.1 0.1 0.2 0.3 0.0 0.0;
        1.0 1.0 0.6 0.5 0.3 0.2 0.0 0.0;
        0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0]
    @test getPopinetHeight(CartesianIndex(1,5),f,2) == -0.3
    @test getPopinetHeight(CartesianIndex(2,5),f,2) == -0.9
    @test getPopinetHeight(CartesianIndex(3,5),f,2) == -1.4
    @test getCurvature(CartesianIndex(2,5),f,2) ≈ 0.0672718547928328
    # NOTE: 3D?
end

@testset "flow.jl" begin
    for mem ∈ arrays
        # With zero velocity and no forcing (viscosity/gravity/surface tension all off by
        # default), MPFMomStep! has nothing to advect: u and f must be exactly unchanged.
        sim = quiescentDropletSim(8; mem)
        f0 = copy(sim.intf.f)
        sim_step!(sim)
        @test all(iszero, sim.flow.u)
        @test sim.intf.f == f0
        @test all(isfinite, sim.flow.p)

        # Under a genuinely divergence-free, rotational flow (TGV), the conservative VOF
        # scheme should keep the dark-fluid volume constant to a tight tolerance.
        sim = TGVDropletSim(16; mem)
        # initialize the flow field
        sim_step!(sim)
        V0 = sum(@view sim.intf.f[inside(sim.intf.f)])
        for _ in 1:3
            sim_step!(sim)
        end
        @test all(isfinite, sim.flow.u)
        @test all(isfinite, sim.intf.f)
        @test sum(@view sim.intf.f[inside(sim.intf.f)]) ≈ V0 rtol=1e-4
    end
end

@testset "cVOF.jl" begin
    for mem ∈ arrays
        # Without an InterfaceSDF, the domain is entirely the dark fluid (f≡1)
        intf = cVOF((4,4); T=Float64, mem)
        @test all(==(1), intf.f)
        @test all(==(0), intf.α)
        @test all(==(0), intf.n̂)
        @test intf.perdir == ()

        # μ==0 / η==0 are sentinels for "disabled", stored as `nothing`
        intf0 = cVOF((4,4); T=Float64, mem, μ=0., η=0.)
        @test intf0.μ === nothing
        @test intf0.η === nothing
        intf1 = cVOF((4,4); T=Float64, mem, μ=0.5, λμ=0.2, λρ=0.3, η=1.5)
        @test intf1.μ == 0.5 && intf1.λμ == 0.2 && intf1.λρ == 0.3 && intf1.η == 1.5

        # An InterfaceSDF is PLIC-reconstructed into f (see the applyVOF! test above), then
        # the ghost cells get a Neumann (zero-gradient) extension of the interior
        interSDF = x -> (-x[1]-3x[2]+4.5)/√10
        intf2 = cVOF((2,2); T=Float64, mem, InterfaceSDF=interSDF, perdir=())
        fRef = [0 0 2/3 2/3; 0 0 2/3 2/3; 1/24 1/24 23/24 23/24; 1/24 1/24 23/24 23/24]
        @test intf2.f ≈ fRef
    end
end

@testset "InterfaceAdvection.jl" begin
    for mem ∈ arrays
        sim = TwoPhaseSimulation((8,8), (1.,0.), 8.; T=Float64, mem, InterfaceSDF=x->x[1]-4, perdir=(2,))
        @test sim isa TwoPhaseSimulation
        @test sim.intf isa InterfaceAdvection.cVOF
        # getproperty falls through to the wrapped WaterLily.Simulation for non-own fields
        @test sim.L == 8.
        @test sim.U == 1.
        @test sim.flow isa WaterLily.Flow
        @test sim.body isa WaterLily.NoBody

        # setproperty! must fall through the same way, and not recurse on `sim`'s own fields
        sim.ϵ = 2.
        @test sim.sim.ϵ == 2.
        newintf = cVOF((8,8); T=Float64, mem)
        sim.intf = newintf
        @test sim.intf === newintf
    end
end

@testset "redistaning.jl" begin
    import InterfaceAdvection: redistaning, _redistaningStage!, inside

    # redistaning(I,ϕ) = sign(ϕ[I])*(1-|𝛁ϕ|), 𝛁ϕ central-differenced over 2 cells.
    # A 1D ramp of slope 2 (uniform in the other dim) gives an exact, closed-form gradient.
    ϕ = Float64[2*(i-3) for i∈1:5, j∈1:5]
    @test redistaning(CartesianIndex(4,3),ϕ,ϕ) ≈ -1.0
    # A diagonal plane ϕ=i+j has |𝛁ϕ|=√2 everywhere in the interior.
    ϕd = Float64[i+j for i∈1:6, j∈1:6]
    @test redistaning(CartesianIndex(3,3),ϕd,ϕd) ≈ 1-√2

    # _redistaningStage!(ϕ,ϕ⁰,ϕini,dτ,α): ϕ .= α*ϕ⁰+(1-α)*(ϕ+dτ*L(ϕ)). On a field with spatially
    # uniform L (the diagonal plane above), any consistent Runge-Kutta stage sequence with
    # weights summing correctly must integrate exactly like forward-Euler, ϕⁿ⁺¹ = ϕⁿ+dτ*L;
    # the tiny residual below is the in-place (Gauss-Seidel-like) sweep order of @loop
    # reading already-updated neighbors.
    dτ = 0.01
    L = 1-√2
    ϕ0 = copy(ϕd)
    _redistaningStage!(ϕd,ϕ0,ϕd,dτ,0.0)
    _redistaningStage!(ϕd,ϕ0,ϕd,dτ,0.75)
    _redistaningStage!(ϕd,ϕ0,ϕd,dτ,1/3)
    @test maximum(abs.(ϕd[inside(ϕd)] .- (ϕ0[inside(ϕd)] .+ dτ*L))) < 1e-3

    for mem ∈ arrays
        # LevelSet(sim) builds ϕ=2f-1, reusing sim.intf.f⁰/α's storage as ϕ/ϕ⁰.
        sim = TwoPhaseSimulation((8,8), (0.,0.), 8.; T=Float64, mem, InterfaceSDF=x->x[1]-4, perdir=(2,))
        f0 = copy(sim.intf.f)
        ls = LevelSet(sim)
        @test ls.ϕ === sim.intf.f⁰
        @test ls.ϕ⁰ === sim.intf.α
        @test ls.ϕ ≈ 2 .* f0 .- 1

        # Planar interface at x=8: after reinitialization, ϕ should be a genuine signed-distance
        # field, i.e. monotone across the interface, correctly signed, finite everywhere, still
        # periodic in y, and (within a couple of cells of the interface, where the naive
        # central-difference scheme is most accurate) close to the true distance -(x-8).
        sim = TwoPhaseSimulation((16,16), (0.,0.), 16.; T=Float64, mem, InterfaceSDF=x->x[1]-8, perdir=(2,))
        ls = LevelSet(sim)
        redistaning!(ls; d=4, dτ=0.05, perdir=(2,))
        ϕ = Array(ls.ϕ)
        @test all(isfinite, ϕ)
        @test issorted(ϕ[2:end-1,9], rev=true)
        @test ϕ[:,1] == ϕ[:,end-1] && ϕ[:,end] == ϕ[:,2]
        for ix∈8:11
            @test ϕ[ix,9] ≈ -(ix-1.5-8) atol=0.15 # NOTE: Need to make it stricter when implemented a better algorithm
        end
    end
end

@testset "metrics.jl" begin
    import InterfaceAdvection: ρkeI, ρuI, ρgh, EnsI

    u = zeros(4,4,2)
    u[2,2,1] = 1; u[3,2,1] = 2; u[2,2,2] = 3; u[2,3,2] = 4
    f = zeros(4,4); f[2,2] = 0.5
    λρ = 0.2 # getρ(I,f,λρ) = λρ+(1-λρ)f[I] = 0.6 @ I
    I = CartesianIndex(2,2)

    @test ρkeI(I,u,f,λρ) ≈ 4.5
    @test ρkeI(I,u,f,λρ,(1.,1.)) ≈ 2.1
    @test ρuI(1,I,u,f,λρ) ≈ 0.9
    @test ρuI(2,I,u,f,λρ) ≈ 2.1
    @test ρuI(1,I,u,f,λρ,(1.,1.)) ≈ 0.3

    @test ρgh(I,(0.,-1.),f,λρ,(0.,0.)) ≈ 0.3

    ω = zeros(4,4); ω[2,2] = 1; ω[3,2] = 2; ω[2,3] = 3; ω[3,3] = 4
    @test EnsI(I,ω) ≈ 3.75

    # 3D EnsI exercises WaterLily.shiftDir to pick the two directions orthogonal to `i`
    I3 = CartesianIndex(2,2,2)
    ω3 = zeros(4,4,4,3)
    ω3[2,2,2,1]=1; ω3[2,3,2,1]=2; ω3[2,2,3,1]=3; ω3[2,3,3,1]=4
    ω3[2,2,2,2]=1; ω3[2,2,3,2]=2; ω3[3,2,2,2]=3; ω3[3,2,3,2]=4
    ω3[2,2,2,3]=1; ω3[3,2,2,3]=2; ω3[2,3,2,3]=3; ω3[3,3,2,3]=4
    @test EnsI(I3,ω3) ≈ 11.25
end