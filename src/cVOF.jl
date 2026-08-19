
"""
    cVOF{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}}

Composite type for 2D or 3D two-phase conservative Volume-of-Fluid (cVOF) advection scheme.

The dark fluid is advected using operator-split cVOF method proposed by  [Weymouth & Yue (2010)](https://doi.org/10.1016/j.jcp.2009.12.018).
This guarentees mass conservation and preserves sharp interface across fluids.
The primary variable is the volume fraction of the heavy fluid, the cell-averaged color function, `f`.
We use Piecewise Linear Interface Calculation (PLIC) to reconstruct sharp interface.
The dark fluid is indicated with negative distance. That is to say, the normal is pointed into the light fluid.

  - `normalScheme`: Interface-normal reconstruction scheme `normalScheme(f,n̂,I)` called by
        `reconstructInterface!`, e.g. `getInterfaceNormal_WH!` (default), `getInterfaceNormal_WY!`,
        `getInterfaceNormal_MYC!`, `getInterfaceNormal_Y!`, `getInterfaceNormal_Column!`,
        `getInterfaceNormal_PCD!`, or `getInterfaceNormal_SLIC!`. See `normalEstimation.jl`.
"""
struct cVOF{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Nf}

    # field variables
    f  :: Sf  # volume fraction
    f⁰ :: Sf  # volume fraction for RK2 scheme
    α  :: Sf  # intercept for PLIC
    n̂  :: Vf  # normal vector for PLIC
    normalScheme :: Nf # interface-normal reconstruction scheme normalScheme(f,n̂,I); a type parameter so reconstructInterface! specializes on it
    fᶠ :: Sf  # store VOF flux
    c̄  :: AbstractArray{Int8} # cell-centered indicator value for dilation term

    # Varables for CMOM
    ρu :: Vf  # momentum
    ρuf:: Vf  # mass flux from VOF advection

    # Originally for interface-aware Flux limiter but now a vector buffer
    dρ :: Vf # face-center density change indicator

    # physical properties
    μ  :: Union{T,Nothing}   # store dynamcs viscosity of dark fluid (corresponding to ν)
    λρ :: T   # density ratio = light/dark fluid
    λμ :: T   # dynamic viscosity ratio = light/dark fluid
    η  :: Union{T,Nothing}   # surface tension

    # domain configuration
    perdir :: NTuple  # tuple of periodic direction

    function cVOF(N::NTuple{D};
                mem=Array, T=Float32,
                InterfaceSDF=nothing,
                μ=1e-3, λμ=1e-2, λρ=1e-3, η=nothing,
                normalScheme=getInterfaceNormal_WH!,
                perdir=()
    ) where D

        # Declare grid size
        Ng = N.+2
        Nv = (Ng...,D)

        # Allocate essential variables
        f = ones(T,Ng) |> mem
        α = zeros(T,Ng) |> mem
        n̂ = zeros(T,Nv) |> mem
        c̄ = zeros(Int8,Ng) |> mem

        # Initialize variables
        applyVOF!(f,α,n̂,InterfaceSDF); !isnothing(InterfaceSDF) && BCf!(f;perdir)
        f⁰ = copy(f) |> mem
        fᶠ = zeros(T,Ng) |> mem

        # CMOM
        ρu = zeros(T,Nv) |> mem
        ρuf= zeros(T,Nv) |> mem

        # Yet another vecotr variable for starage purpose
        # originally for density ratio
        dρ = ones(T,Nv) |> mem

        # correct η
        ηc = ifelse(η==0,nothing,η)
        μc = ifelse(μ==0,nothing,μ)

        println("μ: $(μc), λρ: $(λρ)")

        new{D,T,typeof(f),typeof(n̂),typeof(normalScheme)}(
            f, f⁰, α, n̂, normalScheme, fᶠ, c̄,
            ρu, ρuf,
            dρ,
            μc, λρ, λμ, ηc,
            perdir
        )
    end
end
