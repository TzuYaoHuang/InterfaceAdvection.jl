
"""
    cVOF{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}}

Composite type for 2D or 3D two-phase conservative Volume-of-Fluid (cVOF) advection scheme.

The dark fluid is advected using operator-split cVOF method proposed by  [Weymouth & Yue (2010)](https://doi.org/10.1016/j.jcp.2009.12.018).
This guarentees mass conservation and preserves sharp interface across fluids.
The primary variable is the volume fraction of the heavy fluid, the cell-averaged color function, `f`. 
We use Piecewise Linear Interface Calculation (PLIC) to reconstruct sharp interface. 
The dark fluid is indicated with negative distance. That is to say, the normal is pointed into the light fluid.
"""
struct cVOF{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}}

    # field variable
    f  :: Sf  # volume fraction
    f⁰ :: Sf  # volume fraction for RK2 scheme
    α  :: Sf  # intercept for PLIC
    n̂  :: Vf  # normal vector for PLIC
    fᶠ :: Sf  # store VOF flux
    c̄  :: AbstractArray{Int8} # cell-centered indicator value for dilation term

    # Varable for energy-conserving scheme
    ρu :: Vf  # momentum
    ρuf:: Vf  # mass flux from VOF advection
    ρf :: Vf
    uOld::Vf 

    # physical properties
    μ  :: Union{T,Nothing}   # store dynamcs viscosity of dark fluid (corresponding to ν)
    λρ :: T   # density ratio = light/dark fluid
    λμ :: T   # dynamic viscosity ratio = light/dark fluid
    η  :: Union{T,Nothing}   # surface tension

    # domain configuration
    perdir :: NTuple  # tuple of periodic direction

    function cVOF(
        N::NTuple{D}; 
        arr=Array, T=Float64, 
        InterfaceSDF::Function=(x)->5-x[1], 
        μ=1e-3, λμ=1e-2, λρ=1e-3, η=nothing,
        perdir=()
    ) where D

        # Declare grid size
        Ng = N.+2
        Nv = (Ng...,D)

        # Allocate essential variables
        f = ones(T,Ng) |> arr
        α = zeros(T,Ng) |> arr
        n̂ = zeros(T,Nv) |> arr
        c̄ = zeros(Int8,Ng) |> arr

        # Initialize variables
        applyVOF!(f,α,n̂,InterfaceSDF)
        BCVOF!(f,α,n̂;perdir)
        f⁰ = copy(f) |> arr
        fᶠ = zeros(T,Ng) |> arr

        # Energy conserving
        ρu = zeros(T,Nv) |> arr
        ρuf= zeros(T,Nv) |> arr
        ρf= zeros(T,Nv) |> arr
        uOld= zeros(T,Nv) |> arr

        # correct η
        ηc = ifelse(η==0,nothing,η)
        μc = ifelse(μ==0,nothing,μ)

        new{D,T,typeof(f),typeof(n̂)}(
            f, f⁰, α, n̂, fᶠ, c̄,
            ρu, ρuf, ρf, uOld,
            μc, λρ, λμ, ηc,
            perdir
        )
    end
end