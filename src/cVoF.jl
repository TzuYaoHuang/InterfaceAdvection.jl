
"""
    cVOF{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}}

Composite type for 2D or 3D two-phase conservative Volume-of-Fluid (cVoF) advection scheme.

The dark fluid is advected using operator-split cVoF method proposed by  [Weymouth & Yue (2010)](https://doi.org/10.1016/j.jcp.2009.12.018).
This guarentees mass conservation and preserves sharp interface across fluids.
The primary variable is the volume fraction of the heavy fluid, the cell-averaged color function, `f`. 
We use Piecewise Linear Interface Calculation (PLIC) to reconstruct sharp interface. 
The dark fluid is indicated with negative distance.
"""
struct cVoF{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray}
    # field variable
    f  :: Sf  # volume fraction
    f⁰ :: Sf  # volume fraction for RK2 scheme
    α  :: Sf  # intercept for PLIC
    n̂  :: Vf  # normal vector for PLIC
    c̄  :: AbstractArray{Int8} # cell-centered indicator value for dilation term
    # physical variable
    λρ :: T   # density ratio = light/dark fluid
    λμ :: T   # dynamic viscosity ratio = light/dark fluid
    η  :: T   # surface tension
    # domain configuration
    perdir :: NTuple
    dirdir :: NTuple
    function cVoF(
        N::NTuple{D}; 
        arr=Array, T=Float64, 
        InterfaceSDF::Function=(x)->5-x[1], 
        λμ=1e-2, λρ=1e-3, η=0,
        perdir=(), dirdir=()
    ) where D
        # TODO: initialize
    end
end