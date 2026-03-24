module IntfAdvCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using Printf
import WaterLily: div,δ

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

import InterfaceAdvection: _scalar_op

function _scalar_op(op::F, ::CUDA.CuArray) where {F<:Function}
    CUDA.@allowscalar op()
end

end # module
