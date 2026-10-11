module IntfAdvAMDGPUExt

using AMDGPU

using Printf

"""
    __init__()

Asserts AMDGPU is functional when loading this extension.
"""
__init__() = @assert AMDGPU.functional()

import InterfaceAdvection: _scalar_op

function _scalar_op(op::F, ::AMDGPU.ROCArray) where {F<:Function}
    AMDGPU.@allowscalar op()
end

end # module
