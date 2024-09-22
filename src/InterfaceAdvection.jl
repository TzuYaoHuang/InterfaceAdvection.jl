module InterfaceAdvection
using WaterLily

include("util.jl")

include("VOFutil.jl")
export applyVOF!, BCVOF!

include("normalEstimation.jl")
export reconstructInterface!, getInterfaceNormal_WY!, getInterfaceNormal_PCD!

include("PLIC.jl")
export getIntercept, getVolumeFraction

include("surfaceTension.jl")

include("cVOF.jl")

include("flow.jl")

end
