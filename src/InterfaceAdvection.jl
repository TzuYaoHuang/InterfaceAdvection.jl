module InterfaceAdvection
using WaterLily

include("util.jl")

include("VOFutil.jl")
export applyVOF!

include("PLIC.jl")
export getIntercept, getVolumeFraction

include("surfaceTension.jl")

include("cVOF.jl")

include("flow.jl")

end
