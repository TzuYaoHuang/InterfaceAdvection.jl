module IntfAdvCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using Printf
import WaterLily: div,δ
import InterfaceAdvection: reportFillError

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()


function reportFillError(f::CuArray{T,D},u,u⁰,δt,d,tol) where {T,D}
    maxf, maxid = findmax(f)
    minf, minid = findmin(f)
    if maxf-1 > tol
        CUDA.@allowscalar du⁰,du = abs(div(maxid,u⁰)),abs(div(maxid,u))
        @printf("|∇⋅u⁰| = %+13.8f, |∇⋅u| = %+13.8f\n",du⁰,du)
        for d∈1:D
            CUDA.@allowscalar @printf("    %d -- uLeftδt: %+13.8f, uRightδt: %+13.8f\n", d, (u[maxid,d]+u⁰[maxid,d])*δt, (u[maxid+δ(d,maxid),d]+u⁰[maxid+δ(d,maxid),d])*δt)
        end
        errorMsg = "max VOF @ $(maxid.I) ∉ [0,1] @ direction $d, Δf = $(maxf-1)"
        (du⁰+du > 10) && error("divergence, $(du⁰+du), is exploding!")
        try
            error(errorMsg)
        catch e
            Base.printstyled("ERROR: "; color=:red, bold=true)
            Base.showerror(stdout, e, Base.catch_backtrace()); println()
        end
    end
    if minf < -tol
        CUDA.@allowscalar du⁰,du = abs(div(minid,u⁰)),abs(div(minid,u))
        @printf("|∇⋅u⁰| = %+13.8f, |∇⋅u| = %+13.8f\n",du⁰,du)
        for d∈1:D
            CUDA.@allowscalar @printf("    %d -- uLeftδt: %+13.8f, uRightδt: %+13.8f\n", d, (u[minid,d]+u⁰[minid,d])*δt, (u[minid+δ(d,minid),d]+u⁰[minid+δ(d,minid),d])*δt)
        end
        errorMsg = "min VOF @ $(minid.I) ∉ [0,1] @ direction $d, Δf = $(-minf)"
        (du⁰+du > 10) && error("divergence, $(du⁰+du), is exploding!")
        try
            error(errorMsg)
        catch e
            Base.printstyled("ERROR: "; color=:red, bold=true)
            Base.showerror(stdout, e, Base.catch_backtrace()); println()
        end
    end
end

end # module
