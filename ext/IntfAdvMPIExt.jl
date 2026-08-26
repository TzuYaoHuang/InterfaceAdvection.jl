module IntfAdvMPIExt

using WaterLily
using MPI, ImplicitGlobalGrid

# WaterLily's `@loop` macro eagerly computes an MPI rank-local coordinate offset from
# the loop array's own `eltype`, regardless of whether the loop body actually calls
# `loc(...)`. `WaterLily._loop_offset` diverts `Bool` arrays to `Float32` for this
# (a narrow eltype can't hold an offset like `rank*nx_loc`, and `loc(i,I,T)`'s own
# arithmetic needs a float `T` anyway), but doesn't cover other narrow integer types.
# `cVOF.c̄` (the dilation-term indicator in cVOF.jl) is `Int8`, which hits the exact
# same overflow — so add that missing case here rather than widening WaterLily's own
# `Bool`-only method.
#
# Dispatch must be on the *concrete* `Parallel` type WaterLilyMPIExt defines (not
# exported, so fetched via `Base.get_extension`): a method on the abstract
# `WaterLily.AbstractParMode` would be exactly as specific as WaterLilyMPIExt's own
# `(::Type{T}, ::Parallel) where T` in the second argument while more specific in the
# first (`T<:Integer` vs unconstrained `T`) — that's an *ambiguity*, not an override.
# Matching `::Parallel` exactly resolves it in our favor.
#
# The lookup has to happen in `__init__`, not at module top-level: when Pkg precompiles
# *this* extension, it loads only its own declared triggers (`MPI`, `ImplicitGlobalGrid`)
# plus `InterfaceAdvection`'s own deps — it does not also sweep and activate WaterLily's
# unrelated `WaterLilyMPIExt`, so `Parallel` genuinely doesn't exist yet in that isolated
# process. `__init__` reruns in every real session, where loading `WaterLily`
# (InterfaceAdvection's hard dependency) alongside `MPI`+`ImplicitGlobalGrid` does activate
# `WaterLilyMPIExt` first, so the lookup succeeds there.
function __init__()
    ext = Base.get_extension(WaterLily, :WaterLilyMPIExt)
    isnothing(ext) && return   # precompiling this extension in isolation — nothing to patch yet
    # `@eval` (not a plain `function` def) because a global method definition can't be
    # nested inside a function body; `$(ext.Parallel)` splices the runtime-resolved type
    # into the signature before the method gets installed on WaterLily's method table.
    @eval WaterLily._loop_offset(::Type{T}, p::$(ext.Parallel)) where T<:Integer =
        WaterLily._loop_offset(Float32, p)  # reuses WaterLilyMPIExt's own Float32 path
    nothing
end

end # module
