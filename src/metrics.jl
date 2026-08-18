import WaterLily: fSV,fsum

# NOTE: This should be removed once shiftdir is included
# Not yet released by WaterLily (post-v1.8.0 only), so defined locally rather than imported.
# Shift the index of direction `d` to `i` steps away in dimension space of `D`.
shiftDir(d,D,i) = mod(d+i-1,D)+1

"""
    ρkeI(I::CartesianIndex,u,U=0)

Compute ``½ρ∥𝐮-𝐔∥²`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
This function take multiphase into account so as the staggered arragement.
"""
ρkeI(I::CartesianIndex{D},u,f,λρ,U=fSV(zero,D)) where D = 0.25fsum(D) do i
    ((u[I,i]-U[i])^2+(u[I+δ(i,I),i]-U[i])^2)*getρ(I,f,λρ)
end

"""
    ρgh(I,g,f,λρ,StatWL)

Compute potential energy of a cell given gravitational field tuple `g`.
StatWL is static WL: The height of each index if evenly distribute the water -- the final water level if g acting in the direction.
"""
ρgh(I::CartesianIndex{D},g,f,λρ,StatWL) where D = -getρ(I,f,λρ)*fsum((i)-> g[i]*(loc(0,I)[i]-StatWL[i]), D)

"""
    EnsI(I::CartesianIndex,u,U=0)

Compute ``½α∥𝛚∥²`` at center of cell `I` where `ω` can be used
to subtract a background flow (by default, `U=0`).
This function take multiphase into account so as the staggered arragement.
"""
EnsI(I::CartesianIndex{3},ω) = 0.5*0.25fsum(3) do i
    ix,iy = shiftDir(i,3,1),shiftDir(i,3,2)
    ω[I,i]^2+ω[I+δ(ix,I),i]^2+ω[I+δ(iy,I),i]^2+ω[I+δ(ix,I)+δ(iy,I),i]^2
end
EnsI(I::CartesianIndex{2},ω) = 0.5*0.25*(
    ω[I]^2+ω[I+δ(1,I)]^2+ω[I+δ(2,I)]^2+ω[I+δ(1,I)+δ(2,I)]^2
)

"""
    ρuI(i,I::CartesianIndex,u,U=0)

Compute ``ρ(𝐮-𝐔)`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
This function take multiphase into account so as the staggered arragement.
"""
ρuI(i,I::CartesianIndex{D},u,f,λρ,U=fSV(zero,D)) where D = begin
    0.5(u[I,i]+u[I+δ(i,I),i]-2U[i])*getρ(I,f,λρ)
end