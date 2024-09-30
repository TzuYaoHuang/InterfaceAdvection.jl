import WaterLily: fSV,fsum

"""
    ρkeI(I::CartesianIndex,u,U=0)

Compute ``½ρ∥𝐮-𝐔∥²`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
This function take multiphase into account so as the staggered arragement.
"""
ρkeI(I::CartesianIndex{D},u,f,λρ,U=fSV(zero,D)) where D = 0.25fsum(D) do i
    ((u[I,i]-U[i])^2+(u[I+δ(i,I),i]-U[i])^2)*(f[I]*(1-λρ) + λρ)
end
"""
    EnsI(I::CartesianIndex,u,U=0)

Compute ``½α∥𝛚∥²`` at center of cell `I` where `ω` can be used
to subtract a background flow (by default, `U=0`).
This function take multiphase into account so as the staggered arragement.
"""
EnsI(I::CartesianIndex{D},ω) where D = 0.5*0.25fsum(D) do i
    ix,iy = (1,2)
    ω[I,i]^2+ω[I+δ(ix,I),i]^2+ω[I+δ(iy,I),i]^2+ω[I+δ(ix,I)+δ(iy,I),i]^2
end
EnsI2D(I::CartesianIndex{D},ω) where D = 0.5*0.25*(
    ω[I]^2+ω[I+δ(1,I)]^2+ω[I+δ(2,I)]^2+ω[I+δ(1,I)+δ(2,I)]^2
)
"""
    ρuI(i,I::CartesianIndex,u,U=0)

Compute ``ρ(𝐮-𝐔)`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
This function take multiphase into account so as the staggered arragement.
"""
ρuI(i,I::CartesianIndex{D},u,f,λρ,U=fSV(zero,D)) where D = begin
    0.5(u[I,i]+u[I+δ(i,I),i]-2U[i])*(f[I]*(1-λρ) + λρ)
end

getAnotherDir(d,n) = filter(i-> i≠d,(1:n...,))