### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ df7ea790-040d-11f0-2bc0-63538b86512f
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__))

	using Plots
	using Plots.PlotMeasures
	using EllipsisNotation
	using LaTeXStrings
	using CUDA

	using WaterLily
	import WaterLily: @loop, inside, apply!, ∂, div

	Pkg.add(url="https://github.com/TzuYaoHuang/InterfaceAdvection.jl", rev="demo/VOF")
	using InterfaceAdvection
	import InterfaceAdvection: ϕ, BCf!, cleanWisp!, inside_uWB
end

# ╔═╡ c3d9659e-c410-42ae-b561-a1f0d6ea875d
begin
	default()
	Plots.scalefontsizes()
	default(fontfamily="Palatino",linewidth=1.5, framestyle=:axes, label=nothing, grid=false, tick_dir=:out, size=(550,480),right_margin=5mm,left_margin=5mm,top_margin=5mm,bottom_margin=5mm,markerstrokewidth=0,markersize=8)
	Plots.scalefontsizes(1.5)
end

# ╔═╡ 2488b501-9399-4b5c-9e80-1c5707a9808c
begin
	print("Check if the activated project is correct. Shoould be like \n/PATH/TO/YOUR/DOWNLOAD/InterfaceAdvection.jl/example/Project.toml\nYours:\n")
	println(Base.ACTIVE_PROJECT[])
end

# ╔═╡ 02d24b74-93ae-4ac1-9bf6-1609e93a4aa8
print("You are running with $(Threads.nthreads()) threads")

# ╔═╡ f9d2664a-bec3-433e-a07f-f5db1cca8e7a
"""
    advectgVOF!(f, fᶠAll, α, n̂, u, Δt, c̄, ρuf,λρ; dirSplit, dilation, perdir)

This is the function for geometric VOF demonstration.
`f` is the VOF field to be advected.
`fᶠAll` is where to store face flux in all direction.
`α` and `n̂` store the intercept and normal of interface in each cell.
`u` and `Δt` pass in the velocity field and time step size
`c̄` is used to take care (de-)activation of dilation term.
`dirSplit` indicates diretional split or not.
`dilation` indicates to include dilation term or not.
`perdir` tells the function which direction(s) is periodic.
`ρuf` and `λρ` are not relevant in this demonstration. They are responsible for storing mass flux for momentum-conserving algorithm
"""
function advectgVOF!(f::AbstractArray{T,D}, fᶠAll, α, n̂, u, Δt, c̄, ρuf,λρ; dirSplit=true, dilation=true, perdir=()) where {T,D}
	tol = 10eps(eltype(f))

	# onset the dilation term depends on the option and the VOF field
	@loop c̄[I] = ifelse(f[I]<0.5,0,1)*dilation over I ∈ CartesianIndices(f)
	
	dirSplit && for d ∈ 1:D
		fᶠ = @view fᶠAll[..,1]
		reconstructInterface!(f,α,n̂;perdir)
		getVOFFlux!(fᶠ,f,α,n̂,u,u,Δt,d,ρuf,λρ)
		@loop f[I] += fᶠ[I]-fᶠ[I+δ(d,I)] + c̄[I]*∂(d,I,u)*Δt over I∈inside(f)
		cleanWisp!(f,tol)
		BCf!(f;perdir)
	end

	!dirSplit && begin 
		reconstructInterface!(f,α,n̂;perdir)
		for d ∈ 1:D
			fᶠ = @view fᶠAll[..,d]
			getVOFFlux!(fᶠ,f,α,n̂,u,u,Δt,d,ρuf,λρ)
		end
		@loop f[I] -= div(I,fᶠAll) over I∈inside(f)
		cleanWisp!(f,tol)
		BCf!(f;perdir)
	end
end

# ╔═╡ 0cec37e8-1868-4916-95e9-809c878184fc
# define the velocity
begin
	UV(i,x,t,T) = i==1 ? U(x[1],x[2],t,T) : V(x[1],x[2],t,T)
	U(x,y,t,T) = sin(π*x)^2*sin(2π*y)*cos(π*t/T)
	V(x,y,t,T) = -sin(2π*x)*sin(π*y)^2*cos(π*t/T)
end

# ╔═╡ 51d9c402-0956-42fe-a201-22a87e888a8e
function generateCoord(scalarArray::AbstractArray{T,D};normalize=1,shift=zeros(T,D)) where {T,D}
    Ng = size(scalarArray)
    N = Ng.-2
    cenTuple = ntuple((i) -> ((1:Ng[i]) .- 1.5 .- N[i]/2 .- shift[i])/normalize,D)
    edgTuple = ntuple((i) -> ((1:Ng[i]) .- 2.0 .- N[i]/2 .- shift[i])/normalize,D)
    limTuple = ntuple((i) -> ([0,N[i]] .- N[i]/2 .- shift[i])/normalize,D)

    return cenTuple,edgTuple,limTuple
end

# ╔═╡ ad2416f8-76af-42c3-adf8-fd1c21e34d3f
function plotContour!(plt,xc,yc,f;clim=(0,1),levels=[0.5],color=:Black,lw=1.5)
    clamp!(f,clim...)
    Plots.contour!(plt,xc,yc,f',levels=levels,color=color,lw=lw,clim=clim)
    return plt
end

# ╔═╡ eaddbc65-8688-433a-b13c-78c8bf61db07
function organizePlot!(plt,xlim,ylim)
    Plots.plot!(plt,xlimit=xlim,ylimit=ylim,aspect_ratio=:equal)
end

# ╔═╡ 371449c8-bd83-4c11-b48e-1f3045b7ab3c
function plotVolLoss(tArray,fList)
	Plots.plot(tArray,(fList.-fList[1])/fList[1],color=:blue)
	Plots.plot!(ylabel="Relative Volume Change",xlabel=L"t",xlimit=(0,4))
	Plots.plot!(size=(550,400))
end

# ╔═╡ caee182a-097c-49c1-98ac-1b3e83c6c8a9
function getaVOFFlux!(fᶠ,f,u,Δt,d,upwind)
	fᶠ .= 0
	@loop getaVOFFlux!(fᶠ,f,u[IFace,d]*Δt,d,IFace,upwind) over IFace∈inside_uWB(size(f),d)
end

# ╔═╡ aa826573-1d51-4d77-84cb-82bd4510b86a
function getaVOFFlux!(fᶠ,f,δl,d,IFace::CartesianIndex{D},upwind) where D
	ICell = ifelse(δl>0, IFace-δ(d,IFace), IFace)
	fᶠ[IFace] = ifelse(upwind, f[ICell]*δl, ϕ(d,IFace,f)*δl)
end

# ╔═╡ 024fbdef-babe-4ea1-b3ce-2b731b8a3122
"""
    advectaVOF!(f, fᶠAll, α, n̂, u, Δt, c̄, ρuf,λρ; dirSplit, dilation, upwind, perdir)

This is the function for vanilla algebraic VOF demonstration.
`f` is the VOF field to be advected.
`fᶠAll` is where to store face flux in all direction.
`u` and `Δt` pass in the velocity field and time step size
`c̄` is used to take care (de-)activation of dilation term.
`dirSplit` indicates diretional split or not.
`dilation` indicates to include dilation term or not.
`upwind` controls using donor-acceptor concept.
`perdir` tells the function which direction(s) is periodic.
"""
function advectaVOF!(f::AbstractArray{T,D}, fᶠAll, u, Δt, c̄; dirSplit=true, dilation=true, upwind=true, perdir=()) where {T,D}
	tol = 10eps(eltype(f))

	# onset the dilation term depends on the option and the VOF field
	@loop c̄[I] = ifelse(f[I]<0.5,0,1)*dilation over I ∈ CartesianIndices(f)
	
	dirSplit && for d ∈ 1:D
		fᶠ = @view fᶠAll[..,1]
		getaVOFFlux!(fᶠ,f,u,Δt,d,upwind)
		@loop f[I] += fᶠ[I]-fᶠ[I+δ(d,I)] + c̄[I]*∂(d,I,u)*Δt over I∈inside(f)
		cleanWisp!(f,tol)
		BCf!(f;perdir)
	end

	!dirSplit && begin 
		for d ∈ 1:D
			fᶠ = @view fᶠAll[..,d]
			getaVOFFlux!(fᶠ,f,u,Δt,d,upwind)
		end
		@loop f[I] -= div(I,fᶠAll) over I∈inside(f)
		cleanWisp!(f,tol)
		BCf!(f;perdir)
	end	
end

# ╔═╡ 363f08b8-2f23-428f-85f3-67b55c934ac2
"""
    plotEvolvingVOF(;N=32, geometric=true, dirSplit=true, dilation=true, upwind=true, CFL=0.5, PREC=Float32, arr=Array)

This is the function for general VOF demonstration.
`geometric` is to choose geometric or algebraic.
`dirSplit` indicates diretional split or not.
`dilation` indicates to include dilation term or not.
`upwind` controls using donor-acceptor concept.
`CFL` is just the time step size as we set the velocity scale and grid size to be unit.
`PREC` is the precision. (Float32, Float64 ...)
`arr` is the array type. (Array, CUDA.CuArray, ...)
"""
function plotEvolvingVOF(;N=32, geometric=true, dirSplit=true, dilation=true, upwind=true, CFL=0.5, PREC=Float32, arr=Array)

	# define time series
	δt = PREC(CFL)
    T = 4N
    tArray = 0:δt:T

	# define cVOF struct
	c = cVOF( (N,N); 
		arr, T=PREC, 
		InterfaceSDF=(x) -> √sum((x.-[0.50,0.75]*N).^2) - 0.15N
	)

	# initial the velocity field
	vel = zero(c.ρu)

	# place to store volue loss
	fList = []
    divList = []

	# for animation
	cenTuple,edgTuple,limTuple = generateCoord(c.f;normalize=N)

	c.fᶠ .= 0
	apply!((i,x)->UV(i,x/N,0,T),vel)
	@loop c.fᶠ[I] = abs(div(I,vel)) over I∈inside(c.f)
	print("The maximum divergence of the velocity field is $(maximum(c.fᶠ)).")

	# the main loop
	anim = Plots.Animation()
	for tᵢ ∈ tArray

		# record the new problem
        push!(fList,sum(c.f[inside(c.f)]))

		# update the velocity (velocity is prescribed but unsteady)
        apply!((i,x)->UV(i,x/N,tᵢ,T),vel)
		BC!(vel,[0,0],false,c.perdir)

		# the advection part
        geometric && advectgVOF!(
			c.f, c.ρu, c.α, c.n̂, vel, δt, c.c̄, c.ρuf,c.λρ; 
			dirSplit, dilation, c.perdir
		)
		!geometric && advectaVOF!(
			c.f, c.ρu, vel, δt, c.c̄; 
			dirSplit, dilation, upwind, c.perdir
		)

		# plotting to save for post-processing
        plt = Plots.plot()
        plt = plotContour!(plt, cenTuple[1], cenTuple[2], c.f |> Array, 
			clim=(0,1),color=:isoluminant_cgo_70_c39_n256, 
			levels=[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8])
        Plots.plot!(plt,cbar_title=L"f",xlabel=L"x",ylabel=L"y")
        organizePlot!(plt,limTuple[1],limTuple[2])
        frame(anim,plt)
        flush(stdout)
    end
	
	return tArray/N, anim, fList, divList
end

# ╔═╡ 6d32e55a-fb3c-4679-92b7-1b18b4c5ab53
tArray,anim,fList,dList = plotEvolvingVOF(;N=64, geometric=true, dirSplit=true, dilation=false, upwind=true, CFL=0.25, PREC=Float64, arr=Array);

# ╔═╡ d9104001-d23a-4bd3-8f30-e8ee98d4c127
mp4(anim,fps=length(tArray)÷15)

# ╔═╡ fcaaab61-37d8-40d4-af1c-ec295382ddd1
plotVolLoss(tArray,fList)

# ╔═╡ Cell order:
# ╟─df7ea790-040d-11f0-2bc0-63538b86512f
# ╟─c3d9659e-c410-42ae-b561-a1f0d6ea875d
# ╟─2488b501-9399-4b5c-9e80-1c5707a9808c
# ╟─02d24b74-93ae-4ac1-9bf6-1609e93a4aa8
# ╠═6d32e55a-fb3c-4679-92b7-1b18b4c5ab53
# ╟─d9104001-d23a-4bd3-8f30-e8ee98d4c127
# ╟─fcaaab61-37d8-40d4-af1c-ec295382ddd1
# ╟─363f08b8-2f23-428f-85f3-67b55c934ac2
# ╟─f9d2664a-bec3-433e-a07f-f5db1cca8e7a
# ╟─024fbdef-babe-4ea1-b3ce-2b731b8a3122
# ╟─0cec37e8-1868-4916-95e9-809c878184fc
# ╟─51d9c402-0956-42fe-a201-22a87e888a8e
# ╟─ad2416f8-76af-42c3-adf8-fd1c21e34d3f
# ╟─eaddbc65-8688-433a-b13c-78c8bf61db07
# ╟─371449c8-bd83-4c11-b48e-1f3045b7ab3c
# ╟─caee182a-097c-49c1-98ac-1b3e83c6c8a9
# ╟─aa826573-1d51-4d77-84cb-82bd4510b86a
