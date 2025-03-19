### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ df7ea790-040d-11f0-2bc0-63538b86512f
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__))
	Pkg.add(url="https://github.com/TzuYaoHuang/InterfaceAdvection.jl", rev="demo/VOF")
	print("\nCheck if the project is correct:")
	println(Base.ACTIVE_PROJECT[])
end

# ╔═╡ Cell order:
# ╠═df7ea790-040d-11f0-2bc0-63538b86512f
