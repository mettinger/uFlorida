##
using PlotlyJS

display(PlotlyJS.plot(1:10,1:10))

##
using Distributions

probabilityDistribution = MixtureModel([Exponential(2.5), Normal(5., .5), Normal(7., .5)], [.9,.05,.05])



x = 0:.1:10
y = [pdf(probabilityDistribution, i) for i in x]
display(PlotlyJS.plot(x,y))

##
using PyCall

math = pyimport("math")
math.sin(math.pi / 4)


##
using Conda
Conda.PYTHONDIR


##
using Distributed
using LinearAlgebra

M = Matrix{Float64}[rand(1000,1000) for i = 1:100];
a = pmap(svd, M);

##
using CUDA

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
loss(x, y) # ~ 3

##

collect(Base.product(["a","b"], [3, 4]))

##

#f(;x=10, y=10, z=10) = x^2, y^2, z^2
#f(;x , y, z) = x^2, y^2, z^2

function f(;x, y, z)
    return x^2, y^2, z^2
end

d = Dict(:z=>7, :x=>4, :y=>5);

print(f(;d...))

##
addprocs(6)

##
@everywhere include("kuramotoSpatialJulia.jl")


##
pmapList = pmapListGet()
print("workers: " * string(nworkers()))
pmap(kuramotoPMAP, pmapList)