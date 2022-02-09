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
using Distributed
z = pmap(x -> x[1] + x[2], Base.product([1,2], [3, 4]))

##

collect(1.:.1:2.)

##

sizeof(collect(Base.product(["a","b"], [3, 4])))

##
f = x -> x * x
f(3)



#=
nowString = Dates.format(now(),"mm-dd-HH-MM")

layout = Layout(title="Natural frequencies" * "<br>nOsc: " * string(nOsc) * "<br>K: " * string(K) * "<br>" * distributionTitle)
p = PlotlyJS.plot(PlotlyJS.histogram(x=W), layout);
display(p)
PlotlyJS.savefig(p,  "../data/plots/natFreq_" * nowString * ".jpeg")

x = 0:.1:10
y = [pdf(probabilityDistribution, i) for i in x]
display(PlotlyJS.plot(x,y))

##
println("Number of oscillators: " * string(nOsc))
println("K: " * string(K))
println("Natural frequencies: " * string(probabilityDistribution))
println("Solving...")

prob = ODEProblem(kuramoto2d!, theta0, (0, upperTimeBound), [kernelMatrix, W]);
sol = solve(prob, method = method, reltol = 1e-8, abstol = 1e-8, saveat = saveat, progress = true, progress_steps = 10, jac=jac)
print(sol.retcode)

##
if plotFlag
    t = sol.t
    u = sol.u
    odePhi = transpose(reduce(hcat, u))
    orderParameterAbs = [orderParameterGet(odePhi[i,:]) for i in 1:length(t)]
    layout = Layout(title="Order Parameter" * "<br>nOsc: " * string(nOsc) * "<br>K: " * string(K) * "<br>" * distributionTitle)
    p = PlotlyJS.plot(t, orderParameterAbs, layout);
    display(p)
    PlotlyJS.savefig(p,  "../data/plots/orderParam_" * nowString * ".jpeg")
    
    #display(PlotlyJS.plot(PlotlyJS.scatter(x=cos.(odePhi[1000,:]), y=sin.(odePhi[1000,:]), mode="markers")))
    #display(PlotlyJS.plot(PlotlyJS.heatmap(z=kernelMatrix)))
    #display(PlotlyJS.plot(PlotlyJS.scatter(x=cos.(theta0), y=sin.(theta0), mode="markers")))
end

if saveFlag
    df = DataFrame(sol)
    CSV.write("../data/kuramoto_out_" * nowString * ".csv" , df)
end

=#