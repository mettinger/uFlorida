using Distributed
using Dates

##

const nProcs = 6
addprocs(nProcs)
@everywhere include("kuramoto.jl")

#nowString = Dates.format(now(),"mm-dd-HH-MM")

pmapList = pmapListGet(0)

println("Workers: " * string(nProcs))
println("Trials: " * string(length(pmapList)))
pmap(kuramotoPMAP, pmapList)


# PARAMETERS TO SET
##

#=
const kernelSwitch = 0
if kernelSwitch == 0
    const nOsc = 1024 # should be perfect square
    const K = 2
    const kernelMatrix = (K/nOsc) * ones((nOsc, nOsc))
elseif kernelSwitch == 1
    const nOsc = 1024 # should be perfect square
    const K = 4 
    const distanceMatrix = distanceMatrixGet(nOsc)
    const kernelMatrix = (K/nOsc) * kernelMatrixGet(distanceMatrix)
elseif kernelSwitch == 2
    const distanceMatrix = Matrix{Float64}(DataFrame(CSV.File("distanceMatrix.csv", header=false)))
    const nOsc = size(distanceMatrix)[1]
    const K = K_nOsc_Ratio * nOsc
    const kernelMatrix = (K/nOsc) * kernelMatrixGet(distanceMatrix)
end

const distributionSwitch = 3
const upperTimeBound = 100

const saveFlag = true
const plotFlag = true
const tsit5Flag = false 
const jacFlag = false
saveat = upperTimeBound/1000.

# PARAMETERS END 

##
if tsit5Flag
    const method = Tsit5()
else
    const method = AutoTsit5(Rosenbrock23()) 
    #const method = lsoda()
end

if jacFlag
    const jac = jac!
else
    const jac = nothing
end

# theta0, W are initial phase, intrinsic freq
Random.seed!(1234)
const theta0 = 2 * pi * rand(Float64, nOsc);

if distributionSwitch == 0
    const probabilityDistribution = Cauchy(0,1)
    const distributionTitle = string(probabilityDistribution)
elseif distributionSwitch == 1
    const probabilityDistribution = MixtureModel([Normal(-2.0, 1.0), Normal(2.0, 1.0)])
    const distributionTitle = string(probabilityDistribution.components) * "<br>" * string(probabilityDistribution.prior)
elseif distributionSwitch == 2
    const probabilityDistribution = Exponential(2.5)
    const distributionTitle = string(probabilityDistribution)
elseif distributionSwitch == 3
    const probabilityDistribution = MixtureModel([Exponential(2.5), Normal(5., .5), Normal(7., .5)], [.9,.05,.05])
    const distributionTitle = string(probabilityDistribution.components) * "<br>" * string(probabilityDistribution.prior)
end

const W = rand(probabilityDistribution, nOsc) 

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