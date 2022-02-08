##
using Distributed
using Dates

const nProcs = 6
addprocs(nProcs)
@everywhere include("kuramoto.jl")

const nOsc = 1024

##
# PARAMETERS TO SET

const kList = collect(1.:5.)
const kernelSwitch = 2
const distributionSwitch = 2
# PARAMETERS END

if distributionSwitch == 0
    probabilityDistribution = MixtureModel([Exponential(2.5), Normal(5., .5), Normal(7., .5)], [.9,.05,.05])
    directory = "Mixed"
    description = "Mixed"
elseif distributionSwitch == 1
    probabilityDistribution = Exponential(2.5)
    directory = "Exponential_2.5"
    description = "Exponential_2.5"
elseif distributionSwitch == 2
    probabilityDistribution = Cauchy(0,1)
    directory = "sphere/Cauchy_0_1"
    description = "Cauchy_0_1"
else
    println("Distribution error!")
    return
end

if kernelSwitch == 0
    const kernelMatrix = ones((nOsc, nOsc))
elseif kernelSwitch == 1
    const distanceMatrix = distanceMatrixGet(nOsc)
    const kernelMatrix = kernelMatrixGet(distanceMatrix)
elseif kernelSwitch == 2
    const distanceMatrix = Matrix{Float64}(DataFrame(CSV.File("../data/sphere/greatCircleDistance.csv", header=false)))
    const kernelMatrix = kernelMatrixGet(distanceMatrix)
end

theta0 = 2 * pi * rand(Float64, nOsc)
const W = rand(probabilityDistribution, nOsc) 
##

#pmapList = pmapListGet(kList, distributionList, distanceMatrixList)
const pmapList = Base.product(kList, [theta0], [W], [description], [directory], [kernelMatrix])
println("Workers: " * string(nProcs))
println("Trials: " * string(length(pmapList)))
pmap(kuramotoPMAP, pmapList)


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