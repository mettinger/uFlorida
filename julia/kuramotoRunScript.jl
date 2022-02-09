##
using Distributed
using Dates
using Random

const nProcs = 6
addprocs(nProcs)
@everywhere include("kuramoto.jl")

const nOsc = 1024
Random.seed!(1234)

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
    directory = "sphere/temp/Cauchy_0_1"
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
const pmapList = Base.product(kList, [theta0], [W], [description], [directory], [kernelMatrix])
println("Workers: " * string(nProcs))
println("Trials: " * string(length(pmapList)))
pmap(kuramotoPMAP, pmapList)
